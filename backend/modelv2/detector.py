# modelv2/detector.py - Logika detekcji dla modelu v2 z otoczkami wypukłymi
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any
import json
from datetime import datetime
import glob
from .detection_bounds import VehicleBoundsDetector

class ModelV2Detector:
    """Klasa obsługująca detekcję anomalii z otoczkami wypukłymi przy użyciu modelu v2"""
    
    def __init__(self):
        self.model = None
        self.model_path = "yolov8n.pt"  # Używamy standardowego YOLO (nie dotrenowanego)
        self.bounds_processor = VehicleBoundsDetector()
        
    def load_model(self) -> Dict[str, Any]:
        """Ładuje model YOLO v2 z obsługą otoczek wypukłych"""
        try:
            # Załaduj standardowy model YOLO (zostanie pobrany automatycznie jeśli nie istnieje)
            self.model = YOLO(self.model_path)
            self.model.to('cpu')  # Force CPU usage
            
            # Załaduj model także w bounds_processor
            bounds_load_result = self.bounds_processor.load_model(self.model_path)
            if not bounds_load_result["success"]:
                return {
                    "success": False,
                    "error": f"Błąd ładowania modelu w bounds_processor: {bounds_load_result['error']}"
                }
            
            return {
                "success": True,
                "message": f"Model v2 załadowany pomyślnie: {self.model_path}",
                "model_path": self.model_path,
                "model_type": "yolo_v2_with_bounds",
                "device": "cpu",
                "features": ["convex_hull", "expanded_rectangle"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd ładowania modelu: {str(e)}"
            }
    
    def is_model_loaded(self) -> bool:
        """Sprawdza czy model jest załadowany"""
        return self.model is not None
    
    def detect_anomalies(self, image_path: str, include_bounds: bool = True) -> Dict[str, Any]:
        """
        Wykrywa największy obiekt na obrazie z otoczkami wypukłymi i rozszerzonymi prostokątami
        
        Args:
            image_path: Ścieżka do obrazu
            include_bounds: Czy uwzględnić otoczki wypukłe i rozszerzone prostokąty
            
        Returns:
            Słownik z wynikami detekcji zawierający także otoczki wypukłe
        """
        if not self.is_model_loaded():
            return {
                "success": False,
                "error": "Model nie jest załadowany. Użyj load_model() najpierw."
            }
        
        try:
            # Sprawdź czy plik istnieje
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Plik obrazu nie istnieje: {image_path}"
                }
            
            # Wczytaj obraz
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu: {image_path}"
                }
            
            # Uruchom predykcję YOLO - wykryj wszystkie obiekty
            results = self.model.predict(
                image_path,
                imgsz=640,
                conf=0.05,    # Niski próg pewności
                iou=0.3,      # IoU threshold
                max_det=300,  # Maksymalna liczba detekcji
                save=False,
                device='cpu',
                verbose=False
            )
            
            # Znajdź największy obiekt (według powierzchni)
            largest_detection = None
            largest_area = 0
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    
                    # Oblicz powierzchnię
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    
                    # Sprawdź czy to największy obiekt
                    if area > largest_area:
                        largest_area = area
                        # Pobierz nazwę klasy
                        class_name = self.model.names[cls] if hasattr(self.model, 'names') and cls in self.model.names else "object"
                        
                        largest_detection = {
                            "id": 1,
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "confidence": round(float(conf), 3),
                            "class": class_name,
                            "class_id": int(cls),
                            "area": int(area),
                            "center": [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
                        }
            
            detections = []
            if largest_detection:
                # Dodaj otoczki wypukłe i rozszerzone prostokąty jeśli wymagane
                if include_bounds:
                    bounds_result = self.bounds_processor.process_single_detection(
                        image, largest_detection["bbox"], expansion_factor=0.05
                    )
                    if bounds_result["success"]:
                        largest_detection["bounds"] = bounds_result["bounds"]
                    else:
                        largest_detection["bounds"] = {
                            "convex_hull": None,
                            "expanded_rectangle": None,
                            "error": bounds_result.get("error", "Unknown error")
                        }
                
                detections.append(largest_detection)
            
            # Przygotuj wynik
            result = {
                "success": True,
                "detections": detections,
                "detection_count": len(detections),
                "has_anomaly": len(detections) > 0,
                "image_shape": image.shape,
                "model_info": {
                    "model_path": self.model_path,
                    "model_type": "yolo_v2_with_bounds",
                    "features": ["convex_hull", "expanded_rectangle", "largest_object_only"] if include_bounds else ["largest_object_only"]
                },
                "bounds_included": include_bounds,
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd podczas detekcji: {str(e)}"
            }
    
    def detect_with_visualization(self, image_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Wykrywa anomalie i tworzy wizualizację z otoczkami wypukłymi
        
        Args:
            image_path: Ścieżka do obrazu wejściowego
            output_path: Ścieżka do zapisu wizualizacji (opcjonalnie)
            
        Returns:
            Słownik z wynikami detekcji i informacją o wizualizacji
        """
        # Najpierw uruchom standardową detekcję
        result = self.detect_anomalies(image_path, include_bounds=True)
        
        if not result["success"]:
            return result
        
        try:
            # Wczytaj obraz do wizualizacji
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu do wizualizacji: {image_path}"
                }
            
            # Narysuj wizualizacje dla każdej detekcji
            for detection in result["detections"]:
                # Narysuj podstawowy bounding box
                bbox = detection["bbox"]
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Dodaj etykietę
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                cv2.putText(image, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Narysuj otoczki jeśli dostępne
                if "bounds" in detection and detection["bounds"]:
                    bounds = detection["bounds"]
                    
                    # Otoczka wypukła (niebieski)
                    if bounds.get("convex_hull") is not None:
                        hull_points = np.array(bounds["convex_hull"], np.int32)
                        cv2.polylines(image, [hull_points], True, (255, 0, 0), 2)
                    
                    # Rozszerzony prostokąt (czerwony)
                    if bounds.get("expanded_rectangle") is not None:
                        exp_rect = bounds["expanded_rectangle"]
                        cv2.rectangle(image, (exp_rect[0], exp_rect[1]), 
                                    (exp_rect[2], exp_rect[3]), (0, 0, 255), 2)
            
            # Zapisz wizualizację jeśli podano ścieżkę
            if output_path:
                success = cv2.imwrite(output_path, image)
                if not success:
                    result["visualization_error"] = f"Nie można zapisać wizualizacji do: {output_path}"
                else:
                    result["visualization_path"] = output_path
            
            result["visualization_created"] = output_path is not None
            return result
            
        except Exception as e:
            result["visualization_error"] = f"Błąd podczas tworzenia wizualizacji: {str(e)}"
            return result
    
    def detect_and_center_object(self, image_path: str, output_size: tuple = (640, 640)) -> Dict[str, Any]:
        """
        Wykrywa największy obiekt i centruje go na obrazie
        
        Args:
            image_path: Ścieżka do obrazu wejściowego
            output_size: Rozmiar wyjściowy (width, height)
            
        Returns:
            Słownik z wynikami detekcji i wycentrowanym obrazem
        """
        # Najpierw uruchom standardową detekcję
        result = self.detect_anomalies(image_path, include_bounds=True)
        
        if not result["success"] or result["detection_count"] == 0:
            return result
        
        try:
            # Wczytaj obraz
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu: {image_path}"
                }
            
            # Pobierz największy obiekt
            main_detection = result["detections"][0]
            bbox = main_detection["bbox"]
            
            # Wycentruj obiekt
            center_result = self.bounds_processor.center_object_in_image(image, bbox, output_size)
            
            if center_result["success"]:
                # Nie dodajemy centered_image do wyników JSON (numpy array nie może być serializowany)
                # Obraz jest dostępny przez center_result["centered_image"] dla dalszego przetwarzania
                result["centered_bbox"] = center_result["new_bbox"]
                result["shift"] = center_result["shift"]
                result["output_size"] = output_size
                result["centering_successful"] = True
                
                # Zaktualizuj współrzędne otoczek dla wycentrowanego obrazu
                shift_x, shift_y = center_result["shift"]
                if "bounds" in main_detection and main_detection["bounds"]:
                    bounds = main_detection["bounds"]
                    
                    # Przesuń otoczkę wypukłą
                    if bounds.get("convex_hull"):
                        centered_hull = []
                        for point in bounds["convex_hull"]:
                            centered_hull.append([int(point[0] - shift_x), int(point[1] - shift_y)])
                        result["centered_convex_hull"] = centered_hull
                    
                    # Przesuń rozszerzony prostokąt
                    if bounds.get("expanded_rectangle"):
                        exp_rect = bounds["expanded_rectangle"]
                        result["centered_expanded_rectangle"] = [
                            int(exp_rect[0] - shift_x),
                            int(exp_rect[1] - shift_y),
                            int(exp_rect[2] - shift_x),
                            int(exp_rect[3] - shift_y)
                        ]
            else:
                result["center_error"] = center_result.get("error", "Unknown centering error")
            
            return result
            
        except Exception as e:
            result["center_error"] = f"Błąd podczas centrowania: {str(e)}"
            return result

    def _extract_object_region(self, image: np.ndarray, bbox: List[int], margin_percent: float = 0.1) -> np.ndarray:
        """
        Wycina region obiektu z obrazu z dodanym marginesem
        
        Args:
            image: Obraz jako numpy array
            bbox: Bounding box [x1, y1, x2, y2]
            margin_percent: Margines w procentach (0.1 = 10%)
            
        Returns:
            Wycięty fragment obrazu
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Oblicz wymiary obiektu
        obj_width = x2 - x1
        obj_height = y2 - y1
        
        # Dodaj margines
        margin_w = int(obj_width * margin_percent)
        margin_h = int(obj_height * margin_percent)
        
        # Nowe współrzędne z marginesem
        new_x1 = max(0, x1 - margin_w)
        new_y1 = max(0, y1 - margin_h)
        new_x2 = min(w, x2 + margin_w)
        new_y2 = min(h, y2 + margin_h)
        
        return image[new_y1:new_y2, new_x1:new_x2]

    def _calculate_color_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Oblicza różnicę kolorów między dwoma obrazami (MSE)
        
        Args:
            img1, img2: Obrazy jako numpy arrays
            
        Returns:
            Wartość różnicy (MSE)
        """
        if img1.shape != img2.shape:
            # Przeskaluj drugi obraz do rozmiaru pierwszego
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Konwertuj na float32 dla dokładności
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Oblicz MSE
        mse = np.mean((img1_f - img2_f) ** 2)
        return float(mse)

    def _find_best_matching_image(self, object_region: np.ndarray, processed_images_dir: str) -> Dict[str, Any]:
        """
        Znajduje najlepiej dopasowany obraz w folderze processed_images
        
        Args:
            object_region: Wycięty region obiektu
            processed_images_dir: Ścieżka do folderu z przetworzonymi obrazami
            
        Returns:
            Słownik z informacjami o najlepszym dopasowaniu
        """
        print(f"🔍 [DEBUG MATCHING] Searching in: {processed_images_dir}")
        print(f"📁 [DEBUG MATCHING] Directory exists: {os.path.exists(processed_images_dir)}")
        
        if not os.path.exists(processed_images_dir):
            print(f"❌ [DEBUG MATCHING] Directory does not exist")
            return {
                "success": False,
                "error": f"Folder nie istnieje: {processed_images_dir}"
            }
        
        # Znajdź wszystkie obrazy w folderze
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            pattern1 = os.path.join(processed_images_dir, ext)
            pattern2 = os.path.join(processed_images_dir, ext.upper())
            files1 = glob.glob(pattern1)
            files2 = glob.glob(pattern2)
            image_files.extend(files1)
            image_files.extend(files2)
            print(f"📋 [DEBUG MATCHING] Pattern {ext}: found {len(files1)} files")
            print(f"📋 [DEBUG MATCHING] Pattern {ext.upper()}: found {len(files2)} files")
        
        print(f"📊 [DEBUG MATCHING] Total image files found: {len(image_files)}")
        if image_files:
            print(f"📋 [DEBUG MATCHING] Files: {[os.path.basename(f) for f in image_files]}")
        
        if not image_files:
            return {
                "success": False,
                "error": f"Brak obrazów w folderze: {processed_images_dir}"
            }
        
        best_match = None
        best_difference = float('inf')
        comparison_results = []
        
        print(f"🔄 [DEBUG MATCHING] Comparing with {len(image_files)} candidates...")
        
        for i, img_path in enumerate(image_files):
            print(f"📂 [DEBUG MATCHING] Processing {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
            try:
                # Wczytaj obraz
                candidate_img = cv2.imread(img_path)
                if candidate_img is None:
                    print(f"❌ [DEBUG MATCHING] Failed to load image: {img_path}")
                    continue
                
                print(f"✅ [DEBUG MATCHING] Loaded candidate: {candidate_img.shape}")
                print(f"🖼️  [DEBUG MATCHING] Object region shape: {object_region.shape}")
                
                # Oblicz różnicę kolorów
                difference = self._calculate_color_difference(object_region, candidate_img)
                print(f"📈 [DEBUG MATCHING] Calculated difference: {difference}")
                
                comparison_results.append({
                    "path": img_path,
                    "difference": difference,
                    "filename": os.path.basename(img_path)
                })
                
                # Sprawdź czy to najlepsze dopasowanie
                if difference < best_difference:
                    print(f"🎯 [DEBUG MATCHING] New best match! Difference: {difference} < {best_difference}")
                    best_difference = difference
                    best_match = img_path
                    
            except Exception as e:
                print(f"❌ [DEBUG MATCHING] Error processing {img_path}: {str(e)}")
                comparison_results.append({
                    "path": img_path,
                    "difference": None,
                    "error": str(e),
                    "filename": os.path.basename(img_path)
                })
        
        if best_match is None:
            print("❌ [DEBUG MATCHING] No valid matches found")
            return {
                "success": False,
                "error": "Nie znaleziono żadnego pasującego obrazu",
                "comparison_results": comparison_results
            }
        
        print(f"🏆 [DEBUG MATCHING] Best match: {os.path.basename(best_match)} with difference: {best_difference}")
        
        return {
            "success": True,
            "best_match_path": best_match,
            "best_difference": best_difference,
            "comparison_results": sorted(comparison_results, key=lambda x: x.get("difference", float('inf'))),
            "total_candidates": len(image_files)
        }

    def _create_color_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Tworzy mapę różnic kolorów gdzie czerwony oznacza największą różnicę
        
        Args:
            img1: Pierwszy obraz (region obiektu)
            img2: Drugi obraz (najlepsze dopasowanie)
            
        Returns:
            Mapa różnic jako obraz BGR
        """
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Konwertuj na float32
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Oblicz różnicę dla każdego piksela (norma euclidesowa w przestrzeni RGB)
        diff = np.sqrt(np.sum((img1_f - img2_f) ** 2, axis=2))
        
        # Normalizuj do zakresu 0-255
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Utwórz mapę kolorów: czerwony dla największych różnic
        colormap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
        
        # Opcjonalnie można zmienić mapę kolorów tak, by czerwony był dla max różnic:
        # Odwróć kolejność kolorów
        colormap = cv2.applyColorMap(255 - diff_normalized, cv2.COLORMAP_JET)
        
        return colormap

    def _center_crop(self, image_file):
        """
        Detectuj obiekty, znajdź największy box, powiększ go o 10%,
        i przytnij obraz do tego obszaru.
        """
        def expand_bbox(bbox_coords, img_shape, expansion_factor=0.1):
            H, W = img_shape[:2]
            x1, y1, x2, y2 = bbox_coords
            
            # Oblicz szerokość i wysokość boxa
            width = x2 - x1
            height = y2 - y1
            
            # Oblicz powiększenie
            width_expansion = width * expansion_factor / 2
            height_expansion = height * expansion_factor / 2
            
            # Nowe współrzędne
            new_x1 = max(0, int(x1 - width_expansion))
            new_y1 = max(0, int(y1 - height_expansion))
            new_x2 = min(W, int(x2 + width_expansion))
            new_y2 = min(H, int(y2 + height_expansion))
            
            return (new_x1, new_y1, new_x2, new_y2)
        
        def crop_to_bbox(img, bbox_coords):
            x1, y1, x2, y2 = bbox_coords
            
            H, W = img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            # Wytnij obszar
            cropped_img = img[y1:y2, x1:x2]
            return cropped_img
        
        # Uruchom detekcję YOLO
        results = self.model.predict(image_file,
                                imgsz=640,
                                conf=0.03,
                                iou=0.3,
                                max_det=300,
                                save=False,
                                device='cpu',
                                verbose=False)
        
        # Wczytaj obraz
        img = cv2.imread(image_file)
        if img is None:
            return None
        
        largest_box = None
        max_area = 0
        
        # Znajdź największy bounding box
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > max_area:
                        max_area = area
                        largest_box = (x1, y1, x2, y2)
                        
        if largest_box is not None:
            # Powiększ bounding box o 10%
            expanded_box = expand_bbox(largest_box, img.shape, expansion_factor=0.1)
            
            # Przytnij obraz do powiększonego bounding boxa
            cropped_img = crop_to_bbox(img, expanded_box)
            
            return cropped_img
        else:
            return None
            
    def find_and_compare_with_processed_images(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Główna metoda: wykrywa obiekt, używa center_crop do wycinania i porównuje 
        z folderem processed_images, tworząc mapę różnic kolorów
        
        Args:
            image_path: Ścieżka do obrazu wejściowego
            output_dir: Folder do zapisu wyników (opcjonalnie)
            
        Returns:
            Słownik z wynikami porównania i mapą różnic
        """
        print(f"🔍 [DEBUG DETECTOR] Starting analysis for: {image_path}")
        
        try:
            # Użyj _center_crop do wykrycia i wycięcia największego obiektu
            print("🎯 [DEBUG DETECTOR] Running center crop detection...")
            object_region = self._center_crop(image_path)
            
            if object_region is None:
                print("❌ [DEBUG DETECTOR] No objects detected by center crop")
                return {
                    "success": False,
                    "error": "Brak wykrytych obiektów - center crop nie znalazł żadnego obiektu"
                }
            
            print(f"✅ [DEBUG DETECTOR] Object region extracted via center crop: {object_region.shape}")
            
            # Dla kompatybilności z resztą kodu, uruchom też standardową detekcję w celu uzyskania metadanych
            print("📊 [DEBUG DETECTOR] Running standard detection for metadata...")
            detection_result = self.detect_anomalies(image_path, include_bounds=True)
            
            if detection_result["success"] and detection_result["detection_count"] > 0:
                main_detection = detection_result["detections"][0]
                bbox = main_detection["bbox"]
                print(f"📦 [DEBUG DETECTOR] Main object bbox (metadata): {bbox}")
            else:
                # Fallback - stwórz podstawowe metadane
                print("⚠️ [DEBUG DETECTOR] Standard detection failed, using fallback metadata")
                detection_result = {
                    "success": True,
                    "detection_count": 1,
                    "detections": [{
                        "id": 1,
                        "bbox": [0, 0, object_region.shape[1], object_region.shape[0]],
                        "confidence": 0.5,
                        "class": "object",
                        "area": object_region.shape[0] * object_region.shape[1]
                    }]
                }
                bbox = detection_result["detections"][0]["bbox"]
            
            # Określ ścieżkę do folderu processed_images
            current_dir = os.path.dirname(os.path.abspath(__file__))
            processed_images_dir = os.path.join(current_dir, "..", "..", "data-processing/processed_clean_data")
            processed_images_dir = os.path.normpath(processed_images_dir)
            print(f"📁 [DEBUG DETECTOR] Processed images dir: {processed_images_dir}")
            print(f"📁 [DEBUG DETECTOR] Directory exists: {os.path.exists(processed_images_dir)}")
            
            if os.path.exists(processed_images_dir):
                files = os.listdir(processed_images_dir)
                print(f"📋 [DEBUG DETECTOR] Files in directory: {files}")
            
            # Znajdź najlepiej dopasowany obraz
            print("🔍 [DEBUG DETECTOR] Finding best matching image...")
            matching_result = self._find_best_matching_image(object_region, processed_images_dir)
            print(f"🎯 [DEBUG DETECTOR] Matching result: success={matching_result.get('success', False)}")
            
            if matching_result.get("error"):
                print(f"❌ [DEBUG DETECTOR] Matching error: {matching_result['error']}")
            
            if not matching_result["success"]:
                return {
                    "success": False,
                    "error": matching_result["error"],
                    "detection_result": detection_result,
                    "object_region_shape": object_region.shape,
                    "processed_images_dir": processed_images_dir
                }
            
            print(f"✅ [DEBUG DETECTOR] Best match found: {matching_result['best_match_path']}")
            print(f"📈 [DEBUG DETECTOR] Match difference (MSE): {matching_result['best_difference']}")
            
            # Wczytaj najlepiej dopasowany obraz
            print("📂 [DEBUG DETECTOR] Loading best match image...")
            best_match_img = cv2.imread(matching_result["best_match_path"])
            if best_match_img is None:
                print("❌ [DEBUG DETECTOR] Failed to load best match image")
                return {
                    "success": False,
                    "error": f"Nie można wczytać najlepiej dopasowanego obrazu: {matching_result['best_match_path']}"
                }
            
            print(f"✅ [DEBUG DETECTOR] Best match loaded: {best_match_img.shape}")
            
            # Utwórz mapę różnic kolorów
            print("🔥 [DEBUG DETECTOR] Creating color difference map...")
            color_diff_map = self._create_color_difference_map(object_region, best_match_img)
            print(f"🎨 [DEBUG DETECTOR] Color diff map created: {color_diff_map.shape}")
            
            # Przygotuj wyniki
            result = {
                "success": True,
                "detection_result": detection_result,
                "object_bbox": bbox,
                "object_region_shape": object_region.shape,
                "matching_result": matching_result,
                "best_match_path": matching_result["best_match_path"],
                "best_match_difference": matching_result["best_difference"],
                "color_diff_map_shape": color_diff_map.shape,
                "processed_images_dir": processed_images_dir,
                "center_crop_used": True,  # Nowa flaga wskazująca użycie center_crop
                "timestamp": datetime.now().isoformat()
            }
            
            # Zapisz wyniki jeśli podano folder wyjściowy
            if output_dir and os.path.exists(output_dir):
                print(f"💾 [DEBUG DETECTOR] Saving results to: {output_dir}")
                try:
                    # Zapisz wycięty region obiektu (z center_crop)
                    object_region_path = os.path.join(output_dir, f"center_crop_region_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    success1 = cv2.imwrite(object_region_path, object_region)
                    print(f"📁 [DEBUG DETECTOR] Center crop region saved: {success1} -> {object_region_path}")
                    if success1:
                        result["object_region_saved"] = object_region_path
                    
                    # Zapisz mapę różnic
                    diff_map_path = os.path.join(output_dir, f"color_diff_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    success2 = cv2.imwrite(diff_map_path, color_diff_map)
                    print(f"🔥 [DEBUG DETECTOR] Diff map saved: {success2} -> {diff_map_path}")
                    if success2:
                        result["color_diff_map_saved"] = diff_map_path
                    
                    # Zapisz porównanie side-by-side
                    # Przeskaluj obrazy do tego samego rozmiaru
                    h, w = object_region.shape[:2]
                    best_match_resized = cv2.resize(best_match_img, (w, h))
                    comparison = np.hstack([object_region, best_match_resized, color_diff_map])
                    
                    comparison_path = os.path.join(output_dir, f"center_crop_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    success3 = cv2.imwrite(comparison_path, comparison)
                    print(f"📊 [DEBUG DETECTOR] Center crop comparison saved: {success3} -> {comparison_path}")
                    if success3:
                        result["comparison_saved"] = comparison_path
                    
                except Exception as e:
                    print(f"❌ [DEBUG DETECTOR] Save error: {str(e)}")
                    result["save_error"] = f"Błąd podczas zapisu: {str(e)}"
            else:
                print(f"⚠️  [DEBUG DETECTOR] Output dir not provided or doesn't exist: {output_dir}")
            
            print("🎉 [DEBUG DETECTOR] Center crop analysis completed successfully")
            return result
            
        except Exception as e:
            print(f"❌ [DEBUG DETECTOR] Exception during center crop analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Błąd podczas porównywania z center crop: {str(e)}",
                "center_crop_used": True
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu"""
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": self.model_path,
            "model_type": "yolo_v2_with_bounds",
            "features": ["convex_hull", "expanded_rectangle", "visualization"],
            "classes": self.model.names if self.model else None,
            "bounds_processor": "DetectionBoundsProcessor",
            "timestamp": datetime.now().isoformat()
        }

# Globalna instancja detektora
detector = ModelV2Detector()

def load_model() -> Dict[str, Any]:
    """Funkcja pomocnicza do ładowania modelu"""
    return detector.load_model()

def detect_anomalies(image_path: str, include_bounds: bool = True) -> Dict[str, Any]:
    """Funkcja pomocnicza do wykrywania anomalii z otoczkami wypukłymi"""
    return detector.detect_anomalies(image_path, include_bounds)

def detect_with_visualization(image_path: str, output_path: str = None) -> Dict[str, Any]:
    """Funkcja pomocnicza do wykrywania z wizualizacją"""
    return detector.detect_with_visualization(image_path, output_path)

def detect_and_center_object(image_path: str, output_size: tuple = (640, 640)) -> Dict[str, Any]:
    """Funkcja pomocnicza do wykrywania i centrowania największego obiektu"""
    return detector.detect_and_center_object(image_path, output_size)

def get_model_info() -> Dict[str, Any]:
    """Funkcja pomocnicza do pobierania info o modelu"""
    return detector.get_model_info()

def is_model_loaded() -> bool:
    """Funkcja pomocnicza sprawdzająca czy model jest załadowany"""
    return detector.is_model_loaded()

def find_and_compare_with_processed_images(image_path: str, output_dir: str = None) -> Dict[str, Any]:
    """Funkcja pomocnicza do znajdowania i porównywania z przetworzonymi obrazami"""
    return detector.find_and_compare_with_processed_images(image_path, output_dir)
