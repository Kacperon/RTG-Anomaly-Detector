# modelv2/detection_bounds.py - Wykrywanie otoczki wypukłej i rozszerzonych granic
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from PIL import Image
from ultralytics import YOLO

class VehicleBoundsDetector:
    """Klasa do wykrywania otoczek wypukłych i rozszerzonych granic pojazdów"""
    
    def __init__(self):
        self.model = None
        # Używamy domyślnego modelu YOLO dla wykrywania pojazdów
        self.model_path = None
        
    def load_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ładuje model YOLO dla wykrywania pojazdów
        
        Args:
            model_path: Ścieżka do modelu (domyślnie: yolov8n.pt)
        """
        try:
            if model_path:
                self.model_path = model_path
            else:
                # Użyj standardowego modelu YOLOv8 (nie dotrenowanego)
                self.model_path = "yolov8n.pt"  # Zostanie automatycznie pobrany jeśli nie istnieje
            
            self.model = YOLO(self.model_path)
            self.model.to('cpu')
            
            return {
                "success": True,
                "message": f"Model załadowany pomyślnie: {self.model_path}",
                "model_path": self.model_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd ładowania modelu: {str(e)}"
            }
    
    def detect_vehicles(self, image_path: str, conf_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Wykrywa wszystkie obiekty na obrazie przy użyciu YOLO (nie tylko pojazdy)
        
        Args:
            image_path: Ścieżka do obrazu
            conf_threshold: Próg pewności dla detekcji
            
        Returns:
            Słownik z wykrytymi obiektami
        """
        if not self.model:
            return {
                "success": False,
                "error": "Model nie jest załadowany. Użyj load_model() najpierw."
            }
        
        try:
            # Uruchom predykcję dla wszystkich klas (nie ograniczaj do pojazdów)
            results = self.model.predict(
                image_path,
                imgsz=640,
                conf=conf_threshold,
                verbose=False
            )
            
            # Wczytaj obraz do analizy
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu: {image_path}"
                }
            
            height, width = image.shape[:2]
            objects = []
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Pobierz nazwę klasy z modelu
                    class_name = self.model.names.get(cls, f"class_{cls}") if hasattr(self.model, 'names') else f"object_{cls}"
                    
                    obj = {
                        "id": i + 1,
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "confidence": round(conf, 3),
                        "class": class_name,
                        "class_id": cls
                    }
                    objects.append(obj)
            
            return {
                "success": True,
                "vehicles": objects,  # Zachowujemy nazwę "vehicles" dla kompatybilności z resztą kodu
                "vehicle_count": len(objects),
                "image_shape": [height, width],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd podczas wykrywania pojazdów: {str(e)}"
            }
    
    def get_convex_hull(self, image_path: str, bbox: List[int]) -> Dict[str, Any]:
        """
        Oblicza otoczkę wypukłą obszaru pojazdu
        
        Args:
            image_path: Ścieżka do obrazu
            bbox: Lista współrzędnych [x1, y1, x2, y2]
            
        Returns:
            Słownik z punktami otoczki wypukłej
        """
        try:
            # Wczytaj obraz
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu: {image_path}"
                }
            
            x1, y1, x2, y2 = bbox
            
            # Wytnij region pojazdu
            vehicle_region = image[y1:y2, x1:x2]
            
            # Konwersja do skali szarości
            gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
            
            # Wykrycie krawędzi
            edges = cv2.Canny(gray, 50, 150)
            
            # Znajdź kontury
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {
                    "success": False,
                    "error": "Nie znaleziono konturów w regionie pojazdu"
                }
            
            # Połącz wszystkie kontury
            all_points = np.vstack(contours)
            
            # Oblicz otoczkę wypukłą
            hull = cv2.convexHull(all_points)
            
            # Przelicz współrzędne względem całego obrazu
            hull_global = hull.copy()
            hull_global[:, 0, 0] += x1  # dodaj offset x
            hull_global[:, 0, 1] += y1  # dodaj offset y
            
            # Konwertuj do listy punktów
            hull_points = hull_global.reshape(-1, 2).tolist()
            
            return {
                "success": True,
                "convex_hull": hull_points,
                "hull_area": cv2.contourArea(hull),
                "original_bbox": bbox,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd obliczania otoczki wypukłej: {str(e)}"
            }
    
    def get_expanded_bbox(self, bbox: List[int], expansion_factor: float = 0.05, image_shape: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Oblicza prostokąt rozszerzony o określony procent
        
        Args:
            bbox: Lista współrzędnych [x1, y1, x2, y2]
            expansion_factor: Współczynnik rozszerzenia (domyślnie 0.05 = 5%)
            image_shape: Kształt obrazu (height, width) do ograniczenia granic
            
        Returns:
            Słownik z rozszerzonym prostokątem
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Oblicz wymiary
            width = x2 - x1
            height = y2 - y1
            
            # Oblicz rozszerzenie
            expand_x = int(width * expansion_factor)
            expand_y = int(height * expansion_factor)
            
            # Nowe współrzędne
            new_x1 = int(x1 - expand_x)
            new_y1 = int(y1 - expand_y)
            new_x2 = int(x2 + expand_x)
            new_y2 = int(y2 + expand_y)
            
            # Ograniczenia do granic obrazu
            if image_shape:
                img_height, img_width = image_shape
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(img_width, new_x2)
                new_y2 = min(img_height, new_y2)
            
            expanded_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
            
            # Oblicz statystyki
            original_area = int(width * height)
            new_width = new_x2 - new_x1
            new_height = new_y2 - new_y1
            expanded_area = int(new_width * new_height)
            area_increase = ((expanded_area - original_area) / original_area) * 100
            
            return {
                "success": True,
                "original_bbox": bbox,
                "expanded_bbox": expanded_bbox,
                "expansion_factor": expansion_factor,
                "original_area": original_area,
                "expanded_area": expanded_area,
                "area_increase_percent": round(area_increase, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd obliczania rozszerzonego prostokąta: {str(e)}"
            }
    
    def analyze_vehicle_bounds(self, image_path: str, conf_threshold: float = 0.5, expansion_factor: float = 0.05) -> Dict[str, Any]:
        """
        Kompleksowa analiza granic pojazdów:
        1. Wykrywa pojazdy przy użyciu YOLO
        2. Oblicza otoczki wypukłe
        3. Tworzy rozszerzone prostokąty
        
        Args:
            image_path: Ścieżka do obrazu
            conf_threshold: Próg pewności dla YOLO
            expansion_factor: Współczynnik rozszerzenia prostokątów
            
        Returns:
            Słownik z kompleksową analizą
        """
        try:
            # Wykryj pojazdy
            vehicles_result = self.detect_vehicles(image_path, conf_threshold)
            if not vehicles_result["success"]:
                return vehicles_result
            
            vehicles = vehicles_result["vehicles"]
            image_shape = vehicles_result["image_shape"]
            
            analyzed_vehicles = []
            
            for vehicle in vehicles:
                vehicle_analysis = {
                    "vehicle_info": vehicle,
                    "bounds_analysis": {}
                }
                
                bbox = vehicle["bbox"]
                
                # Oblicz otoczkę wypukłą
                hull_result = self.get_convex_hull(image_path, bbox)
                vehicle_analysis["bounds_analysis"]["convex_hull"] = hull_result
                
                # Oblicz rozszerzony prostokąt
                expanded_result = self.get_expanded_bbox(bbox, expansion_factor, image_shape)
                vehicle_analysis["bounds_analysis"]["expanded_bbox"] = expanded_result
                
                analyzed_vehicles.append(vehicle_analysis)
            
            return {
                "success": True,
                "analyzed_vehicles": analyzed_vehicles,
                "total_vehicles": len(analyzed_vehicles),
                "image_path": image_path,
                "image_shape": image_shape,
                "analysis_settings": {
                    "confidence_threshold": conf_threshold,
                    "expansion_factor": expansion_factor
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd podczas analizy granic pojazdów: {str(e)}"
            }
    
    def visualize_bounds(self, image_path: str, analysis_result: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """
        Wizualizuje wyniki analizy granic na obrazie
        
        Args:
            image_path: Ścieżka do obrazu źródłowego
            analysis_result: Wynik analizy z analyze_vehicle_bounds
            output_path: Ścieżka do zapisania wyniku (opcjonalne)
            
        Returns:
            Informacje o wizualizacji
        """
        try:
            if not analysis_result["success"]:
                return {
                    "success": False,
                    "error": "Analiza nie powiodła się"
                }
            
            # Wczytaj obraz
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": f"Nie można wczytać obrazu: {image_path}"
                }
            
            # Kolory
            colors = {
                "original": (0, 255, 0),      # Zielony - oryginalne bbox
                "expanded": (255, 0, 0),      # Niebieski - rozszerzony bbox
                "convex_hull": (0, 0, 255)    # Czerwony - otoczka wypukła
            }
            
            for vehicle_analysis in analysis_result["analyzed_vehicles"]:
                vehicle_info = vehicle_analysis["vehicle_info"]
                bounds = vehicle_analysis["bounds_analysis"]
                
                # Rysuj oryginalne bbox
                bbox = vehicle_info["bbox"]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), colors["original"], 2)
                
                # Dodaj label
                label = f"{vehicle_info['class']} ({vehicle_info['confidence']:.2f})"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["original"], 2)
                
                # Rysuj rozszerzony bbox
                if bounds["expanded_bbox"]["success"]:
                    exp_bbox = bounds["expanded_bbox"]["expanded_bbox"]
                    exp_x1, exp_y1, exp_x2, exp_y2 = exp_bbox
                    cv2.rectangle(image, (exp_x1, exp_y1), (exp_x2, exp_y2), colors["expanded"], 2)
                    
                    # Label dla rozszerzonego
                    exp_label = f"Expanded +{bounds['expanded_bbox']['expansion_factor']*100:.0f}%"
                    cv2.putText(image, exp_label, (exp_x1, exp_y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["expanded"], 2)
                
                # Rysuj otoczkę wypukłą
                if bounds["convex_hull"]["success"]:
                    hull_points = np.array(bounds["convex_hull"]["convex_hull"], dtype=np.int32)
                    cv2.drawContours(image, [hull_points], -1, colors["convex_hull"], 2)
            
            # Zapisz obraz jeśli podano ścieżkę
            if output_path:
                cv2.imwrite(output_path, image)
                return {
                    "success": True,
                    "output_path": output_path,
                    "vehicles_visualized": len(analysis_result["analyzed_vehicles"])
                }
            else:
                return {
                    "success": True,
                    "image_array": image,
                    "vehicles_visualized": len(analysis_result["analyzed_vehicles"])
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd wizualizacji: {str(e)}"
            }
    
    def process_single_detection(self, image_path: str, bbox: List[int], expansion_factor: float = 0.05) -> Dict[str, Any]:
        """
        Przetwarza pojedynczą detekcję - oblicza otoczkę wypukłą i rozszerzony prostokąt
        
        Args:
            image_path: Ścieżka do obrazu lub obiekt numpy array
            bbox: Lista współrzędnych [x1, y1, x2, y2]
            expansion_factor: Współczynnik rozszerzenia prostokąta
            
        Returns:
            Słownik z wynikami przetwarzania
        """
        try:
            # Jeśli image_path to numpy array (cv2 image)
            if isinstance(image_path, np.ndarray):
                image = image_path
                image_shape = image.shape[:2]  # (height, width)
            else:
                # Wczytaj obraz z pliku
                image = cv2.imread(image_path)
                if image is None:
                    return {
                        "success": False,
                        "error": f"Nie można wczytać obrazu: {image_path}"
                    }
                image_shape = image.shape[:2]
            
            # Oblicz otoczkę wypukłą dla jakiegokolwiek obiektu
            hull_result = self._get_convex_hull_for_any_object(image, bbox)
            
            # Oblicz rozszerzony prostokąt
            expanded_result = self.get_expanded_bbox(bbox, expansion_factor, image_shape)
            
            if hull_result["success"] and expanded_result["success"]:
                return {
                    "success": True,
                    "bounds": {
                        "convex_hull": hull_result["convex_hull"],
                        "expanded_rectangle": expanded_result["expanded_bbox"]
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Hull: {hull_result.get('error', 'OK')}, Expanded: {expanded_result.get('error', 'OK')}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd przetwarzania detekcji: {str(e)}"
            }
    
    def _get_convex_hull_for_any_object(self, image: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """
        Pomocnicza metoda do obliczania otoczki wypukłej dla jakiegokolwiek obiektu
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Wytnij region obiektu
            object_region = image[y1:y2, x1:x2]
            
            if object_region.size == 0:
                return {
                    "success": False,
                    "error": "Pusty region obiektu"
                }
            
            # Konwertuj na skale szarości
            if len(object_region.shape) == 3:
                gray = cv2.cvtColor(object_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = object_region
            
            # Zastosuj kilka różnych metod segmentacji
            # Metoda 1: Threshold Otsu
            _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Metoda 2: Adaptacyjny threshold
            binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Metoda 3: Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Spróbuj każdej metody po kolei
            for binary in [binary1, binary2, edges]:
                # Operacje morfologiczne aby wypełnić dziury
                kernel = np.ones((3,3), np.uint8)
                binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                
                # Znajdź kontury
                contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Znajdź największy kontur
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Sprawdź czy kontur ma sensowny rozmiar (przynajmniej 5% obszaru)
                    contour_area = cv2.contourArea(largest_contour)
                    region_area = object_region.shape[0] * object_region.shape[1]
                    
                    if contour_area > region_area * 0.05:  # Przynajmniej 5% obszaru
                        # Oblicz otoczkę wypukłą
                        hull = cv2.convexHull(largest_contour)
                        
                        # Przesuń współrzędne z powrotem do oryginalnego układu
                        hull_points = []
                        for point in hull:
                            x, y = point[0]
                            hull_points.append([int(x + x1), int(y + y1)])
                        
                        return {
                            "success": True,
                            "convex_hull": hull_points
                        }
            
            # Jeśli żadna metoda nie znalazła konturów, użyj prostokąta jako otoczki
            hull_points = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]
            
            return {
                "success": True,
                "convex_hull": hull_points
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd obliczania otoczki wypukłej: {str(e)}"
            }
    
    def center_object_in_image(self, image: np.ndarray, bbox: List[int], output_size: tuple = (640, 640)) -> Dict[str, Any]:
        """
        Centruje obiekt na obrazie i przycina go do zadanego rozmiaru
        
        Args:
            image: Obraz numpy array
            bbox: Lista współrzędnych [x1, y1, x2, y2]
            output_size: Rozmiar wyjściowy (width, height)
            
        Returns:
            Słownik z wycentrowanym obrazem i nowymi współrzędnymi
        """
        try:
            x1, y1, x2, y2 = bbox
            obj_width = x2 - x1
            obj_height = y2 - y1
            obj_center_x = x1 + obj_width // 2
            obj_center_y = y1 + obj_height // 2
            
            img_height, img_width = image.shape[:2]
            output_width, output_height = output_size
            
            # Oblicz nowe współrzędne tak aby obiekt był na środku
            new_x1 = output_width // 2 - obj_width // 2
            new_y1 = output_height // 2 - obj_height // 2
            new_x2 = new_x1 + obj_width
            new_y2 = new_y1 + obj_height
            
            # Oblicz przesunięcie kamery
            shift_x = obj_center_x - output_width // 2
            shift_y = obj_center_y - output_height // 2
            
            # Oblicz obszar do wycinania z oryginalnego obrazu
            crop_x1 = max(0, shift_x)
            crop_y1 = max(0, shift_y)
            crop_x2 = min(img_width, shift_x + output_width)
            crop_y2 = min(img_height, shift_y + output_height)
            
            # Wytnij fragment
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Utwórz nowy obraz o zadanym rozmiarze i wypełnij go czernią
            centered_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Oblicz pozycję wklejenia
            paste_x = max(0, -shift_x)
            paste_y = max(0, -shift_y)
            paste_w = cropped.shape[1]
            paste_h = cropped.shape[0]
            
            # Wklej obraz
            centered_image[paste_y:paste_y + paste_h, paste_x:paste_x + paste_w] = cropped
            
            # Zaktualizuj współrzędne bounding box
            adjusted_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
            
            return {
                "success": True,
                "centered_image": centered_image,
                "new_bbox": adjusted_bbox,
                "shift": [int(shift_x), int(shift_y)]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd centrowania obiektu: {str(e)}"
            }

# Globalna instancja detektora
bounds_detector = VehicleBoundsDetector()

def load_model(model_path: Optional[str] = None) -> Dict[str, Any]:
    """Funkcja pomocnicza do ładowania modelu"""
    return bounds_detector.load_model(model_path)

def detect_vehicles(image_path: str, conf_threshold: float = 0.5) -> Dict[str, Any]:
    """Funkcja pomocnicza do wykrywania pojazdów"""
    return bounds_detector.detect_vehicles(image_path, conf_threshold)

def analyze_vehicle_bounds(image_path: str, conf_threshold: float = 0.5, expansion_factor: float = 0.05) -> Dict[str, Any]:
    """Funkcja pomocnicza do analizy granic pojazdów"""
    return bounds_detector.analyze_vehicle_bounds(image_path, conf_threshold, expansion_factor)

def visualize_bounds(image_path: str, analysis_result: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
    """Funkcja pomocnicza do wizualizacji wyników"""
    return bounds_detector.visualize_bounds(image_path, analysis_result, output_path)
