# modelv2/detector.py - Logika detekcji dla modelu v2 z otoczkami wypukłymi
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any
import json
from datetime import datetime
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
