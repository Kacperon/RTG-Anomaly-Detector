# modelv1/detector.py - Logika detekcji dla modelu v1
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any
import json
from datetime import datetime

class ModelV1Detector:
    """Klasa obsługująca detekcję anomalii przy użyciu modelu v1"""
    
    def __init__(self):
        self.model = None
        self.model_path = "modelv1/runs/detect/vehicle_anomaly/weights/best.pt"
        
    def load_model(self) -> Dict[str, Any]:
        """Ładuje model YOLO v1"""
        try:
            if not os.path.exists(self.model_path):
                return {
                    "success": False,
                    "error": f"Model nie istnieje w ścieżce: {self.model_path}"
                }
            
            self.model = YOLO(self.model_path)
            self.model.to('cpu')  # Force CPU usage
            
            return {
                "success": True,
                "message": f"Model v1 załadowany pomyślnie: {self.model_path}",
                "model_path": self.model_path,
                "model_type": "yolo_v1",
                "device": "cpu",
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
    
    def detect_anomalies(self, image_path: str) -> Dict[str, Any]:
        """
        Wykrywa anomalie na obrazie
        
        Args:
            image_path: Ścieżka do obrazu
            
        Returns:
            Słownik z wynikami detekcji
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
            image = Image.open(image_path).convert('L')
            image_np = np.array(image)
            
            # Uruchom predykcję YOLO
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
            
            # Przetwórz wyniki
            detections = []
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    
                    # Pobierz nazwę klasy
                    class_name = self.model.names[cls] if hasattr(self.model, 'names') and cls in self.model.names else "anomaly"
                    
                    detection = {
                        "id": i + 1,
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "confidence": round(conf, 3),
                        "class": class_name,
                        "class_id": cls,
                        "area": int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
                        "center": [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
                    }
                    detections.append(detection)
            
            # Przygotuj wynik
            result = {
                "success": True,
                "detections": detections,
                "detection_count": len(detections),
                "has_anomaly": len(detections) > 0,
                "image_shape": image_np.shape,
                "model_info": {
                    "model_path": self.model_path,
                    "model_type": "yolo_v1"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd podczas detekcji: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zwraca informacje o modelu"""
        return {
            "model_loaded": self.is_model_loaded(),
            "model_path": self.model_path,
            "model_type": "yolo_v1",
            "classes": self.model.names if self.model else None,
            "timestamp": datetime.now().isoformat()
        }

# Globalna instancja detektora
detector = ModelV1Detector()

def load_model() -> Dict[str, Any]:
    """Funkcja pomocnicza do ładowania modelu"""
    return detector.load_model()

def detect_anomalies(image_path: str) -> Dict[str, Any]:
    """Funkcja pomocnicza do wykrywania anomalii"""
    return detector.detect_anomalies(image_path)

def get_model_info() -> Dict[str, Any]:
    """Funkcja pomocnicza do pobierania info o modelu"""
    return detector.get_model_info()

def is_model_loaded() -> bool:
    """Funkcja pomocnicza sprawdzająca czy model jest załadowany"""
    return detector.is_model_loaded()
