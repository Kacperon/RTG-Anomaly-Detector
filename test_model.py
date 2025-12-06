#!/usr/bin/env python3
"""
Test modelu Vehicle Anomaly Detection
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json

def test_model_loading():
    """Test loading różnych modeli"""
    print("🔧 Testowanie ładowania modeli...")
    
    models_to_test = [
        "runs/detect/vehicle_anomaly2/weights/best.pt",
        "runs/detect/vehicle_anomaly/weights/best.pt", 
        "yolov8n.pt",
        "yolov8s.pt"
    ]
    
    working_models = []
    
    for model_path in models_to_test:
        try:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                print(f"✅ {model_path} - ładowanie OK")
                working_models.append(model_path)
            else:
                print(f"❌ {model_path} - plik nie istnieje")
        except Exception as e:
            print(f"❌ {model_path} - błąd: {e}")
    
    return working_models

def test_model_inference(model_path, test_image_path=None):
    """Test inferencji modelu"""
    print(f"\n🔍 Testowanie inferencji dla: {model_path}")
    
    try:
        # Load model with CPU device
        model = YOLO(model_path)
        model.to('cpu')  # Force CPU usage
        print(f"✅ Model załadowany na CPU: {model_path}")
        
        # Test z przykładowym obrazem
        if test_image_path is None:
            # Utwórz test image
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_image_path = "temp_test_image.jpg"
            cv2.imwrite(test_image_path, test_image)
            print("📝 Utworzono tymczasowy obraz testowy")
        
        # Run prediction with CPU device
        results = model.predict(
            test_image_path,
            imgsz=640,
            conf=0.15,
            iou=0.45,
            save=False,
            verbose=False,
            device='cpu'  # Force CPU
        )
        
        print(f"✅ Predykcja wykonana")
        print(f"📊 Liczba wykrytych obiektów: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        # Test z rzeczywistymi danymi z data/
        real_test_images = []
        for root, dirs, files in os.walk("data"):
            for file in files[:3]:  # Test tylko 3 obrazy
                if file.lower().endswith(('.bmp', '.jpg', '.png')):
                    real_test_images.append(os.path.join(root, file))
        
        if real_test_images:
            print(f"\n📸 Testowanie z rzeczywistymi obrazami:")
            for img_path in real_test_images:
                try:
                    results = model.predict(img_path, conf=0.15, save=False, verbose=False, device='cpu')
                    detections = len(results[0].boxes) if results[0].boxes is not None else 0
                    print(f"  {os.path.basename(img_path)}: {detections} wykryć")
                except Exception as e:
                    print(f"  {os.path.basename(img_path)}: błąd - {e}")
        
        # Clean up temp file
        if test_image_path == "temp_test_image.jpg" and os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd inferencji: {e}")
        return False

def test_api_compatibility():
    """Test kompatybilności z API"""
    print("\n🌐 Testowanie kompatybilności API...")
    
    try:
        # Import Flask app components
        sys.path.append('.')
        
        # Test model loading jak w app.py
        MODEL_PATH = "runs/detect/vehicle_anomaly2/weights/best.pt"
        FALLBACK_MODEL = "yolov8n.pt"
        
        try:
            if os.path.exists(MODEL_PATH):
                model = YOLO(MODEL_PATH)
                model.to('cpu')  # Force CPU
                print(f"✅ Główny model załadowany na CPU: {MODEL_PATH}")
            else:
                model = YOLO(FALLBACK_MODEL)
                model.to('cpu')  # Force CPU
                print(f"✅ Fallback model załadowany na CPU: {FALLBACK_MODEL}")
                
            # Test prediction format jak w app.py
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            test_path = "temp_api_test.jpg"
            cv2.imwrite(test_path, test_image)
            
            results = model.predict(
                test_path,
                imgsz=1280,
                conf=0.15,
                iou=0.45,
                max_det=100,
                save=False,
                device='cpu'  # Force CPU
            )
            
            # Process results jak w app.py
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                    
                    detections.append({
                        "id": i + 1,
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                        "confidence": round(conf, 3),
                        "class": "anomaly",
                        "area": int((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                    })
            
            print(f"✅ API format test OK - {len(detections)} detection objects created")
            
            # Clean up
            if os.path.exists(test_path):
                os.remove(test_path)
                
            return True
            
        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("🚗 Vehicle Anomaly Detection - Model Test")
    print("=" * 50)
    
    # Test 1: Model loading
    working_models = test_model_loading()
    
    if not working_models:
        print("\n❌ Żaden model nie działa! Sprawdź instalację.")
        return False
    
    # Test 2: Inference test
    best_model = working_models[0]  # Use first working model
    inference_ok = test_model_inference(best_model)
    
    if not inference_ok:
        print(f"\n❌ Inferencja nie działa dla {best_model}")
        return False
    
    # Test 3: API compatibility
    api_ok = test_api_compatibility()
    
    if not api_ok:
        print("\n❌ API compatibility test failed")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Wszystkie testy przeszły pomyślnie!")
    print(f"🎯 Zalecany model: {best_model}")
    print("\n📝 Zalecenia:")
    print("   1. Użyj tego modelu w app.py")
    print("   2. Sprawdź czy ścieżki są poprawne")
    print("   3. Uruchom aplikację z ./start.sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
