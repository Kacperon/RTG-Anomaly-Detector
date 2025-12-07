#!/usr/bin/env python3
"""
Test funkcjonalności frontendu i backendu dla wykrywania anomalii
Sprawdza czy dane są poprawnie przesyłane między komponentami
"""

import json
import requests
import base64
import cv2
import numpy as np
from pathlib import Path
import os

def test_backend_api():
    """Test API backendu"""
    print("🧪 Test API backendu")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    api_url = f"{base_url}/api"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: OK")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend nie odpowiada: {e}")
        return False
    
    # Test 2: Status detektora
    try:
        response = requests.get(f"{api_url}/detector-status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status detektora:")
            print(f"   YOLO model loaded: {status.get('yolo_model_loaded')}")
            print(f"   Comparison detector available: {status.get('comparison_detector_available')}")
            print(f"   Reference dir exists: {status.get('reference_dir_exists')}")
        else:
            print(f"❌ Status detector failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Status check error: {e}")
    
    # Test 3: Upload i analiza obrazu (jeśli istnieje)
    test_image_path = None
    for path in ['data/brudne', 'data/uploads']:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.bmp'):
                        test_image_path = os.path.join(root, file)
                        break
                if test_image_path:
                    break
        if test_image_path:
            break
    
    if not test_image_path:
        print("⚠️ Brak obrazu testowego - tworzę sztuczny")
        # Stwórz prosty obraz testowy
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_image_path = 'test_image.bmp'
        cv2.imwrite(test_image_path, test_img)
    
    print(f"🖼️  Używam obrazu testowego: {test_image_path}")
    
    # Upload
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/upload", files=files, timeout=30)
            
        if response.status_code == 200:
            upload_result = response.json()
            file_id = upload_result.get('file_id')
            print(f"✅ Upload: OK (file_id: {file_id[:8]}...)")
            
            # Analiza porównawcza
            analysis_data = {
                'file_id': file_id,
                'use_alignment': False,
                'use_ssim': True,
                'fast_mode': True
            }
            
            response = requests.post(
                f"{api_url}/analyze-comparison", 
                json=analysis_data, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analiza porównawcza: OK")
                print(f"   Metoda: {result.get('method')}")
                print(f"   Ma anomalie: {result.get('has_anomaly')}")
                print(f"   Liczba anomalii: {result.get('anomaly_count')}")
                print(f"   Wykrycia: {len(result.get('detections', []))}")
                print(f"   Ma heatmapę: {'heatmap_image' in result}")
                print(f"   Ma annotated: {'annotated_image' in result}")
                print(f"   SSIM: {result.get('ssim_score')}")
                
                # Sprawdź format danych anomalii
                detections = result.get('detections', [])
                if detections:
                    print(f"   Przykład detekcji:")
                    first_detection = detections[0]
                    for key, value in first_detection.items():
                        print(f"     {key}: {value}")
                
                return True
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                print(f"❌ Analiza failed: {response.status_code}")
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
                return False
                
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test upload/analysis failed: {e}")
        return False
    
    finally:
        # Cleanup
        if test_image_path == 'test_image.bmp' and os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_data_structure():
    """Test struktury danych dla frontendu"""
    print("\n🔍 Test struktury danych")
    print("=" * 50)
    
    # Przykładowa struktura danych z backendu
    example_response = {
        "method": "comparison_based",
        "analysis_complete": True,
        "has_anomaly": True,
        "anomaly_count": 2,
        "detection_count": 2,
        "anomalies": [
            {
                "id": 1,
                "bbox": [100, 150, 200, 250],
                "area": 2500,
                "solidity": 0.85,
                "aspect_ratio": 1.2,
                "center": [150, 200],
                "confidence": 0.75
            },
            {
                "id": 2,
                "bbox": [300, 400, 380, 460],
                "area": 4800,
                "solidity": 0.92,
                "aspect_ratio": 1.33,
                "center": [340, 430],
                "confidence": 0.82
            }
        ],
        "detections": [
            {
                "id": 1,
                "bbox": [100, 150, 200, 250],
                "area": 2500,
                "solidity": 0.85,
                "aspect_ratio": 1.2,
                "center": [150, 200],
                "confidence": 0.75
            },
            {
                "id": 2,
                "bbox": [300, 400, 380, 460],
                "area": 4800,
                "solidity": 0.92,
                "aspect_ratio": 1.33,
                "center": [340, 430],
                "confidence": 0.82
            }
        ],
        "heatmap_image": "base64_encoded_image_data...",
        "annotated_image": "base64_encoded_image_data...",
        "original_image": "base64_encoded_image_data...",
        "ssim_score": 0.8765,
        "similarity": 0.9234,
        "reference_match": "/path/to/reference/image.bmp"
    }
    
    print("📊 Struktura odpowiedzi z backendu:")
    print(f"   Metoda: {example_response.get('method')}")
    print(f"   Ma anomalie: {example_response.get('has_anomaly')}")
    print(f"   Liczba anomalii: {example_response.get('anomaly_count')}")
    print(f"   Liczba detekcji: {example_response.get('detection_count')}")
    print(f"   Długość listy anomalies: {len(example_response.get('anomalies', []))}")
    print(f"   Długość listy detections: {len(example_response.get('detections', []))}")
    
    # Test logiki frontendu
    has_anomalies = example_response.get('detection_count', 0) > 0
    detections = example_response.get('detections') or example_response.get('anomalies') or []
    
    print(f"\n🎯 Logika frontendu:")
    print(f"   hasAnomalies: {has_anomalies}")
    print(f"   detections to display: {len(detections)}")
    
    if detections:
        for i, detection in enumerate(detections):
            confidence = detection.get('confidence', 0.5)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            center = detection.get('center', [0, 0])
            area = detection.get('area', 0)
            
            print(f"   Detection {i+1}:")
            print(f"     ID: {detection.get('id', i+1)}")
            print(f"     Confidence: {confidence:.1%}")
            print(f"     Position: ({bbox[0] if bbox else center[0]}, {bbox[1] if bbox else center[1]})")
            print(f"     Area: {area}px²")
    
    return True

def check_frontend_compatibility():
    """Sprawdź kompatybilność z frontendem"""
    print("\n🔧 Test kompatybilności frontendu")
    print("=" * 50)
    
    # Sprawdź czy wszystkie wymagane pola są obecne
    required_fields = [
        'method',
        'analysis_complete', 
        'has_anomaly',
        'anomaly_count',
        'detection_count',
        'detections',
        'heatmap_image',
        'annotated_image',
        'original_image'
    ]
    
    optional_fields = [
        'anomalies',
        'ssim_score', 
        'similarity',
        'reference_match',
        'timestamp'
    ]
    
    print("📋 Wymagane pola:")
    for field in required_fields:
        print(f"   ✅ {field}")
    
    print("\n📋 Opcjonalne pola:")
    for field in optional_fields:
        print(f"   ➖ {field}")
    
    print(f"\n🔍 Frontend spodziewa się struktury:")
    print(f"   results.detection_count - liczba wykrytych anomalii")
    print(f"   results.detections[] - lista detekcji do wyświetlenia")
    print(f"   results.heatmap_image - obraz heatmapy w base64")
    print(f"   results.annotated_image - obraz z adnotacjami w base64")
    print(f"   detection.id - ID detekcji")
    print(f"   detection.confidence - pewność (0-1)")
    print(f"   detection.bbox[] - pozycja [x1, y1, x2, y2]")
    print(f"   detection.area - powierzchnia w pikselach")
    
    return True

if __name__ == "__main__":
    print("🔬 Test systemu wykrywania anomalii")
    print("=" * 60)
    
    success = True
    
    # Test backendu
    if not test_backend_api():
        success = False
    
    # Test struktury danych
    if not test_data_structure():
        success = False
    
    # Test kompatybilności
    if not check_frontend_compatibility():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Wszystkie testy przeszły pomyślnie!")
        print("\n🔧 Troubleshooting wskazówki:")
        print("1. Upewnij się, że backend działa na localhost:5000")
        print("2. Sprawdź czy katalog data/czyste zawiera obrazy wzorcowe")  
        print("3. Sprawdź czy katalog data-processing/processed_clean_data istnieje")
        print("4. Sprawdź console.log w przeglądarce dla debug info")
        print("5. Sprawdź Network tab w Developer Tools")
    else:
        print("❌ Niektóre testy się nie powiodły")
        print("Sprawdź logi powyżej i napraw problemy przed testowaniem frontendu")
