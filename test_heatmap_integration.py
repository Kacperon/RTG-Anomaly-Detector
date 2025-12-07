#!/usr/bin/env python3
"""
Test integracji heatmapy - sprawdza czy backend poprawnie generuje heatmapę
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import requests
import base64
import json
from pathlib import Path

def test_heatmap_endpoint():
    """Test endpointu analyze-comparison"""
    
    # URL endpointu
    url = "http://localhost:5000/api/analyze-comparison"
    
    # Przykładowe dane testowe
    test_data = {
        "file_id": "test123",  # To musi być plik, który istnieje w uploads
        "use_alignment": True,
        "use_ssim": True
    }
    
    try:
        # Wyślij zapytanie POST
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint odpowiada poprawnie!")
            print(f"Method: {data.get('method')}")
            print(f"Analysis complete: {data.get('analysis_complete')}")
            print(f"Has anomaly: {data.get('has_anomaly')}")
            print(f"Anomaly count: {data.get('anomaly_count')}")
            print(f"Has heatmap image: {'heatmap_image' in data}")
            print(f"Has annotated image: {'annotated_image' in data}")
            print(f"Has original image: {'original_image' in data}")
            
            # Sprawdź długość danych base64 (czy obrazy nie są puste)
            if 'heatmap_image' in data and data['heatmap_image']:
                print(f"Heatmap image size: ~{len(data['heatmap_image'])} chars")
            if 'annotated_image' in data and data['annotated_image']:
                print(f"Annotated image size: ~{len(data['annotated_image'])} chars")
                
        else:
            print("❌ Błąd odpowiedzi:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Nie można połączyć się z serwerem. Czy backend jest uruchomiony?")
    except Exception as e:
        print(f"❌ Błąd: {e}")

def check_backend_status():
    """Sprawdź status backendu"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend jest dostępny")
            return True
        else:
            print(f"❌ Backend odpowiada z kodem {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend nie odpowiada. Uruchom serwer:")
        print("   cd backend && python app.py")
        return False
    except Exception as e:
        print(f"❌ Błąd sprawdzania statusu: {e}")
        return False

def main():
    print("🧪 Test integracji heatmapy")
    print("=" * 50)
    
    # Sprawdź status backendu
    if not check_backend_status():
        return
    
    print("\n📡 Testowanie endpointu analyze-comparison...")
    test_heatmap_endpoint()
    
    print("\n" + "=" * 50)
    print("💡 Instrukcje testowania:")
    print("1. Uruchom backend: cd backend && python app.py")
    print("2. Uruchom frontend: cd frontend && npm start")
    print("3. Załaduj obraz RTG i kliknij 'Rozpocznij analizę'")
    print("4. Sprawdź czy przełączniki: Oryginalny | Z anomaliami | Heatmapa działają")

if __name__ == "__main__":
    main()
