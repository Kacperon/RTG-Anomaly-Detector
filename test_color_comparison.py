#!/usr/bin/env python3
"""
Skrypt testowy dla nowej funkcjonalności porównywania kolorów
"""

import os
import sys
from backend.modelv2.detector import find_and_compare_with_processed_images, load_model

def test_color_comparison():
    """Test funkcjonalności porównywania kolorów"""
    
    print("=== Test porównywania kolorów z przetworzonymi obrazami ===")
    
    # Załaduj model
    print("\n1. Ładowanie modelu...")
    load_result = load_model()
    if not load_result["success"]:
        print(f"❌ Błąd ładowania modelu: {load_result['error']}")
        return
    
    print("✅ Model załadowany pomyślnie")
    
    # Przykładowa ścieżka do obrazu (użyj dowolnego obrazu z data/uploads lub data/brudne)
    test_image_path = None
    
    # Sprawdź dostępne obrazy w folderach
    possible_dirs = [
        "data/uploads",
        "data/brudne/202511190032", 
        "data/brudne/202511190033",
        "uploads"
    ]
    
    for dir_path in possible_dirs:
        full_path = os.path.join(os.getcwd(), dir_path)
        if os.path.exists(full_path):
            for filename in os.listdir(full_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_image_path = os.path.join(full_path, filename)
                    break
        if test_image_path:
            break
    
    if not test_image_path:
        print("❌ Brak obrazów testowych. Umieść jakiś obraz w folderze data/uploads/")
        print("📁 Sprawdzone foldery:", possible_dirs)
        return
    
    print(f"\n2. Używając obrazu testowego: {test_image_path}")
    
    # Utwórz folder wyników jeśli nie istnieje
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Uruchom porównanie
    print("\n3. Uruchamianie porównania kolorów...")
    result = find_and_compare_with_processed_images(test_image_path, results_dir)
    
    if result["success"]:
        print("✅ Porównanie zakończone pomyślnie!")
        print(f"📊 Wykryto obiektów: {result['detection_result']['detection_count']}")
        if result['detection_result']['detection_count'] > 0:
            bbox = result["object_bbox"]
            print(f"📦 Bounding box obiektu: {bbox}")
            print(f"🖼️  Rozmiar wyciętego regionu: {result['object_region_shape']}")
        
        if "matching_result" in result:
            matching = result["matching_result"]
            if matching["success"]:
                print(f"🎯 Najlepsze dopasowanie: {os.path.basename(matching['best_match_path'])}")
                print(f"📈 Różnica kolorów (MSE): {matching['best_difference']:.2f}")
                print(f"🔍 Przeanalizowano kandydatów: {matching['total_candidates']}")
            else:
                print(f"⚠️  Brak dopasowania: {matching['error']}")
        
        # Wyświetl informacje o zapisanych plikach
        if "object_region_saved" in result:
            print(f"💾 Zapisano region obiektu: {result['object_region_saved']}")
        if "color_diff_map_saved" in result:
            print(f"💾 Zapisano mapę różnic: {result['color_diff_map_saved']}")
        if "comparison_saved" in result:
            print(f"💾 Zapisano porównanie: {result['comparison_saved']}")
            
    else:
        print(f"❌ Błąd podczas porównania: {result['error']}")
        if "processed_images_dir" in result:
            print(f"📁 Sprawdzany folder: {result['processed_images_dir']}")
    
    # Wyświetl informacje o folderze processed_images
    print(f"\n4. Informacje o folderze przetworzonych obrazów:")
    processed_dir = os.path.join(os.getcwd(), "procesed_imagines")
    print(f"📁 Ścieżka: {processed_dir}")
    print(f"🗂️  Istnieje: {os.path.exists(processed_dir)}")
    if os.path.exists(processed_dir):
        files = [f for f in os.listdir(processed_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"🖼️  Liczba obrazów: {len(files)}")
        if files:
            print(f"📋 Pierwsze 5 plików: {files[:5]}")
    else:
        print("💡 Utwórz folder 'procesed_imagines' i umieść w nim obrazy do porównania")

if __name__ == "__main__":
    test_color_comparison()
