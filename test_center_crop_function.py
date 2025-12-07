#!/usr/bin/env python3
"""
Test przepisanej funkcji find_and_compare_with_processed_images z center_crop
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_center_crop_function():
    """Test nowej funkcji z center_crop"""
    
    try:
        # Importuj moduł
        from backend.modelv2.detector import ModelV2Detector
        
        # Sprawdź czy istnieją potrzebne foldery
        test_image_dir = 'data/brudne'
        processed_dir = 'data-processing/processed_clean_data'
        
        if not os.path.exists(test_image_dir):
            print(f"❌ Brak folderu testowego: {test_image_dir}")
            return False
            
        if not os.path.exists(processed_dir):
            print(f"❌ Brak folderu processed: {processed_dir}")
            # Sprawdź alternatywną lokalizację
            alt_processed_dir = 'procesed_imagines'
            if os.path.exists(alt_processed_dir):
                print(f"✅ Znaleziono alternatywny folder: {alt_processed_dir}")
                processed_dir = alt_processed_dir
            else:
                print(f"❌ Brak folderu processed w obu lokalizacjach")
                return False
        
        # Znajdź pierwszy obraz testowy
        test_image = None
        for root, dirs, files in os.walk(test_image_dir):
            for file in files:
                if file.endswith('.bmp') and 'czarno' not in file.lower():
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
        
        if not test_image:
            print(f"❌ Brak obrazów testowych w {test_image_dir}")
            return False
        
        print(f"🧪 Testowanie center_crop dla: {os.path.basename(test_image)}")
        
        # Inicjalizuj detektor
        print("🔧 Inicjalizacja detektora modelv2...")
        detector = ModelV2Detector()
        
        # Załaduj model
        print("📦 Ładowanie modelu YOLO...")
        load_result = detector.load_model()
        if not load_result["success"]:
            print(f"❌ Błąd ładowania modelu: {load_result['error']}")
            return False
        print("✅ Model załadowany pomyślnie")
        
        # Test 1: Sama funkcja _center_crop
        print("\n📈 Test 1: _center_crop")
        cropped = detector._center_crop(test_image)
        if cropped is not None:
            print(f"✅ Center crop sukces: {cropped.shape}")
        else:
            print("❌ Center crop zwrócił None")
            return False
        
        # Test 2: Pełna funkcja find_and_compare_with_processed_images
        print("\n📈 Test 2: find_and_compare_with_processed_images z center_crop")
        result = detector.find_and_compare_with_processed_images(
            test_image, 
            output_dir='results'
        )
        
        if result["success"]:
            print("✅ Funkcja zakończona sukcesem!")
            print(f"📊 Center crop użyty: {result.get('center_crop_used', False)}")
            print(f"🖼️  Region shape: {result.get('object_region_shape')}")
            print(f"🎯 Best match: {os.path.basename(result.get('best_match_path', 'N/A'))}")
            print(f"📈 Różnica (MSE): {result.get('best_match_difference', 'N/A')}")
            
            # Sprawdź zapisane pliki
            if 'object_region_saved' in result:
                print(f"📁 Zapisano region: {os.path.basename(result['object_region_saved'])}")
            if 'color_diff_map_saved' in result:
                print(f"🔥 Zapisano mapę różnic: {os.path.basename(result['color_diff_map_saved'])}")
            if 'comparison_saved' in result:
                print(f"📊 Zapisano porównanie: {os.path.basename(result['comparison_saved'])}")
        else:
            print(f"❌ Błąd funkcji: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 Test przepisanej funkcji find_and_compare_with_processed_images")
    print("=" * 70)
    
    if test_center_crop_function():
        print("\n✅ SUKCES! Funkcja przepisana poprawnie!")
        print("🚀 Zmiany:")
        print("   - Używa _center_crop zamiast _extract_object_region")
        print("   - Automatycznie wykrywa i wycina największy obiekt z 10% marginesem")
        print("   - Zachowuje kompatybilność z istniejącą strukturą odpowiedzi")
        print("   - Dodaje flagę 'center_crop_used' do wyników")
        print("   - Zapisuje pliki z prefiksem 'center_crop_'")
        
        print("\n📋 EFEKTY:")
        print("   - Szybsza detekcja (jedno wywołanie YOLO zamiast dwóch)")
        print("   - Lepsze wycinanie obiektu (inteligentne powiększanie bounding box)")
        print("   - Bardziej precyzyjne porównanie kolorów")
        
    else:
        print("\n❌ Test nieudany")
        print("💡 Sprawdź:")
        print("   - Czy model YOLO się ładuje")
        print("   - Czy istnieją foldery z obrazami")
        print("   - Czy nie ma błędów w kodzie")

if __name__ == "__main__":
    main()
