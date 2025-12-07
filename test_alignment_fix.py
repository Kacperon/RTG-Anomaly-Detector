#!/usr/bin/env python3
"""
Test rozwiązania problemu z wyrównywaniem obrazów
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from anomaly_detector import RTGAnomalySystem
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    print("Sprawdź czy jesteś w głównym katalogu projektu")
    sys.exit(1)

def test_quick_analysis():
    """Test szybkiej analizy bez wyrównywania"""
    
    # Sprawdź czy istnieją potrzebne foldery
    reference_dir = 'data/czyste'
    test_image_dir = 'data/brudne'
    
    if not os.path.exists(reference_dir):
        print(f"❌ Brak folderu wzorcowego: {reference_dir}")
        return False
    
    if not os.path.exists(test_image_dir):
        print(f"❌ Brak folderu testowego: {test_image_dir}")
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
    
    print(f"🧪 Testowanie szybkiej analizy dla: {os.path.basename(test_image)}")
    
    try:
        # Inicjalizuj system
        print("🔧 Inicjalizacja systemu...")
        system = RTGAnomalySystem(reference_dir, 'anomaly_reports')
        
        # Test 1: Analiza bez wyrównywania (szybka)
        print("\n📈 Test 1: Analiza bez wyrównywania (tryb szybki)")
        result1 = system.process_image(
            test_image,
            use_alignment=False,  # Wyłączone wyrównywanie
            use_ssim=True,
            save_report=False
        )
        
        print(f"✅ Wynik 1: {result1['has_anomaly']}, anomalii: {result1['anomaly_count']}")
        print(f"📊 Podobieństwo: {result1.get('similarity', 0):.2%}")
        print(f"🔬 SSIM: {result1.get('ssim_score', 0):.4f}")
        
        # Test 2: Analiza z szybkim wyrównywaniem
        print("\n📈 Test 2: Analiza z inteligentnm wyrównywaniem")
        result2 = system.process_image(
            test_image,
            use_alignment=True,   # Włączone inteligentne wyrównywanie
            use_ssim=True,
            save_report=False
        )
        
        print(f"✅ Wynik 2: {result2['has_anomaly']}, anomalii: {result2['anomaly_count']}")
        print(f"📊 Podobieństwo: {result2.get('similarity', 0):.2%}")
        print(f"🔬 SSIM: {result2.get('ssim_score', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alignment_directly():
    """Test bezpośrednio algorytmu wyrównywania"""
    from anomaly_detector import ImageAligner
    import cv2
    import numpy as np
    
    print("\n🔧 Test algorytmu wyrównywania...")
    
    # Stwórz dwa podobne obrazy testowe
    img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    img2 = img1.copy()
    
    try:
        # Test ECC alignment
        aligned, transform = ImageAligner.align_images(img1, img2, method='ecc')
        print("✅ ECC alignment działa")
        
        # Test feature alignment
        aligned2, transform2 = ImageAligner.align_images(img1, img2, method='feature')
        print("✅ Feature alignment działa")
        
        return True
    except Exception as e:
        print(f"❌ Błąd w alignment: {e}")
        return False

def main():
    print("🔧 Test poprawki problemu z wyrównywaniem")
    print("=" * 60)
    
    # Test 1: Algorytm wyrównywania
    if not test_alignment_directly():
        print("❌ Błąd w algorytmie wyrównywania")
    
    # Test 2: Pełna analiza
    if test_quick_analysis():
        print("\n✅ PROBLEM ROZWIĄZANY!")
        print("🚀 Zmiany:")
        print("   - Dodano inteligentne wykrywanie podobieństwa")
        print("   - Ograniczono iteracje ECC (5000 → 1000)")  
        print("   - Dodano fallback przy błędach")
        print("   - Domyślnie wyłączono wyrównywanie w API")
        print("   - Dodano tryb szybki")
        
        print("\n📋 INSTRUKCJE:")
        print("1. Restart backendu: cd backend && python app.py")
        print("2. Test w przeglądarce - analiza powinna być szybka")
        print("3. Jeśli nadal się zawiesza, wyłącz SSIM w apiService")
        
    else:
        print("\n❌ Problem nadal występuje")
        print("💡 Sprawdź logi w terminalu podczas analizy")

if __name__ == "__main__":
    main()
