#!/usr/bin/env python3
"""
Test maskowania tła w systemie detekcji anomalii RTG
Porównuje wyniki z i bez ignorowania białego tła
"""

import cv2
import numpy as np
import os
from backend.anomaly_detector import RTGAnomalySystem, AnomalyDetector
from pathlib import Path

def test_background_masking():
    """Test różnych metod maskowania tła"""
    
    print("🧪 Test maskowania tła w detekcji anomalii RTG")
    print("=" * 60)
    
    # Katalogi
    reference_dir = 'data/czyste'
    test_dir = 'data/brudne'
    
    if not os.path.exists(reference_dir):
        print(f"❌ Nie znaleziono katalogu wzorcowego: {reference_dir}")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ Nie znaleziono katalogu testowego: {test_dir}")
        return
    
    # Znajdź pierwszy dostępny obraz testowy
    test_images = list(Path(test_dir).rglob('*.bmp'))
    if not test_images:
        print(f"❌ Nie znaleziono obrazów testowych w: {test_dir}")
        return
    
    test_image = str(test_images[0])
    print(f"🖼️  Testowy obraz: {Path(test_image).name}")
    
    try:
        # System z różnymi ustawieniami
        system = RTGAnomalySystem(reference_dir, 'test_results')
        
        print("\n" + "🔍 Test 1: BEZ maskowania tła")
        print("-" * 40)
        result_no_mask = system.process_image(
            test_image,
            use_alignment=True,
            use_ssim=True,
            save_report=False,
            ignore_background=False
        )
        
        print(f"✅ Wykryto {result_no_mask['anomaly_count']} anomalii")
        if result_no_mask.get('ssim_score'):
            print(f"   SSIM: {result_no_mask['ssim_score']:.4f}")
        
        print("\n" + "🎯 Test 2: Z maskowaniem tła (Otsu)")
        print("-" * 40)
        result_with_mask = system.process_image(
            test_image,
            use_alignment=True,
            use_ssim=True,
            save_report=False,
            ignore_background=True
        )
        
        print(f"✅ Wykryto {result_with_mask['anomaly_count']} anomalii")
        if result_with_mask.get('ssim_score'):
            print(f"   SSIM: {result_with_mask['ssim_score']:.4f}")
        
        print("\n" + "📊 PORÓWNANIE WYNIKÓW")
        print("=" * 40)
        print(f"Bez maskowania:     {result_no_mask['anomaly_count']} anomalii")
        print(f"Z maskowaniem:      {result_with_mask['anomaly_count']} anomalii")
        
        diff = result_no_mask['anomaly_count'] - result_with_mask['anomaly_count']
        if diff > 0:
            print(f"🎯 Maskowanie usunęło {diff} fałszywych pozytywów z tła")
        elif diff < 0:
            print(f"⚠️ Maskowanie mogło usunąć {-diff} prawdziwych anomalii")
        else:
            print("🔄 Brak różnicy w liczbie wykrytych anomalii")
        
        # Test różnych metod maskowania
        print("\n" + "🔧 Test różnych metod maskowania")
        print("-" * 40)
        
        detector = AnomalyDetector()
        img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
        
        methods = ['otsu', 'adaptive', 'threshold']
        for method in methods:
            try:
                mask = detector._create_background_mask(img, method=method)
                roi_percent = np.sum(mask) / mask.size * 100
                print(f"{method:>10}: {roi_percent:5.1f}% ROI (obszary nie-tła)")
            except Exception as e:
                print(f"{method:>10}: BŁĄD - {e}")
        
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")
        return
    
    print("\n" + "✅ Test zakończony pomyślnie")

def visualize_background_mask():
    """Wizualizuje maskę tła dla przykładowego obrazu"""
    
    test_dir = 'data/brudne'
    test_images = list(Path(test_dir).rglob('*.bmp'))
    
    if not test_images:
        print("❌ Brak obrazów do wizualizacji")
        return
    
    img_path = str(test_images[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Nie można wczytać obrazu: {img_path}")
        return
    
    detector = AnomalyDetector()
    
    # Różne metody maskowania
    methods = ['otsu', 'adaptive', 'threshold']
    
    print(f"🖼️  Wizualizacja masek tła dla: {Path(img_path).name}")
    
    for method in methods:
        mask = detector._create_background_mask(img, method=method)
        
        # Zapisz wizualizację
        output_path = f"background_mask_{method}.jpg"
        
        # Stwórz obraz porównawczy
        comparison = np.hstack([
            img,  # Oryginalny
            mask.astype(np.uint8) * 255,  # Maska
            img * mask.astype(np.uint8)  # ROI
        ])
        
        cv2.imwrite(output_path, comparison)
        
        roi_percent = np.sum(mask) / mask.size * 100
        print(f"💾 {method}: {output_path} (ROI: {roi_percent:.1f}%)")

if __name__ == "__main__":
    print("🔬 Testy maskowania tła RTG")
    print("=" * 50)
    
    # Test podstawowy
    test_background_masking()
    
    print("\n" + "🎨 Generowanie wizualizacji masek...")
    visualize_background_mask()
    
    print("\n" + "📁 Sprawdź wygenerowane pliki:")
    print("   - background_mask_otsu.jpg")
    print("   - background_mask_adaptive.jpg") 
    print("   - background_mask_threshold.jpg")
