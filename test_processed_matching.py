#!/usr/bin/env python3
"""
Test wyszukiwania najbardziej podobnego obrazu włączając przetworzone obrazy
Sprawdza czy system znajduje obrazy z katalogu data-processing/processed_clean_data
"""

import cv2
import numpy as np
import os
from backend.anomaly_detector import RTGAnomalySystem, ImageMatcher
from pathlib import Path

def test_image_matching_with_processed():
    """Test wyszukiwania obrazów z uwzględnieniem przetworzonych"""
    
    print("🔍 Test wyszukiwania obrazów z katalogiem przetworzonych")
    print("=" * 70)
    
    # Katalogi
    reference_dir = 'data/czyste'
    processed_dir = 'data-processing/processed_clean_data'
    test_dir = 'data/brudne'
    
    # Sprawdź czy katalogi istnieją
    if not os.path.exists(reference_dir):
        print(f"❌ Nie znaleziono katalogu wzorcowego: {reference_dir}")
        return
    
    if not os.path.exists(processed_dir):
        print(f"❌ Nie znaleziono katalogu przetworzonych: {processed_dir}")
        return
        
    if not os.path.exists(test_dir):
        print(f"❌ Nie znaleziono katalogu testowego: {test_dir}")
        return
    
    # Test 1: Sprawdź czy matcher ładuje oba katalogi
    print("\n🧪 Test 1: Ładowanie obrazów wzorcowych")
    print("-" * 50)
    
    matcher = ImageMatcher(reference_dir, processed_dir)
    
    if len(matcher.reference_images) == 0:
        print("❌ Nie załadowano żadnych obrazów wzorcowych")
        return
    
    # Policz obrazy z różnych źródeł
    original_images = [img for img in matcher.reference_images if img['source'] == 'original']
    processed_images = [img for img in matcher.reference_images if img['source'] == 'processed']
    
    print(f"📊 Obrazy wzorcowe oryginalne: {len(original_images)}")
    print(f"🔧 Obrazy wzorcowe przetworzone: {len(processed_images)}")
    print(f"📈 Łącznie: {len(matcher.reference_images)}")
    
    # Pokaż kilka przykładów przetworzonych obrazów
    if processed_images:
        print(f"\n🔧 Przykłady przetworzonych obrazów:")
        for i, img in enumerate(processed_images[:5]):
            print(f"   {i+1}. {img['path'].name}")
        if len(processed_images) > 5:
            print(f"   ... i {len(processed_images) - 5} więcej")
    
    # Test 2: Znajdź najbardziej podobny obraz dla testowego
    print(f"\n🧪 Test 2: Wyszukiwanie dopasowań")
    print("-" * 50)
    
    # Znajdź pierwszy dostępny obraz testowy
    test_images = list(Path(test_dir).rglob('*.bmp'))
    if not test_images:
        print(f"❌ Nie znaleziono obrazów testowych w: {test_dir}")
        return
    
    test_image_path = str(test_images[0])
    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    if test_img is None:
        print(f"❌ Nie można wczytać obrazu testowego: {test_image_path}")
        return
    
    print(f"🖼️  Testowy obraz: {Path(test_image_path).name}")
    
    # Znajdź najlepsze dopasowania
    matches = matcher.find_best_match(test_img, top_k=10)
    
    print(f"\n🏆 Top 10 dopasowań:")
    for i, match in enumerate(matches, 1):
        source_emoji = "🔧" if match.get('source') == 'processed' else "📁"
        similarity_percent = match['similarity'] * 100
        print(f"{i:2d}. {source_emoji} {match['path'].name[:40]:40} "
              f"({similarity_percent:5.1f}%)")
    
    # Sprawdź czy najlepsze dopasowanie to przetworzone
    if matches:
        best_match = matches[0]
        is_processed = best_match.get('source') == 'processed'
        print(f"\n🎯 Najlepsze dopasowanie:")
        print(f"   Plik: {best_match['path'].name}")
        print(f"   Źródło: {'Przetworzone 🔧' if is_processed else 'Oryginalne 📁'}")
        print(f"   Podobieństwo: {best_match['similarity']:.1%}")
        
        # Sprawdź czy znajduje konkretny plik z cropped
        target_file = "48001F003202511180021_cropped.bmp"
        found_target = any(target_file in str(match['path']) for match in matches)
        
        if found_target:
            target_match = next(match for match in matches if target_file in str(match['path']))
            target_position = next(i for i, match in enumerate(matches) if target_file in str(match['path']))
            print(f"\n🎯 Znaleziono poszukiwany plik '{target_file}':")
            print(f"   Pozycja w rankingu: {target_position + 1}")
            print(f"   Podobieństwo: {target_match['similarity']:.1%}")
        else:
            print(f"\n⚠️ Nie znaleziono plik '{target_file}' w dopasowaniach")
    
    # Test 3: Pełny system z raportem
    print(f"\n🧪 Test 3: System z generowaniem raportu")
    print("-" * 50)
    
    try:
        system = RTGAnomalySystem(
            reference_dir=reference_dir,
            output_dir='test_reports',
            processed_dir=processed_dir
        )
        
        result = system.process_image(
            test_image_path,
            use_alignment=True,
            use_ssim=True,
            save_report=True,
            ignore_background=True
        )
        
        print(f"✅ Analiza zakończona:")
        print(f"   Wykryto anomalii: {result['anomaly_count']}")
        print(f"   Dopasowany obraz: {Path(result['reference_match']).name}")
        print(f"   Podobieństwo: {result['similarity']:.1%}")
        if result.get('ssim_score'):
            print(f"   SSIM: {result['ssim_score']:.4f}")
        if result.get('report_path'):
            print(f"   Raport: {result['report_path']}")
            
    except Exception as e:
        print(f"❌ Błąd podczas analizy: {e}")

def check_processed_directory():
    """Sprawdź zawartość katalogu z przetworzonymi obrazami"""
    
    print("\n📁 Analiza katalogu przetworzonych obrazów")
    print("=" * 50)
    
    processed_dir = Path('data-processing/processed_clean_data')
    
    if not processed_dir.exists():
        print(f"❌ Katalog nie istnieje: {processed_dir}")
        return
    
    # Znajdź wszystkie obrazy
    image_files = list(processed_dir.rglob('*.bmp'))
    image_files.extend(list(processed_dir.rglob('*.jpg')))
    image_files.extend(list(processed_dir.rglob('*.png')))
    
    print(f"📊 Znaleziono {len(image_files)} obrazów")
    
    # Pokaż przykłady
    print(f"\n📋 Lista plików:")
    for i, file_path in enumerate(image_files[:10]):
        file_size = file_path.stat().st_size / 1024  # KB
        print(f"   {i+1:2d}. {file_path.name} ({file_size:.1f} KB)")
    
    if len(image_files) > 10:
        print(f"   ... i {len(image_files) - 10} więcej")
    
    # Sprawdź konkretny plik
    target_file = "48001F003202511180021_cropped.bmp"
    target_path = processed_dir / target_file
    
    if target_path.exists():
        file_size = target_path.stat().st_size / 1024  # KB
        print(f"\n🎯 Znaleziono poszukiwany plik:")
        print(f"   Nazwa: {target_file}")
        print(f"   Rozmiar: {file_size:.1f} KB")
        print(f"   Pełna ścieżka: {target_path}")
        
        # Sprawdź czy obraz da się wczytać
        img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"   Rozdzielczość: {img.shape[1]}x{img.shape[0]}")
            print(f"   ✅ Obraz wczytuje się poprawnie")
        else:
            print(f"   ❌ Nie można wczytać obrazu")
    else:
        print(f"\n❌ Nie znaleziono poszukiwanego pliku: {target_file}")

if __name__ == "__main__":
    print("🔬 Testowanie wyszukiwania obrazów z przetworzonych")
    print("=" * 60)
    
    # Sprawdź katalog
    check_processed_directory()
    
    # Test wyszukiwania
    test_image_matching_with_processed()
    
    print("\n" + "=" * 60)
    print("✅ Testowanie zakończone")
