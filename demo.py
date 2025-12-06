#!/usr/bin/env python3
# demo.py - Prosty skrypt demonstracyjny systemu detekcji anomalii

"""
Skrypt demonstracyjny pokazujący możliwości systemu detekcji anomalii RTG
Użycie: python demo.py
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Wyświetl ozdobny nagłówek"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_dependencies():
    """Sprawdź czy wszystkie zależności są zainstalowane"""
    print_header("🔍 Sprawdzanie zależności")
    
    required = ['cv2', 'numpy', 'scipy', 'skimage', 'PIL']
    missing = []
    
    for module in required:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - BRAK")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️ Brakujące moduły: {', '.join(missing)}")
        print("Zainstaluj za pomocą: pip install -r requirements.txt")
        return False
    
    print("\n✅ Wszystkie zależności są zainstalowane")
    return True


def check_data_structure():
    """Sprawdź strukturę danych"""
    print_header("📁 Sprawdzanie struktury danych")
    
    data_dir = Path('data')
    clean_dir = data_dir / 'czyste'
    dirty_dir = data_dir / 'brudne'
    
    checks = [
        (data_dir, "Katalog główny danych"),
        (clean_dir, "Katalog z obrazami czystymi (wzorcowymi)"),
        (dirty_dir, "Katalog z obrazami do testowania"),
    ]
    
    all_ok = True
    for path, description in checks:
        if path.exists():
            count = len(list(path.rglob('*.bmp')))
            print(f"✅ {description}: {path} ({count} plików .bmp)")
        else:
            print(f"❌ {description}: {path} - BRAK")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ Struktura danych niekompletna!")
        print("Upewnij się, że masz:")
        print("  data/czyste/ - z obrazami wzorcowymi")
        print("  data/brudne/ - z obrazami do testowania")
        return False
    
    return True


def demo_quick_detect():
    """Demo szybkiej detekcji pojedynczego obrazu"""
    print_header("🎯 DEMO 1: Szybka detekcja pojedynczego obrazu")
    
    try:
        from anomaly_detector import quick_detect
        
        # Znajdź przykładowy obraz
        test_images = list(Path('data/brudne').rglob('*.bmp'))
        test_images = [img for img in test_images if 'czarno' not in img.name.lower()]
        
        if not test_images:
            print("❌ Brak obrazów testowych w data/brudne/")
            return
        
        test_image = test_images[0]
        print(f"📸 Testowanie obrazu: {test_image}")
        print("⏳ Przetwarzanie...")
        
        # Wykryj anomalie
        result = quick_detect(str(test_image))
        
        # Wyświetl wyniki
        print("\n" + "─"*80)
        print("📊 WYNIKI ANALIZY")
        print("─"*80)
        
        if result['has_anomaly']:
            print(f"🔴 ANOMALIA WYKRYTA!")
            print(f"   Liczba wykrytych anomalii: {result['anomaly_count']}")
        else:
            print(f"🟢 BRAK ANOMALII")
        
        print(f"\n📈 Metryki:")
        print(f"   Podobieństwo do wzorca: {result['similarity']:.2%}")
        if result.get('ssim_score'):
            print(f"   SSIM score: {result['ssim_score']:.4f}")
        
        print(f"\n📋 Szczegóły:")
        print(f"   Dopasowany wzorzec: {Path(result['reference_match']).name}")
        print(f"   Raport zapisany: {result['report_path']}")
        
        if result.get('anomalies'):
            print(f"\n🔍 Wykryte anomalie:")
            for i, anomaly in enumerate(result['anomalies'][:5], 1):
                bbox = anomaly['bbox']
                print(f"   {i}. Pozycja: ({bbox[0]}, {bbox[1]}) "
                      f"Rozmiar: {bbox[2]}x{bbox[3]} px "
                      f"Powierzchnia: {anomaly['area']:.0f} px²")
        
        print(f"\n💡 Możesz zobaczyć wizualizację w: {result['report_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_batch_processing():
    """Demo przetwarzania wielu obrazów"""
    print_header("📦 DEMO 2: Przetwarzanie partiami")
    
    try:
        from anomaly_detector import RTGAnomalySystem
        
        # Inicjalizuj system
        print("⚙️ Inicjalizacja systemu detekcji...")
        system = RTGAnomalySystem('data/czyste', 'anomaly_reports')
        
        # Znajdź obrazy do przetestowania (ogranicz do 3 dla demo)
        all_images = list(Path('data/brudne').rglob('*.bmp'))
        all_images = [img for img in all_images if 'czarno' not in img.name.lower()]
        test_images = all_images[:3]
        
        if not test_images:
            print("❌ Brak obrazów testowych")
            return
        
        print(f"📸 Znaleziono {len(all_images)} obrazów, testuję {len(test_images)} pierwszych...")
        print("⏳ Przetwarzanie...\n")
        
        # Przetwarzaj obrazy
        results = []
        for i, img_path in enumerate(test_images, 1):
            print(f"  [{i}/{len(test_images)}] {img_path.name}...", end=" ")
            try:
                result = system.process_image(str(img_path), save_report=True)
                results.append(result)
                status = "🔴 ANOMALIA" if result['has_anomaly'] else "🟢 CZYSTE"
                print(f"{status} ({result['anomaly_count']} wykryć)")
            except Exception as e:
                print(f"❌ Błąd: {e}")
        
        # Podsumowanie
        print("\n" + "─"*80)
        print("📊 PODSUMOWANIE")
        print("─"*80)
        
        anomaly_count = sum(1 for r in results if r.get('has_anomaly', False))
        clean_count = len(results) - anomaly_count
        
        print(f"   Przetworzono: {len(results)} obrazów")
        print(f"   Z anomaliami: {anomaly_count} 🔴")
        print(f"   Bez anomalii: {clean_count} 🟢")
        
        if anomaly_count > 0:
            total_anomalies = sum(r.get('anomaly_count', 0) for r in results)
            print(f"   Łącznie wykryto: {total_anomalies} anomalii")
        
        # Średnie metryki
        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)
        print(f"\n📈 Średnie podobieństwo do wzorców: {avg_similarity:.2%}")
        
        print(f"\n💾 Wszystkie raporty zapisane w: anomaly_reports/")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_comparison():
    """Demo porównania różnych metod"""
    print_header("⚖️ DEMO 3: Porównanie metod detekcji")
    
    try:
        from anomaly_detector import RTGAnomalySystem
        
        # Znajdź przykładowy obraz
        test_images = list(Path('data/brudne').rglob('*.bmp'))
        test_images = [img for img in test_images if 'czarno' not in img.name.lower()]
        
        if not test_images:
            print("❌ Brak obrazów testowych")
            return
        
        test_image = str(test_images[0])
        print(f"📸 Testowanie na: {Path(test_image).name}\n")
        
        system = RTGAnomalySystem('data/czyste', 'anomaly_reports')
        
        # Test różnych konfiguracji
        configs = [
            ("SSIM + Wyrównywanie", True, True),
            ("SSIM bez wyrównywania", False, True),
            ("Różnica pikselowa + Wyrównywanie", True, False),
            ("Różnica pikselowa bez wyrównywania", False, False),
        ]
        
        print("⏳ Testowanie różnych konfiguracji...\n")
        
        results_table = []
        for name, use_align, use_ssim in configs:
            print(f"  Testowanie: {name}...", end=" ")
            try:
                result = system.process_image(
                    test_image,
                    use_alignment=use_align,
                    use_ssim=use_ssim,
                    save_report=False  # Nie zapisuj dla tego demo
                )
                results_table.append((name, result))
                print(f"✅ ({result['anomaly_count']} anomalii)")
            except Exception as e:
                print(f"❌ Błąd: {e}")
        
        # Wyświetl porównanie
        print("\n" + "─"*80)
        print("📊 PORÓWNANIE WYNIKÓW")
        print("─"*80)
        print(f"{'Metoda':<40} {'Anomalie':<12} {'Podobieństwo':<15} {'SSIM'}")
        print("─"*80)
        
        for name, result in results_table:
            anomalies = f"{result['anomaly_count']}"
            similarity = f"{result['similarity']:.2%}"
            ssim = f"{result.get('ssim_score', 0):.4f}" if result.get('ssim_score') else "N/A"
            print(f"{name:<40} {anomalies:<12} {similarity:<15} {ssim}")
        
        print("─"*80)
        
        # Rekomendacje
        print("\n💡 REKOMENDACJE:")
        print("   • SSIM + Wyrównywanie: Najbardziej dokładne, wolniejsze")
        print("   • SSIM bez wyrównywania: Szybsze, mniej dokładne dla przesunięć")
        print("   • Różnica pikselowa: Najszybsze, dobre dla wyraźnych anomalii")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_api_integration():
    """Demo integracji z API"""
    print_header("🌐 DEMO 4: Integracja z API Flask")
    
    print("ℹ️ Aby użyć API, uruchom serwer:")
    print("   python app.py")
    print("\nPrzykładowe zapytania:")
    print("\n1. Sprawdź status systemu:")
    print("   curl http://localhost:5000/api/detector-status")
    print("\n2. Prześlij obraz:")
    print("   curl -X POST http://localhost:5000/api/upload \\")
    print("        -F 'file=@data/brudne/.../image.bmp'")
    print("\n3. Przeanalizuj obraz:")
    print("   curl -X POST http://localhost:5000/api/analyze-comparison \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"file_id\": \"...\", \"use_alignment\": true, \"use_ssim\": true}'")
    print("\n4. Przetwarzanie partiami:")
    print("   curl -X POST http://localhost:5000/api/batch-analyze \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"directory\": \"data/brudne\", \"pattern\": \"*.bmp\"}'")
    
    print("\n💡 Zobacz ANOMALY_DETECTION_GUIDE.md dla więcej szczegółów")


def show_menu():
    """Wyświetl menu główne"""
    print_header("🔬 System Detekcji Anomalii RTG - DEMO")
    
    print("Wybierz demo:")
    print("  1. Szybka detekcja pojedynczego obrazu")
    print("  2. Przetwarzanie partiami (3 obrazy)")
    print("  3. Porównanie różnych metod")
    print("  4. Informacje o API")
    print("  5. Uruchom wszystkie demo")
    print("  0. Wyjście")
    
    choice = input("\nWybór (0-5): ").strip()
    return choice


def main():
    """Główna funkcja demo"""
    
    # Sprawdź zależności
    if not check_dependencies():
        print("\n❌ Nie można kontynuować bez wszystkich zależności")
        sys.exit(1)
    
    # Sprawdź dane
    if not check_data_structure():
        print("\n❌ Nie można kontynuować bez prawidłowej struktury danych")
        print("\n💡 Wskazówka: Uruchom najpierw data_prep.py aby przygotować dane")
        sys.exit(1)
    
    # Menu interaktywne
    while True:
        choice = show_menu()
        
        if choice == '0':
            print("\n👋 Do widzenia!")
            break
        elif choice == '1':
            demo_quick_detect()
            input("\n⏎ Naciśnij Enter aby kontynuować...")
        elif choice == '2':
            demo_batch_processing()
            input("\n⏎ Naciśnij Enter aby kontynuować...")
        elif choice == '3':
            demo_comparison()
            input("\n⏎ Naciśnij Enter aby kontynuować...")
        elif choice == '4':
            demo_api_integration()
            input("\n⏎ Naciśnij Enter aby kontynuować...")
        elif choice == '5':
            # Uruchom wszystkie
            demo_quick_detect()
            input("\n⏎ Naciśnij Enter dla następnego demo...")
            demo_batch_processing()
            input("\n⏎ Naciśnij Enter dla następnego demo...")
            demo_comparison()
            input("\n⏎ Naciśnij Enter dla następnego demo...")
            demo_api_integration()
            input("\n⏎ Naciśnij Enter aby kontynuować...")
        else:
            print("\n❌ Nieprawidłowy wybór, spróbuj ponownie")
            input("\n⏎ Naciśnij Enter aby kontynuować...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Przerwano przez użytkownika")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Nieoczekiwany błąd: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
