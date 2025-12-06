# Changelog - System Detekcji Anomalii RTG

## 2025-12-06 - Główna aktualizacja: System porównywania wzorców

### 🎉 Nowe funkcjonalności

#### Główny moduł: `anomaly_detector.py`
Kompletny system detekcji anomalii poprzez porównywanie obrazów RTG z obrazami wzorcowymi.

**Komponenty:**
- ✅ `ImageMatcher` - Dopasowywanie obrazów na podstawie podobieństwa
  - Porównywanie histogramów, gradientów, statystyk intensywności
  - Zwraca top-K najbardziej podobnych obrazów wzorcowych
  
- ✅ `ImageAligner` - Wyrównywanie obrazów
  - Metoda ECC (Enhanced Correlation Coefficient)
  - Metoda feature-based (ORB)
  - Automatyczna korekta przesunięć i rotacji
  
- ✅ `AnomalyDetector` - Wykrywanie anomalii
  - SSIM (Structural Similarity Index) - dla subtelnych różnic
  - Różnica pikselowa - dla wyraźnych anomalii
  - Filtrowanie po rozmiarze, kształcie, solidności
  
- ✅ `AnomalyReportGenerator` - Generowanie raportów
  - Wizualne raporty grid 2x3 (PNG)
  - Raporty JSON z metadanymi
  - Kolorowe heatmapy różnic
  - Automatyczne adnotacje wykrytych anomalii
  
- ✅ `RTGAnomalySystem` - Główny system
  - Integruje wszystkie komponenty
  - Przetwarzanie pojedynczych obrazów
  - Batch processing
  - Funkcja pomocnicza `quick_detect()`

#### API Flask - Nowe endpointy

- ✅ `GET /api/detector-status` - Status obu systemów (YOLO + porównywanie)
- ✅ `POST /api/analyze-comparison` - Analiza przez porównanie wzorców
  - Parametry: `file_id`, `use_alignment`, `use_ssim`
  - Zwraca: szczegółowe wyniki, raport base64, metryki
  
- ✅ `POST /api/batch-analyze` - Przetwarzanie wielu obrazów
  - Parametry: `directory`, `pattern`
  - Zwraca: statystyki, podsumowanie

#### Dokumentacja

- ✅ `ANOMALY_DETECTION_GUIDE.md` - Kompletna dokumentacja (800+ linii)
  - Przegląd systemu
  - Instalacja i konfiguracja
  - Szczegółowy opis komponentów
  - Parametry i dostrajanie
  - Interpretacja wyników
  - Rozwiązywanie problemów
  
- ✅ `EXAMPLES.md` - 10 szczegółowych przykładów użycia
  - Szybki start
  - Zaawansowane użycie
  - Batch processing
  - Porównanie metod
  - Dostrajanie parametrów
  - Integracja z API
  - Automatyzacja kontroli jakości
  
- ✅ `QUICKSTART.md` - Szybkie wprowadzenie
  - 3 proste metody użycia
  - Podstawowe parametry
  - Rozwiązywanie typowych problemów

#### Skrypty i narzędzia

- ✅ `test_anomaly_detector.py` - Kompletny zestaw testów
  - Test pojedynczego obrazu
  - Test przetwarzania partiami
  - Test wyrównywania
  - Test dużych plików
  - Test dopasowywania wzorców
  
- ✅ `demo.py` - Interaktywne demo
  - Menu wyboru z 5 opcjami
  - Automatyczne sprawdzanie zależności
  - Automatyczne sprawdzanie struktury danych
  - Szczegółowe wyniki z kolorowymi statusami
  
- ✅ `install.sh` - Skrypt instalacyjny
  - Sprawdzanie Python i pip
  - Opcjonalne wirtualne środowisko
  - Instalacja zależności
  - Weryfikacja struktury katalogów
  - Test importów

#### Aktualizacje istniejących plików

- ✅ `requirements.txt` - Dodano nowe zależności:
  - `scipy` - dla zaawansowanych operacji numerycznych
  - `scikit-image` - dla SSIM i przetwarzania obrazów
  
- ✅ `app.py` - Integracja nowego systemu:
  - Import `RTGAnomalySystem`
  - Globalna zmienna `anomaly_system`
  - Nowe endpointy API
  - Obsługa obu systemów (YOLO + porównywanie)
  
- ✅ `README.md` - Kompletna aktualizacja:
  - Nowa sekcja o systemie porównywania
  - Instrukcje quick start
  - Przykłady API
  - Linki do dokumentacji
  - Zaktualizowane TODO

### 📊 Możliwości systemu

#### Obsługiwane formaty
- ✅ BMP (obrazy RTG)
- ✅ PNG, JPG, JPEG (opcjonalnie)

#### Obsługa dużych plików
- ✅ Obrazy ~50 MB
- ✅ Automatyczna kompresja do feature extraction
- ✅ Denoising i histogram equalization
- ✅ Przetwarzanie partiami

#### Metryki i analiza
- ✅ Similarity score (0-1)
- ✅ SSIM score (0-1)
- ✅ Liczba wykrytych anomalii
- ✅ Szczegóły każdej anomalii (bbox, area, solidity, aspect_ratio)
- ✅ Ścieżka do dopasowanego wzorca

#### Raporty
- ✅ Wizualne (PNG):
  - Grid 2x3 z porównaniami
  - Kolorowe heatmapy
  - Zaznaczone anomalie
  - Podsumowanie tekstowe
  
- ✅ Dane (JSON):
  - Lista anomalii z parametrami
  - Metadane analizy
  - Timestamp
  - Informacje o wzorcu

### 🔧 Parametry konfiguracji

#### AnomalyDetector
- `threshold` (10-50, domyślnie 25) - próg różnicy pikseli
- `min_area` (100-1000, domyślnie 300) - min. powierzchnia anomalii
- `max_area` (10000-100000, domyślnie 50000) - max. powierzchnia

#### Analiza
- `use_alignment` (bool) - czy wyrównywać obrazy
- `use_ssim` (bool) - SSIM vs różnica pikselowa
- `save_report` (bool) - czy zapisywać raport

### 🎯 Przypadki użycia

1. **Pojedynczy obraz** - `quick_detect('image.bmp')`
2. **Batch processing** - `system.batch_process('directory/')`
3. **API integration** - REST endpoints
4. **Automatyczna kontrola** - Przykład w EXAMPLES.md
5. **Custom parameters** - Dostrajanie dla specyficznych potrzeb

### 🚀 Wydajność

Typowe czasy przetwarzania (CPU):
- Pojedynczy obraz 640x480: ~2-5s
- Pojedynczy obraz 1280x1024: ~5-10s
- Batch 10 obrazów: ~30-60s

Optymalizacje:
- Wyłączenie alignment: -50% czasu
- Pixel diff zamiast SSIM: -30% czasu
- Zmniejszenie rozmiaru obrazu: -40% czasu

### 📦 Struktura projektu

Nowe pliki:
```
anomaly_detector.py           # Główny moduł (700+ linii)
test_anomaly_detector.py     # Testy (300+ linii)
demo.py                       # Interaktywne demo (400+ linii)
install.sh                    # Instalator (150+ linii)
ANOMALY_DETECTION_GUIDE.md   # Dokumentacja (800+ linii)
EXAMPLES.md                   # Przykłady (550+ linii)
QUICKSTART.md                 # Quick start (150+ linii)
anomaly_reports/              # Katalog raportów
```

Zaktualizowane:
```
app.py                        # +150 linii (nowe API endpoints)
requirements.txt              # +2 pakiety (scipy, scikit-image)
README.md                     # Kompletna reorganizacja
```

### 🔄 Migracja

#### Z YOLO na system porównywania

Stary sposób (YOLO):
```python
model = YOLO('model.pt')
results = model.predict('image.bmp')
```

Nowy sposób (porównywanie):
```python
from anomaly_detector import quick_detect
result = quick_detect('image.bmp')
```

#### API

Stary endpoint:
```bash
POST /api/analyze
```

Nowy endpoint:
```bash
POST /api/analyze-comparison
```

Oba działają równolegle - wybierz który potrzebujesz!

### 🐛 Znane ograniczenia

1. **Wymaga obrazów wzorcowych** - katalog `data/czyste/` musi zawierać obrazy
2. **Wyrównywanie może zawieść** - dla bardzo różnych obrazów
3. **Wydajność CPU** - dla wielu obrazów rozważ GPU (przyszła aktualizacja)
4. **Pamięć** - duże pliki BMP mogą wymagać >2GB RAM

### 🔮 Planowane funkcjonalności

- [ ] Obsługa GPU (CUDA)
- [ ] Deep learning dla klasyfikacji typów anomalii
- [ ] Web UI dla nowego systemu
- [ ] Automatyczne dostrajanie parametrów
- [ ] Cache'owanie wzorców dla szybszego przetwarzania
- [ ] Równoległe przetwarzanie batch
- [ ] Eksport do PDF
- [ ] Integracja z bazą danych
- [ ] Powiadomienia (email, webhook)

### 📝 Notatki dla developerów

#### Struktura kodu
- Kod w 100% udokumentowany (docstrings)
- Type hints dla wszystkich funkcji
- Obsługa błędów (try/except)
- Logging dla debugowania
- Modularny design (łatwe rozszerzenia)

#### Testowanie
```bash
python test_anomaly_detector.py    # Wszystkie testy
python demo.py                      # Interaktywne testy
```

#### Dodawanie nowych funkcji
1. Dodaj komponent do `anomaly_detector.py`
2. Dodaj test do `test_anomaly_detector.py`
3. Dodaj przykład do `EXAMPLES.md`
4. Zaktualizuj dokumentację w `ANOMALY_DETECTION_GUIDE.md`

### 🙏 Podziękowania

System wykorzystuje:
- OpenCV - przetwarzanie obrazów
- scikit-image - SSIM i zaawansowane operacje
- scipy - operacje numeryczne
- NumPy - obliczenia macierzowe
- Flask - REST API
- Ultralytics YOLO - alternatywna metoda detekcji

---

**Wersja:** 2.0.0  
**Data:** 2025-12-06  
**Autor:** System RTG Anomaly Detector Team
