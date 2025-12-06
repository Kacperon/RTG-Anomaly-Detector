# System Detekcji Anomalii RTG - Przewodnik

## Przegląd

System wykorzystuje zaawansowane algorytmy porównywania obrazów do wykrywania anomalii na obrazach RTG środków transportowych. Działa poprzez:

1. **Znajdowanie najbardziej podobnego obrazu wzorcowego** (czystego)
2. **Wyrównywanie obrazów** (alignment) dla dokładnego porównania
3. **Obliczanie różnic** między obrazami za pomocą SSIM lub różnicy pikselowej
4. **Wykrywanie i klasyfikację anomalii** na podstawie różnic
5. **Generowanie szczegółowych raportów** z wizualizacją

## Funkcjonalności

### ✅ Zaimplementowane

- ✅ Dopasowywanie obrazów na podstawie podobieństwa (histogram, gradienty, statystyki)
- ✅ Wyrównywanie obrazów (ECC, feature-based alignment)
- ✅ Detekcja anomalii za pomocą SSIM (Structural Similarity Index)
- ✅ Detekcja anomalii za pomocą różnicy pikselowej
- ✅ Filtrowanie anomalii (rozmiar, kształt, solidność)
- ✅ Obsługa obrazów RTG w formacie BMP
- ✅ Przetwarzanie partiami (batch processing)
- ✅ Generowanie raportów wizualnych (grid z porównaniami)
- ✅ Generowanie raportów JSON z metadanymi
- ✅ Integracja z API Flask
- ✅ Kolorowe mapy różnic (heatmapy)
- ✅ Automatyczne adnotacje wykrytych anomalii

### 🔧 Możliwości skalowania

- Kompresja obrazów dla dużych plików (~50 MB)
- Przetwarzanie na GPU (możliwe do włączenia)
- Cache'owanie wzorców dla szybszego przetwarzania
- Równoległe przetwarzanie partii

## Instalacja

### 1. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

Nowe wymagane biblioteki:
- `scipy` - dla zaawansowanych operacji numerycznych
- `scikit-image` - dla SSIM i zaawansowanego przetwarzania obrazów

### 2. Przygotuj dane

Struktura katalogów:
```
data/
  czyste/           # Obrazy wzorcowe (bez anomalii)
    202511180021/
      48001F003202511180021.bmp
      ...
  brudne/           # Obrazy testowe (z anomaliami)
    202511190032/
      48001F003202511190032.bmp
      ...
```

## Użycie

### Metoda 1: Bezpośrednie użycie modułu

#### Pojedynczy obraz

```python
from anomaly_detector import quick_detect

# Szybka detekcja
result = quick_detect('path/to/test/image.bmp')

print(f"Anomalia: {result['has_anomaly']}")
print(f"Liczba anomalii: {result['anomaly_count']}")
print(f"Raport: {result['report_path']}")
```

#### Bardziej zaawansowane użycie

```python
from anomaly_detector import RTGAnomalySystem

# Inicjalizuj system
system = RTGAnomalySystem(
    reference_dir='data/czyste',
    output_dir='anomaly_reports'
)

# Przetwórz obraz
result = system.process_image(
    'path/to/test/image.bmp',
    use_alignment=True,   # Wyrównywanie obrazów
    use_ssim=True,        # Użyj SSIM zamiast prostej różnicy
    save_report=True      # Zapisz raport
)

# Wyniki
print(f"Anomalia wykryta: {result['has_anomaly']}")
print(f"Liczba anomalii: {result['anomaly_count']}")
print(f"Podobieństwo do wzorca: {result['similarity']:.2%}")
print(f"SSIM score: {result['ssim_score']:.4f}")

# Szczegóły anomalii
for i, anomaly in enumerate(result['anomalies'], 1):
    print(f"\nAnomalia {i}:")
    print(f"  Położenie (bbox): {anomaly['bbox']}")
    print(f"  Powierzchnia: {anomaly['area']} px²")
    print(f"  Solidność: {anomaly['solidity']:.2f}")
```

#### Przetwarzanie partiami

```python
from anomaly_detector import RTGAnomalySystem

system = RTGAnomalySystem('data/czyste', 'anomaly_reports')

# Przetwórz wszystkie obrazy BMP w katalogu
results = system.batch_process('data/brudne', pattern='*.bmp')

# Statystyki
anomaly_count = sum(1 for r in results if r['has_anomaly'])
print(f"Przetworzono: {len(results)} obrazów")
print(f"Z anomaliami: {anomaly_count}")
print(f"Bez anomalii: {len(results) - anomaly_count}")
```

### Metoda 2: API Flask

#### Uruchom serwer

```bash
python app.py
```

#### Sprawdź status systemu

```bash
curl http://localhost:5000/api/detector-status
```

#### Prześlij i przeanalizuj obraz

```bash
# 1. Prześlij obraz
curl -X POST http://localhost:5000/api/upload \
  -F "file=@path/to/image.bmp"
# Zwraca: {"file_id": "abc123..."}

# 2. Przeanalizuj metodą porównawczą
curl -X POST http://localhost:5000/api/analyze-comparison \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123...",
    "use_alignment": true,
    "use_ssim": true
  }'
```

#### Odpowiedź API

```json
{
  "method": "comparison_based",
  "analysis_complete": true,
  "has_anomaly": true,
  "anomaly_count": 3,
  "anomalies": [
    {
      "id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 10000,
      "solidity": 0.85,
      "aspect_ratio": 1.2,
      "center": [150, 200]
    }
  ],
  "reference_match": "data/czyste/.../image.bmp",
  "similarity": 0.8765,
  "ssim_score": 0.9234,
  "report_image": "base64_encoded_image...",
  "report_path": "anomaly_reports/report_....png",
  "settings": {
    "alignment_used": true,
    "ssim_used": true
  },
  "timestamp": "2025-12-06T10:30:00"
}
```

#### Przetwarzanie partiami przez API

```bash
curl -X POST http://localhost:5000/api/batch-analyze \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "data/brudne",
    "pattern": "*.bmp"
  }'
```

### Metoda 3: Skrypt testowy

```bash
python test_anomaly_detector.py
```

Uruchamia kompletny zestaw testów:
1. ✅ Test pojedynczego obrazu
2. ✅ Test przetwarzania partiami
3. ✅ Test z/bez wyrównywania
4. ✅ Test obsługi dużych plików
5. ✅ Test jakości dopasowywania wzorców

## Komponenty systemu

### 1. ImageMatcher - Dopasowywanie obrazów

Znajduje najbardziej podobny obraz wzorcowy poprzez porównanie:
- Histogramów
- Gradientów (krawędzie)
- Statystyk intensywności
- Momentów obrazu

```python
from anomaly_detector import ImageMatcher

matcher = ImageMatcher('data/czyste')
matches = matcher.find_best_match(test_image, top_k=5)

for match in matches:
    print(f"{match['path']}: {match['similarity']:.2%}")
```

### 2. ImageAligner - Wyrównywanie obrazów

Wyrównuje obrazy dla dokładnego porównania pikselowego:

**Metoda ECC (Enhanced Correlation Coefficient):**
- Transformacja afiniczna
- Dokładniejsze dla niewielkich przesunięć
- Szybsze obliczenia

**Metoda feature-based (ORB):**
- Wykrywanie punktów kluczowych
- Lepsze dla większych różnic
- Bardziej odporne na zniekształcenia

```python
from anomaly_detector import ImageAligner

aligner = ImageAligner()

# Metoda ECC
aligned, transform = aligner.align_images(reference, image, method='ecc')

# Metoda feature-based
aligned, transform = aligner.align_images(reference, image, method='feature')
```

### 3. AnomalyDetector - Wykrywanie anomalii

Wykrywa anomalie poprzez porównanie obrazów:

**SSIM (Structural Similarity Index):**
- Uwzględnia strukturę obrazu
- Lepsze dla niewielkich różnic w jasności
- Bardziej odporne na szum

**Różnica pikselowa:**
- Prosta różnica bezwzględna
- Szybsze obliczenia
- Dobra dla wyraźnych anomalii

```python
from anomaly_detector import AnomalyDetector

detector = AnomalyDetector(
    threshold=25,      # Próg różnicy
    min_area=300,      # Min. powierzchnia anomalii
    max_area=50000     # Max. powierzchnia anomalii
)

result = detector.detect_anomalies(reference, image, use_ssim=True)

print(f"Wykryto: {result['anomaly_count']} anomalii")
print(f"SSIM score: {result['ssim_score']}")
```

### 4. AnomalyReportGenerator - Generowanie raportów

Tworzy kompleksowe raporty wizualne i JSON:

**Raport wizualny (PNG):**
- Grid 2x3 z porównaniami
- Obraz oryginalny i wzorcowy
- Wyrównany obraz
- Mapa różnic (heatmap)
- Zaznaczone anomalie
- Podsumowanie

**Raport JSON:**
- Lista wszystkich anomalii
- Metadane (SSIM, podobieństwo, etc.)
- Timestamp
- Ustawienia analizy

```python
from anomaly_detector import AnomalyReportGenerator

AnomalyReportGenerator.generate_report(
    original_img=test_image,
    reference_img=ref_image,
    aligned_img=aligned,
    detection_result=result,
    output_path='report.png',
    metadata={'custom_field': 'value'}
)
```

## Parametry i dostrajanie

### Dostrajanie detekcji

```python
detector = AnomalyDetector(
    threshold=25,      # ⬇️ niższe = więcej detekcji, więcej false positives
                       # ⬆️ wyższe = mniej detekcji, mniej false positives
    
    min_area=300,      # Minimalna powierzchnia anomalii (px²)
                       # Odfiltruj małe artefakty
    
    max_area=50000     # Maksymalna powierzchnia anomalii (px²)
                       # Odfiltruj bardzo duże różnice
)
```

### Dostrajanie wyrównywania

```python
# Dla niewielkich przesunięć/rotacji
aligned, _ = aligner.align_images(ref, img, method='ecc')

# Dla większych różnic/zniekształceń
aligned, _ = aligner.align_images(ref, img, method='feature')
```

### Wybór metody detekcji

```python
# SSIM - lepsze dla subtelnych różnic
result = detector.detect_anomalies(ref, img, use_ssim=True)

# Różnica pikselowa - szybsza, lepsza dla wyraźnych anomalii
result = detector.detect_anomalies(ref, img, use_ssim=False)
```

## Obsługa dużych plików

System automatycznie obsługuje duże pliki (~50 MB):

1. **Redukowanie rozmiaru dla feature extraction:**
   - Obrazy zmniejszane do 256x256 dla szybkiego porównywania
   
2. **Denoising:**
   - Automatyczne usuwanie szumu dla lepszej detekcji
   
3. **Histogram equalization:**
   - Normalizacja jasności dla lepszego porównania

4. **Przetwarzanie partiami:**
   - Możliwość przetwarzania wielu obrazów równolegle

```python
# Dla bardzo dużych plików można dodatkowo zmniejszyć rozmiar
img = cv2.imread('large_image.bmp', cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

result = system.process_image(img_resized, ...)
```

## Wyniki i interpretacja

### Metryki

- **similarity** (0-1): Jak bardzo testowany obraz jest podobny do wzorca
  - > 0.9: Bardzo podobny
  - 0.7-0.9: Podobny
  - < 0.7: Różny

- **ssim_score** (0-1): Strukturalne podobieństwo
  - > 0.95: Prawie identyczny
  - 0.8-0.95: Podobny
  - < 0.8: Różny

- **anomaly_count**: Liczba wykrytych regionów anomalii

- **area**: Powierzchnia anomalii w pikselach kwadratowych

- **solidity**: Jak "wypełniony" jest kontur (0-1)
  - > 0.8: Zwarta anomalia
  - < 0.5: Nieregularna anomalia

## Rozwiązywanie problemów

### Zbyt wiele false positives

```python
# Zwiększ próg
detector = AnomalyDetector(threshold=35, min_area=500)

# Użyj SSIM zamiast różnicy pikselowej
result = detector.detect_anomalies(ref, img, use_ssim=True)

# Włącz wyrównywanie
result = system.process_image(img, use_alignment=True)
```

### Zbyt mało detekcji

```python
# Zmniejsz próg
detector = AnomalyDetector(threshold=15, min_area=200)

# Wyłącz wyrównywanie jeśli powoduje problemy
result = system.process_image(img, use_alignment=False)
```

### Problemy z dopasowywaniem wzorca

```python
# Sprawdź top dopasowania
matcher = ImageMatcher('data/czyste')
matches = matcher.find_best_match(img, top_k=5)

for i, match in enumerate(matches, 1):
    print(f"{i}. {match['path'].name}: {match['similarity']:.2%}")
```

## Przykłady użycia

### Przykład 1: Automatyczna kontrola jakości

```python
import os
from anomaly_detector import RTGAnomalySystem

system = RTGAnomalySystem('data/czyste', 'reports')

# Przetwórz wszystkie nowe obrazy
new_images_dir = 'data/incoming'
results = system.batch_process(new_images_dir)

# Przenieś obrazy z anomaliami do osobnego katalogu
anomaly_dir = 'data/detected_anomalies'
os.makedirs(anomaly_dir, exist_ok=True)

for result in results:
    if result['has_anomaly']:
        # Przenieś lub skopiuj plik
        print(f"Anomalia w: {result['report_path']}")
```

### Przykład 2: Integracja z web UI

```python
# Backend Flask endpoint (już zaimplementowany)
@app.route('/api/analyze-comparison', methods=['POST'])
def analyze_image_comparison():
    # ... (zobacz app.py)
```

```javascript
// Frontend (przykład)
async function analyzeImage(fileId) {
  const response = await fetch('/api/analyze-comparison', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_id: fileId,
      use_alignment: true,
      use_ssim: true
    })
  });
  
  const result = await response.json();
  
  if (result.has_anomaly) {
    console.log(`Wykryto ${result.anomaly_count} anomalii`);
    // Wyświetl raport
    displayReport(result.report_image);
  }
}
```

## Wydajność

### Typowe czasy przetwarzania (CPU)

- Pojedynczy obraz (640x480): ~2-5 sekund
- Pojedynczy obraz (1280x1024): ~5-10 sekund
- Partia 10 obrazów: ~30-60 sekund

### Optymalizacja

```python
# 1. Wyłącz wyrównywanie dla szybszego przetwarzania
result = system.process_image(img, use_alignment=False)

# 2. Użyj różnicy pikselowej zamiast SSIM
result = system.process_image(img, use_ssim=False)

# 3. Zmniejsz rozmiar obrazu
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
```

## Przyszłe rozszerzenia

- [ ] Obsługa GPU dla szybszego przetwarzania
- [ ] Deep learning dla klasyfikacji anomalii
- [ ] Automatyczne dostrajanie parametrów
- [ ] Web UI dla łatwiejszego użycia
- [ ] Eksport raportów do PDF
- [ ] Integracja z bazą danych
- [ ] REST API z autentykacją
- [ ] Notyfikacje email o wykrytych anomaliach

## Licencja i autorzy

Projekt: RTG Anomaly Detector
Data utworzenia: Grudzień 2025
