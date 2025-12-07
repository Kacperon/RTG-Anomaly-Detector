# Maskowanie Tła w Systemie Detekcji Anomalii RTG

## Przegląd

System został rozszerzony o zaawansowane maskowanie tła, które ignoruje prawie białe obszary podczas wykrywania anomalii. To kluczowe ulepszenie dla obrazów RTG, gdzie jasne tło może generować fałszywe pozytywne wyniki.

## Nowe Funkcjonalności

### 1. Automatyczne Wykrywanie Tła

System oferuje trzy metody wykrywania obszarów tła:

#### Metoda Otsu (`'otsu'`)
- **Zalecana** dla większości przypadków RTG
- Automatycznie znajduje optymalny próg podziału
- Dobra dla obrazów z wyraźnym kontrastem między obiektem a tłem

#### Metoda Adaptacyjna (`'adaptive'`)
- Używa lokalnego progowania
- Lepsza dla obrazów z nierównomiernym oświetleniem
- Może być przydatna dla skomplikowanych struktur RTG

#### Metoda Progowa (`'threshold'`)
- Stały próg (domyślnie 240/255)
- Szybka i prosta
- Dobra gdy znamy charakterystykę tła w obrazach

### 2. Nowe Parametry Konfiguracyjne

#### Klasa `AnomalyDetector`
```python
detector = AnomalyDetector(
    threshold=25,                    # Próg różnicy pikseli
    min_area=300,                   # Min. powierzchnia anomalii
    max_area=50000,                 # Max. powierzchnia anomalii
    background_threshold=240        # Próg dla białego tła (nowy)
)
```

#### Metoda `detect_anomalies`
```python
result = detector.detect_anomalies(
    reference_img, test_img,
    use_ssim=True,                  # Użyj SSIM
    ignore_background=True,         # Ignoruj tło (nowy)
    background_method='otsu'        # Metoda wykrywania tła (nowy)
)
```

#### Metoda `process_image` w `RTGAnomalySystem`
```python
result = system.process_image(
    image_path,
    use_alignment=True,
    use_ssim=True,
    save_report=True,
    ignore_background=True          # Ignoruj tło (nowy)
)
```

## Przykłady Użycia

### Podstawowe Użycie z Maskowaniem

```python
from backend.anomaly_detector import RTGAnomalySystem

# System z automatycznym maskowaniem tła
system = RTGAnomalySystem('data/czyste', 'results')

# Analiza z maskowaniem tła (zalecane)
result = system.process_image(
    'test_image.bmp',
    ignore_background=True  # Włącz maskowanie
)

print(f"Wykryto anomalii: {result['anomaly_count']}")
```

### Porównanie z/bez Maskowania

```python
# Bez maskowania tła
result_no_mask = system.process_image(
    'test_image.bmp',
    ignore_background=False
)

# Z maskowaniem tła
result_with_mask = system.process_image(
    'test_image.bmp',  
    ignore_background=True
)

print(f"Bez maskowania: {result_no_mask['anomaly_count']} anomalii")
print(f"Z maskowaniem:  {result_with_mask['anomaly_count']} anomalii")
```

### Testowanie Różnych Metod

```python
detector = AnomalyDetector()

methods = ['otsu', 'adaptive', 'threshold']
for method in methods:
    result = detector.detect_anomalies(
        reference_img, test_img,
        ignore_background=True,
        background_method=method
    )
    print(f"{method}: {result['anomaly_count']} anomalii")
```

## Korzyści

### 1. Redukcja Fałszywych Pozytywów
- Eliminuje wykrywanie różnic w obszarach tła
- Skupia się na rzeczywistych strukturach anatomicznych
- Poprawia precyzję detekcji

### 2. Lepsze Wykrywanie ROI
- Automatycznie identyfikuje obszary zainteresowania
- Ignoruje artefakty na krawędziach obrazu
- Optymalizuje dla struktury obrazów RTG

### 3. Konfigurowalność
- Różne metody dla różnych typów obrazów
- Regulowane progi dla specyficznych wymagań
- Łatwe włączanie/wyłączanie funkcji

## Diagnostyka

### Informacje Debug
System wypisuje informacje o procesie maskowania:

```
🎯 Zastosowano maskę ROI (obszary nie-tła)
   Procent obszaru ROI: 67.3%
```

### Testowanie
Użyj skryptu testowego:

```bash
python test_background_masking.py
```

Generuje porównanie wyników i wizualizacje masek.

## Ustawienia Zalecane

### Dla Standardowych RTG
```python
system = RTGAnomalySystem('data/czyste')
result = system.process_image(
    image_path,
    ignore_background=True,      # Włączone
    use_ssim=True,              # Zalecane dla RTG
    use_alignment=True          # Dla lepszego dopasowania
)
```

### Dla Problematycznych Obrazów
Jeśli standardowe ustawienia nie działają:

1. **Spróbuj metody adaptacyjnej:**
```python
detector.detect_anomalies(
    ref, img, 
    background_method='adaptive'
)
```

2. **Dostosuj próg tła:**
```python
detector = AnomalyDetector(background_threshold=220)
```

3. **Wyłącz maskowanie dla bardzo ciemnych obrazów:**
```python
result = system.process_image(
    image_path,
    ignore_background=False
)
```

## Uwagi Techniczne

### Wydajność
- Maskowanie Otsu: +5-10ms na obraz
- Metoda progowa: +1-2ms na obraz
- Operacje morfologiczne: +2-3ms na obraz

### Kompatybilność
- Wszystkie istniejące API zachowują kompatybilność
- Nowe parametry są opcjonalne z sensownymi domyślnymi
- Można łatwo włączyć/wyłączyć nowe funkcje

### Ograniczenia
- Może usunąć małe anomalie blisko krawędzi
- Wymaga dostrojenia dla specyficznych typów RTG
- Najlepsze wyniki dla obrazów z wyraźnym kontrastem

## Rozwiązywanie Problemów

### Zbyt Mało Wykrytych Anomalii
1. Sprawdź procent ROI - czy nie jest zbyt mały?
2. Spróbuj metody `'threshold'` z niższym progiem
3. Rozważ wyłączenie maskowania dla tego typu obrazów

### Zbyt Dużo Fałszywych Pozytywów
1. Upewnij się, że maskowanie jest włączone
2. Spróbuj metody `'otsu'` zamiast `'threshold'`
3. Zwiększ `background_threshold`

### Problemy z Określoną Metodą
- `'otsu'` może nie działać dla obrazów o niskim kontraście
- `'adaptive'` może być zbyt agresywna dla prostych przypadków  
- `'threshold'` wymaga ręcznego dostrojenia progu
