# Funkcjonalność Porównywania Kolorów - Model v2

## Opis
Nowa funkcjonalność w `backend/modelv2/detector.py` pozwala na:

1. **Wykrycie największego obiektu** na obrazie wejściowym
2. **Wycięcie regionu obiektu** z 10% marginesem 
3. **Znalezienie najbardziej podobnego obrazu** w folderze `procesed_imagines`
4. **Utworzenie mapy różnic kolorów** gdzie czerwony oznacza największe różnice

## Jak to działa

### 1. Wykrywanie obiektu
```python
# Model wykrywa największy obiekt na obrazie
detection_result = detector.detect_anomalies(image_path, include_bounds=True)
```

### 2. Wycinanie regionu z marginesem
```python
# Wyciąga region obiektu z 10% marginesem
object_region = detector._extract_object_region(image, bbox, margin_percent=0.1)
```

### 3. Szukanie najlepszego dopasowania
```python
# Porównuje z każdym obrazem w folderze procesed_imagines
# Używa MSE (Mean Squared Error) jako metryki różnicy
matching_result = detector._find_best_matching_image(object_region, processed_images_dir)
```

### 4. Tworzenie mapy kolorów
```python
# Tworzy mapę różnic gdzie:
# - Czerwony = największa różnica kolorów
# - Niebieski = najmniejsza różnica kolorów  
color_diff_map = detector._create_color_difference_map(object_region, best_match_img)
```

## Użycie

### Podstawowe użycie
```python
from backend.modelv2.detector import find_and_compare_with_processed_images, load_model

# Załaduj model
load_model()

# Uruchom porównanie
result = find_and_compare_with_processed_images(
    image_path="path/to/your/image.jpg",
    output_dir="results"  # opcjonalnie - folder do zapisu wyników
)

if result["success"]:
    print(f"Najlepsze dopasowanie: {result['best_match_path']}")
    print(f"Różnica kolorów (MSE): {result['best_match_difference']}")
else:
    print(f"Błąd: {result['error']}")
```

### Wynik zawiera:
```python
{
    "success": True,
    "detection_result": {...},           # Wyniki detekcji YOLO
    "object_bbox": [x1, y1, x2, y2],    # Współrzędne obiektu
    "object_region_shape": (h, w, c),   # Rozmiar wyciętego regionu
    "best_match_path": "path/to/best.jpg", # Najlepiej dopasowany obraz
    "best_match_difference": 1234.56,   # Wartość różnicy (MSE)
    "matching_result": {                 # Szczegóły porównania
        "total_candidates": 10,
        "comparison_results": [...]      # Lista wszystkich porównań
    },
    "object_region_saved": "results/object_region_timestamp.jpg",
    "color_diff_map_saved": "results/color_diff_map_timestamp.jpg", 
    "comparison_saved": "results/comparison_timestamp.jpg"
}
```

## Generowane pliki

Po uruchomieniu z `output_dir` funkcja zapisuje 3 pliki:

1. **object_region_*.jpg** - Wycięty region obiektu (z 10% marginesem)
2. **color_diff_map_*.jpg** - Mapa różnic kolorów (czerwony = max różnica)
3. **comparison_*.jpg** - Porównanie side-by-side: original | best_match | diff_map

## Struktura folderów

```
RTG-Anomaly-Detector/
├── procesed_imagines/           # Folder z obrazami do porównania
│   ├── image1.jpg              # (przykładowe obrazy)
│   ├── image2.bmp
│   └── ...
├── results/                    # Folder wyników (tworzony automatycznie)
│   ├── object_region_*.jpg
│   ├── color_diff_map_*.jpg
│   └── comparison_*.jpg
└── backend/modelv2/
    └── detector.py            # Główna logika
```

## Wymagania

- Wszystkie potrzebne pakiety już są w `backend/requirements.txt`
- Model YOLO jest pobierany automatycznie przy pierwszym użyciu
- Folder `procesed_imagines` musi istnieć i zawierać obrazy do porównania

## Przykład użycia z API

```python
# Możesz dodać endpoint do backend/app.py:

@app.route('/api/compare-colors', methods=['POST'])
def compare_colors():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Zapisz upload
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)
    
    # Uruchom porównanie
    result = find_and_compare_with_processed_images(upload_path, 'results')
    
    return jsonify(result)
```

## Metryki i interpretacja

- **MSE (Mean Squared Error)**: Im mniejsza wartość, tym bardziej podobne obrazy
- **Mapa kolorów**: 
  - Czerwony/Żółty = duże różnice kolorów
  - Niebieski/Fioletowy = małe różnice kolorów
- **Expansion factor**: 0.1 oznacza 10% margines wokół obiektu

## Rozwiązywanie problemów

1. **"Folder nie istnieje"**: Utwórz folder `procesed_imagines` 
2. **"Brak obrazów"**: Umieść pliki .jpg/.png/.bmp w `procesed_imagines`
3. **"Brak wykrytych obiektów"**: Model nie wykrył żadnego obiektu na obrazie
4. **Wysokie wartości MSE**: Obrazy są bardzo różne (to może być oczekiwane)
