# 🚗 Vehicle Scan Anomaly Detector - Quick Guide

## ✅ Co zostało ulepszone:

### 1. 🎯 **Lepsze zaznaczanie anomalii na obrazie**
- **Enhanced bounding boxes**: Zaznaczenia z narożnikami dla lepszej widoczności
- **Kolor-coded detection**: 
  - 🔴 Czerwony: Wysokie prawdopodobieństwo (>70%)
  - 🟠 Pomarańczowy: Średnie prawdopodobieństwo (40-70%)
  - 🟡 Żółty: Niskie prawdopodobieństwo (<40%)
- **Szczegółowe etykiety**: Klasa anomalii + confidence

### 2. 🧠 **Ulepszony model YOLO**
- **Lepsze parametry treningowe**: 
  - Epochs: 50 (zamiast 30)
  - Resolution: 1280 (zamiast 640) 
  - Model: YOLOv8s (zamiast nano)
  - Enhanced data augmentation
- **Niższy próg confidence**: 0.15 (zamiast 0.25) dla lepszej detekcji
- **Auto train/val split**: Automatyczny podział danych na trening/walidację

### 3. 🔍 **Lepsze wykrywanie anomalii**
- **Enhanced preprocessing**: Histogram equalization + adaptive thresholding
- **Better filtering**: Filtrowanie na podstawie area, aspect ratio, solidity
- **Morphological operations**: Opening + closing dla lepszych kształtów

### 4. 🌐 **Zaktualizowany frontend**
- **Vehicle-specific**: Dostosowany do skanów pojazdów
- **Better detection display**: Kolorowe wskaźniki confidence
- **Enhanced info**: Obszar anomalii, pozycja center

## 🚀 **Jak używać:**

### Szybki start:
```bash
./start.sh
```
Otwórz przeglądarkę: http://localhost:3000

### Ręczne uruchomienie:
```bash
# Backend
source venv/bin/activate  
pip install -r requirements.txt
python app.py

# Frontend (w nowym terminalu)
cd frontend
npm install  
npm start
```

### Workflow:
1. **Przygotuj dane**: `python data_prep.py`
2. **Wytrenuj model**: `python train_yolo.py` 
3. **Uruchom app**: `./start.sh`
4. **Załaduj skan pojazdu** w przeglądarce
5. **Rozpocznij analizę** - anomalie będą automatycznie zaznaczone

## 🧪 **Testowanie:**

### Test API:
```bash
python test_enhanced.py
```

### Test z prawdziwymi danymi:
1. Dodaj obrazy do `data/czyste/` (referencyjne)
2. Dodaj obrazy do `data/brudne/` (z anomaliami) 
3. Uruchom: `python data_prep.py`
4. Trenuj: `python train_yolo.py`

## 📊 **Monitoring trenienia:**

Rezultaty trenowania w: `runs/detect/vehicle_anomaly/`
- `weights/best.pt` - najlepszy model
- `results.png` - metryki treningowe
- `confusion_matrix.png` - macierz pomyłek

## 🔧 **Dostosowywanie:**

### Zmiana sensitivity:
W `app.py` linijka ~70:
```python
conf=0.15,  # Niższa wartość = więcej detekcji
```

### Klasy anomalii:
W `data.yaml`:
```yaml
names: ['damage', 'corrosion', 'dent', 'scratch']
nc: 4  # liczba klas
```

---

**🎯 Aplikacja jest teraz zoptymalizowana pod skany pojazdów z lepszym zaznaczaniem anomalii!**
