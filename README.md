# Vehicle Scan Anomaly Detector

Nowoczesny system wykrywania anomalii na skanach pojazdów przy użyciu YOLO z webowym interfejsem.

## 🚗 Funkcjonalności

- **🤖 AI-powered analiza**: YOLOv8 do wykrywania anomalii na skanach pojazdów
- **🌐 Webowy interface**: Nowoczesny React frontend
- **📊 Interaktywny podgląd**: Zoom, porównanie przed/po analizie z zaznaczonymi anomaliami
- **📈 Szczegółowe raporty**: Eksport do PDF i JSON z mapą anomalii
- **⚡ Real-time status**: Live monitoring procesu analizy
- **📱 Responsive design**: Działa na różnych urządzeniach

## 📁 Struktura projektu

```
Vehicle-Scan-Anomaly-Detector/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # Komponenty React
│   │   └── ...
│   └── package.json
├── data/                    # Dane treningowe
│   ├── czyste/             # Skany bez anomalii (czyste pojazdy)
│   ├── brudne/             # Skany z anomaliami (uszkodzone części)
│   ├── images/train/       # Przygotowane obrazy
│   └── labels/train/       # Etykiety YOLO
├── app.py                  # Flask backend
├── data_prep.py           # Przygotowanie datasetu
├── train_yolo.py          # Trening modelu
├── inference_gui.py       # GUI PyQt (legacy)
├── start.sh               # Skrypt startowy
└── requirements.txt       # Zależności Python
```

## 🚀 Szybki start

### Automatyczne uruchomienie
```bash
./start.sh
```

### Ręczne uruchomienie

1. **Backend (Flask)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend (React)**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **Otwórz przeglądarkę**: http://localhost:3000

## 📊 Używanie aplikacji

1. **Załaduj model**: Aplikacja automatycznie załaduje model YOLO
2. **Prześlij obraz**: Przeciągnij i upuść plik skanu pojazdu (.bmp, .png, .jpg)
3. **Rozpocznij analizę**: Kliknij "Rozpocznij analizę"
4. **Przeglądaj wyniki**: Zobacz wykryte anomalie zaznaczone na obrazie
5. **Pobierz raport**: Eksportuj wyniki do PDF lub JSON

## 🛠️ API Backend

### Endpointy Flask

- `GET /api/health` - Status systemu
- `POST /api/load-model` - Ładowanie modelu YOLO
- `POST /api/upload` - Przesyłanie obrazu
- `POST /api/analyze` - Analiza obrazu
- `GET /api/download-report/<file_id>` - Pobieranie raportu

### Przykład użycia API

```bash
# Upload obrazu
curl -X POST -F "file=@image.bmp" http://localhost:5000/api/upload

# Analiza obrazu
curl -X POST -H "Content-Type: application/json" \
     -d '{"file_id": "your-file-id"}' \
     http://localhost:5000/api/analyze
```

## 🎨 Frontend Features

- **📱 Responsive design**: TailwindCSS + React
- **🖼️ Interaktywny viewer**: Zoom, pan, porównanie obrazów
- **📊 Real-time status**: Live monitoring postępu
- **🎯 Drag & Drop**: Intuitive file upload
- **📈 Detailed results**: Comprehensive analysis display

## 🔧 Rozwój

### Struktura komponentów React

```
src/components/
├── Header.js          # Nagłówek aplikacji
├── UploadArea.js      # Obszar upload plików
├── ImageViewer.js     # Podgląd obrazów z zoom
├── ResultsPanel.js    # Panel wyników analizy
└── StatusPanel.js     # Panel statusu systemu
```

### Dodawanie nowych funkcji

1. **Nowe endpointy**: Dodaj w `app.py`
2. **Nowe komponenty**: Utwórz w `frontend/src/components/`
3. **Stylowanie**: Używaj TailwindCSS classes
4. **State management**: React hooks (useState, useEffect)

## 🧪 Dataset Preparation

```bash
# Przygotowanie danych treningowych
python data_prep.py

# Trenowanie modelu
python train_yolo.py
```

## 📦 Zależności

### Backend (Python)
- Flask + Flask-CORS
- OpenCV + Pillow
- Ultralytics YOLO
- NumPy + tqdm

### Frontend (React)
- React 18
- TailwindCSS
- Axios
- Lucide React (ikony)
- React Dropzone

## 🔒 Bezpieczeństwo

- ✅ CORS properly configured
- ✅ File type validation
- ✅ Size limits for uploads
- ✅ Error handling and logging

## 🐛 Debugowanie

### Backend logs
```bash
# Check Flask logs
python app.py  # Shows debug info
```

### Frontend development
```bash
cd frontend
npm start  # Development server with hot reload
```

### Sprawdzanie API
```bash
# Test health endpoint
curl http://localhost:5000/api/health
```

## 📋 TODO

- [ ] Implement real backend API integration
- [ ] Add user authentication
- [ ] Database integration for results
- [ ] Advanced visualization options
- [ ] Batch processing capability
- [ ] Model performance metrics
- [ ] Export to DICOM format

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 Licencja

MIT License - see LICENSE file for details.
