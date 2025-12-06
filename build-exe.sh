#!/bin/bash

# RTG Anomaly Detector - Kompilacja do EXE
echo "🔨 RTG Anomaly Detector - Kompilacja do EXE"
echo "============================================"

# Sprawdź czy Python jest zainstalowany
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 nie jest zainstalowany."
    exit 1
fi

# Sprawdź czy Node.js jest zainstalowany (potrzebny do buildu frontendu)
if ! command -v node &> /dev/null; then
    echo "❌ Node.js nie jest zainstalowany. Potrzebny do kompilacji frontendu."
    echo "   Pobierz z: https://nodejs.org/"
    exit 1
fi

# Utwórz środowisko wirtualne dla kompilacji
echo "📦 Tworzenie środowiska wirtualnego..."
if [ -d "venv-build" ]; then
    rm -rf venv-build
fi
python3 -m venv venv-build
source venv-build/bin/activate

# Zainstaluj zależności
echo "📋 Instalowanie zależności dla kompilacji..."
pip install --upgrade pip
pip install -r requirements-exe.txt

# Kompiluj frontend
echo "🌐 Kompilowanie frontendu..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Błąd podczas kompilacji frontendu"
    exit 1
fi
cd ..

# Stwórz puste pliki .gitkeep jeśli nie istnieją
echo "📁 Przygotowywanie struktury katalogów..."
mkdir -p data/uploads data/results data/anomaly_reports
touch data/uploads/.gitkeep 2>/dev/null || true
touch data/results/.gitkeep 2>/dev/null || true
touch data/anomaly_reports/.gitkeep 2>/dev/null || true

# Sprawdź czy modele YOLO istnieją
echo "🧠 Sprawdzanie modeli..."
if [ ! -f "backend/yolov8n.pt" ]; then
    echo "⚠️  Brak modelu yolov8n.pt - zostanie pobrany automatycznie przy pierwszym uruchomieniu"
fi

# Kompilacja z PyInstaller
echo "🔨 Kompilowanie aplikacji..."
pyinstaller app.spec --clean --noconfirm
if [ $? -ne 0 ]; then
    echo "❌ Błąd podczas kompilacji PyInstaller"
    exit 1
fi

# Sprawdź czy kompilacja się udała
if [ -f "dist/RTGAnomalyDetector" ]; then
    echo ""
    echo "✅ Kompilacja zakończona sukcesem!"
    echo ""
    echo "📂 Plik wykonywalny: dist/RTGAnomalyDetector"
    echo "📁 Rozmiar: $(du -h dist/RTGAnomalyDetector | cut -f1)"
    echo ""
    echo "🚀 Aby uruchomić aplikację:"
    echo "   ./dist/RTGAnomalyDetector"
    echo ""
    echo "📝 Uwagi:"
    echo "   • Aplikacja uruchomi się na http://localhost:5000"
    echo "   • Frontend jest wbudowany w aplikację"
    echo "   • Katalogi data/ muszą być w tym samym folderze co plik wykonywalny"
    echo "   • Przy pierwszym uruchomieniu pobrane zostaną modele YOLO"
    echo ""
elif [ -f "dist/RTGAnomalyDetector.exe" ]; then
    echo ""
    echo "✅ Kompilacja zakończona sukcesem!"
    echo ""
    echo "📂 Plik wykonywalny: dist/RTGAnomalyDetector.exe"
    echo "📁 Rozmiar: $(du -h dist/RTGAnomalyDetector.exe | cut -f1)"
    echo ""
    echo "🚀 Aby uruchomić aplikację:"
    echo "   dist\\RTGAnomalyDetector.exe"
    echo ""
else
    echo ""
    echo "❌ Kompilacja nie powiodła się!"
    echo "Sprawdź błędy powyżej."
    exit 1
fi

# Wyczyść środowisko
deactivate
