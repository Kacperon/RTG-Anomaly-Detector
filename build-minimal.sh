#!/bin/bash

# Minimalna kompilacja RTG Anomaly Detector do EXE
echo "🔨 RTG Anomaly Detector - Kompilacja EXE"
echo "======================================="

# Sprawdź Python
python3 --version || { echo "❌ Brak Python3"; exit 1; }

# Sprawdź Node.js
node --version || { echo "❌ Brak Node.js"; exit 1; }

# Wyczyść stare buildy
rm -rf venv-build dist build

# Utwórz środowisko
echo "📦 Środowisko wirtualne..."
python3 -m venv venv-build
source venv-build/bin/activate

# Zainstaluj tylko niezbędne do kompilacji
echo "📋 Instalowanie zależności..."
pip install --upgrade pip wheel
pip install -r requirements-exe.txt

# Kompiluj frontend
echo "🌐 Frontend..."
cd frontend
npm install --production
npm run build
cd ..

# Przygotuj strukturę
mkdir -p data/{uploads,results,anomaly_reports}
touch data/uploads/.gitkeep data/results/.gitkeep data/anomaly_reports/.gitkeep

# Kompiluj
echo "🔨 Kompilowanie..."
pyinstaller app.spec --clean

# Sprawdź wynik
if [ -f "dist/RTGAnomalyDetector" ]; then
    echo "✅ SUKCES: dist/RTGAnomalyDetector"
    echo "📏 Rozmiar: $(du -h dist/RTGAnomalyDetector | cut -f1)"
else
    echo "❌ BŁĄD kompilacji"
    exit 1
fi

deactivate
