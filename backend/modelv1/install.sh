#!/bin/bash
# install.sh - Skrypt instalacyjny dla systemu detekcji anomalii RTG

set -e

echo "=========================================="
echo "  RTG Anomaly Detector - Instalacja"
echo "=========================================="
echo

# Kolory dla output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funkcje pomocnicze
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Sprawdź Python
echo "Sprawdzanie Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION znaleziony"
else
    print_error "Python 3 nie jest zainstalowany!"
    exit 1
fi

# Sprawdź pip
echo
echo "Sprawdzanie pip..."
if command -v pip3 &> /dev/null; then
    print_success "pip3 znaleziony"
else
    print_error "pip3 nie jest zainstalowany!"
    exit 1
fi

# Utwórz wirtualne środowisko (opcjonalne)
echo
read -p "Czy utworzyć wirtualne środowisko? (zalecane) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Tworzenie wirtualnego środowiska..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Wirtualne środowisko utworzone"
    else
        print_info "Wirtualne środowisko już istnieje"
    fi
    
    echo "Aktywowanie wirtualnego środowiska..."
    source venv/bin/activate
    print_success "Wirtualne środowisko aktywowane"
fi

# Instaluj zależności Python
echo
echo "Instalowanie zależności Python..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Zależności Python zainstalowane"
else
    print_error "Błąd podczas instalacji zależności Python"
    exit 1
fi

# Sprawdź strukturę katalogów
echo
echo "Sprawdzanie struktury katalogów..."
mkdir -p uploads
mkdir -p results
mkdir -p anomaly_reports
mkdir -p data/czyste
mkdir -p data/brudne
print_success "Katalogi utworzone"

# Sprawdź czy są dane
echo
if [ -z "$(ls -A data/czyste)" ]; then
    print_info "Katalog data/czyste jest pusty"
    print_info "Umieść obrazy wzorcowe (czyste) w: data/czyste/"
else
    CLEAN_COUNT=$(find data/czyste -name "*.bmp" | wc -l)
    print_success "Znaleziono $CLEAN_COUNT obrazów wzorcowych"
fi

if [ -z "$(ls -A data/brudne)" ]; then
    print_info "Katalog data/brudne jest pusty"
    print_info "Umieść obrazy testowe (z anomaliami) w: data/brudne/"
else
    DIRTY_COUNT=$(find data/brudne -name "*.bmp" | wc -l)
    print_success "Znaleziono $DIRTY_COUNT obrazów testowych"
fi

# Test importów
echo
echo "Testowanie importów..."
python3 << EOF
import sys
try:
    import cv2
    import numpy
    import scipy
    import skimage
    import flask
    import ultralytics
    print("✅ Wszystkie wymagane moduły są dostępne")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Wszystkie moduły Python zainstalowane poprawnie"
else
    print_error "Niektóre moduły Python nie są dostępne"
    exit 1
fi

# Sprawdź frontend (opcjonalnie)
echo
read -p "Czy zainstalować zależności frontend (React)? [y/N]: " install_frontend
if [[ $install_frontend =~ ^[Yy]$ ]]; then
    if [ -d "frontend" ]; then
        echo "Sprawdzanie Node.js..."
        if command -v npm &> /dev/null; then
            print_success "npm znaleziony"
            
            echo "Instalowanie zależności frontend..."
            cd frontend
            npm install
            if [ $? -eq 0 ]; then
                print_success "Zależności frontend zainstalowane"
            else
                print_error "Błąd podczas instalacji zależności frontend"
            fi
            cd ..
        else
            print_error "npm nie jest zainstalowany!"
            print_info "Zainstaluj Node.js aby używać frontendu"
        fi
    else
        print_info "Katalog frontend nie istnieje"
    fi
fi

# Podsumowanie
echo
echo "=========================================="
echo "  Instalacja zakończona!"
echo "=========================================="
echo
print_success "System detekcji anomalii RTG jest gotowy do użycia"
echo
echo "Kolejne kroki:"
echo
echo "1. Umieść obrazy wzorcowe w: data/czyste/"
echo "2. Umieść obrazy testowe w: data/brudne/"
echo
echo "3. Uruchom demo interaktywne:"
echo "   python demo.py"
echo
echo "4. Lub uruchom szybką detekcję:"
echo "   python -c \"from anomaly_detector import quick_detect; quick_detect('path/to/image.bmp')\""
echo
echo "5. Lub uruchom backend API:"
echo "   python app.py"
echo
echo "6. Lub uruchom pełny web UI:"
echo "   ./start.sh"
echo
echo "Dokumentacja:"
echo "  - README.md - Ogólny przegląd"
echo "  - ANOMALY_DETECTION_GUIDE.md - Szczegółowa dokumentacja"
echo
print_info "Jeśli używasz wirtualnego środowiska, pamiętaj o aktywacji:"
print_info "  source venv/bin/activate"
echo

exit 0
