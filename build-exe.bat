@echo off
REM RTG Anomaly Detector - Kompilacja do EXE dla Windows
echo 🔨 RTG Anomaly Detector - Kompilacja do EXE
echo ============================================

REM Sprawdź czy Python jest zainstalowany
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python nie jest zainstalowany.
    echo    Pobierz z: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Sprawdź czy Node.js jest zainstalowany
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js nie jest zainstalowany. Potrzebny do kompilacji frontendu.
    echo    Pobierz z: https://nodejs.org/
    pause
    exit /b 1
)

REM Usuń stare środowisko jeśli istnieje
if exist "venv-build" (
    echo 🗑️ Usuwanie starego środowiska...
    rmdir /s /q venv-build
)

REM Utwórz środowisko wirtualne dla kompilacji
echo 📦 Tworzenie środowiska wirtualnego...
python -m venv venv-build
call venv-build\Scripts\activate.bat

REM Zainstaluj zależności
echo 📋 Instalowanie zależności dla kompilacji...
python -m pip install --upgrade pip
pip install -r requirements-exe.txt

REM Kompiluj frontend
echo 🌐 Kompilowanie frontendu...
cd frontend
if not exist "node_modules" (
    npm install
)
npm run build
cd ..

REM Stwórz puste pliki .gitkeep jeśli nie istnieją
echo 📁 Przygotowywanie struktury katalogów...
if not exist "data\uploads" mkdir data\uploads
if not exist "data\results" mkdir data\results
if not exist "data\anomaly_reports" mkdir data\anomaly_reports
echo. > data\uploads\.gitkeep 2>nul
echo. > data\results\.gitkeep 2>nul
echo. > data\anomaly_reports\.gitkeep 2>nul

REM Sprawdź czy modele YOLO istnieją
echo 🧠 Sprawdzanie modeli...
if not exist "backend\yolov8n.pt" (
    echo ⚠️  Brak modelu yolov8n.pt - zostanie pobrany automatycznie przy pierwszym uruchomieniu
)

REM Kompilacja z PyInstaller
echo 🔨 Kompilowanie aplikacji...
pyinstaller app.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo ❌ Błąd podczas kompilacji PyInstaller
    pause
    exit /b 1
)

REM Sprawdź czy kompilacja się udała
if exist "dist\RTGAnomalyDetector.exe" (
    echo.
    echo ✅ Kompilacja zakończona sukcesem!
    echo.
    echo 📂 Plik wykonywalny: dist\RTGAnomalyDetector.exe
    for %%A in (dist\RTGAnomalyDetector.exe) do echo 📁 Rozmiar: %%~zA bajtów
    echo.
    echo 🚀 Aby uruchomić aplikację:
    echo    dist\RTGAnomalyDetector.exe
    echo.
    echo 📝 Uwagi:
    echo    • Aplikacja uruchomi się na http://localhost:5000
    echo    • Frontend jest wbudowany w aplikację
    echo    • Katalogi data\ muszą być w tym samym folderze co plik wykonywalny
    echo    • Przy pierwszym uruchomieniu pobrane zostaną modele YOLO
    echo    • W Windows Defender może być potrzebne dodanie wyjątku
    echo.
) else (
    echo.
    echo ❌ Kompilacja nie powiodła się!
    echo Sprawdź błędy powyżej.
    pause
    exit /b 1
)

REM Wyczyść środowisko
call venv-build\Scripts\deactivate.bat

echo Naciśnij dowolny klawisz aby zakończyć...
pause
