@echo off
REM Minimalna kompilacja RTG Anomaly Detector do EXE dla Windows
echo 🔨 RTG Anomaly Detector - Kompilacja EXE
echo =======================================

python --version >nul 2>&1 || (echo ❌ Brak Python && pause && exit /b 1)
node --version >nul 2>&1 || (echo ❌ Brak Node.js && pause && exit /b 1)

REM Wyczyść stare buildy
if exist "venv-build" rmdir /s /q venv-build
if exist "dist" rmdir /s /q dist
if exist "build" rmdir /s /q build

echo 📦 Środowisko wirtualne...
python -m venv venv-build
call venv-build\Scripts\activate.bat

echo 📋 Instalowanie zależności...
python -m pip install --upgrade pip wheel
pip install -r requirements-exe.txt

echo 🌐 Frontend...
cd frontend
npm install --production
npm run build
if %errorlevel% neq 0 (echo ❌ Błąd frontendu && pause && exit /b 1)
cd ..

REM Przygotuj strukturę
if not exist "data\uploads" mkdir data\uploads
if not exist "data\results" mkdir data\results
if not exist "data\anomaly_reports" mkdir data\anomaly_reports
echo. > data\uploads\.gitkeep
echo. > data\results\.gitkeep
echo. > data\anomaly_reports\.gitkeep

echo 🔨 Kompilowanie...
pyinstaller app.spec --clean

if exist "dist\RTGAnomalyDetector.exe" (
    echo ✅ SUKCES: dist\RTGAnomalyDetector.exe
    for %%A in (dist\RTGAnomalyDetector.exe) do echo 📏 Rozmiar: %%~zA bajtów
) else (
    echo ❌ BŁĄD kompilacji
    pause
    exit /b 1
)

call venv-build\Scripts\deactivate.bat
pause
