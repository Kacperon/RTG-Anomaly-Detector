@echo off
REM Skrypt uruchamiający RTG Anomaly Detector na Windows
title RTG Anomaly Detector

echo 🚗 RTG Anomaly Detector - Uruchamianie
echo ======================================

REM Sprawdź czy plik wykonywalny istnieje
if not exist "RTGAnomalyDetector.exe" (
    echo ❌ Nie znaleziono pliku RTGAnomalyDetector.exe
    echo    Upewnij się, że jesteś w właściwym katalogu
    pause
    exit /b 1
)

echo 🚀 Uruchamianie aplikacji...
echo.
echo 📱 Aplikacja zostanie uruchomiona na:
echo    Frontend: http://localhost:3000
echo    Backend:  http://localhost:5000
echo.
echo 🌐 Za chwilę otworzy się przeglądarka...
echo.
echo ⚠️  Nie zamykaj tego okna - aplikacja działa w tle
echo    Aby zatrzymać aplikację, naciśnij Ctrl+C
echo.

REM Uruchom aplikację w tle
start /b RTGAnomalyDetector.exe

REM Poczekaj chwilę na uruchomienie serwera
timeout /t 5 /nobreak >nul

REM Otwórz przeglądarkę (najpierw próbuj localhost:3000, potem 5000)
start http://localhost:5000

echo 🎉 Aplikacja uruchomiona!
echo.
echo Naciśnij dowolny klawisz aby zatrzymać aplikację...
pause >nul

REM Zakończ procesy
taskkill /f /im RTGAnomalyDetector.exe >nul 2>&1
echo 🛑 Aplikacja zatrzymana.
timeout /t 2 /nobreak >nul
