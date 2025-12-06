# RTG Anomaly Detector - Kompilacja do EXE dla Windows

## Wymagania systemowe

### Windows
- Windows 10/11 (64-bit)
- Python 3.8+ (pobierz z https://www.python.org/downloads/)
- Node.js 16+ (pobierz z https://nodejs.org/)
- 4GB RAM minimum, 8GB zalecane
- 2GB wolnego miejsca na dysku

### Linux
- Ubuntu 18.04+ lub podobna dystrybucja
- Python 3.8+
- Node.js 16+
- 4GB RAM minimum, 8GB zalecane

## Instrukcja kompilacji

### Na Windows:
1. Otwórz Command Prompt jako Administrator
2. Przejdź do katalogu projektu:
   ```cmd
   cd C:\ścieżka\do\RTG-Anomaly-Detector
   ```
3. Uruchom kompilację:
   ```cmd
   build-exe.bat
   ```

### Na Linux:
1. Otwórz terminal
2. Przejdź do katalogu projektu:
   ```bash
   cd /ścieżka/do/RTG-Anomaly-Detector
   ```
3. Nadaj uprawnienia i uruchom:
   ```bash
   chmod +x build-exe.sh
   ./build-exe.sh
   ```

## Po kompilacji

### Windows
- Plik wykonywalny: `dist\RTGAnomalyDetector.exe`
- Uruchom przez: `start-windows.bat` lub bezpośrednio `dist\RTGAnomalyDetector.exe`
- Aplikacja uruchomi się na http://localhost:5000

### Linux
- Plik wykonywalny: `dist/RTGAnomalyDetector`
- Uruchom przez: `./dist/RTGAnomalyDetector`
- Aplikacja uruchomi się na http://localhost:5000

## Uwagi dotyczące Windows

### Windows Defender
Windows Defender może oznaczać skompilowany plik jako podejrzany. To normalne dla plików utworzonych przez PyInstaller. Aby to naprawić:

1. Otwórz Windows Security
2. Idź do "Virus & threat protection"
3. Kliknij "Manage settings" pod "Virus & threat protection settings"
4. Dodaj wyjątek dla folderu z aplikacją

### Zależności Windows
Aplikacja zawiera wszystkie potrzebne biblioteki, ale jeśli występują problemy:

1. Zainstaluj Microsoft Visual C++ Redistributable
2. Upewnij się że masz najnowsze Windows Updates

## Rozmiar pliku wykonywalnego

- **Windows**: ~500-800MB (ze względu na biblioteki CV2, PyTorch)
- **Linux**: ~400-600MB
- Pierwszy start może potrwać dłużej (pobieranie modeli YOLO)

## Rozwiązywanie problemów

### Błąd: "Module not found"
- Usuń folder `build/` i `dist/`
- Uruchom kompilację ponownie

### Aplikacja się nie uruchamia
- Sprawdź czy port 5000 jest wolny
- Uruchom plik .exe z wiersza poleceń aby zobaczyć błędy

### Błędy z OpenCV na Windows
- Sprawdź czy masz zainstalowane Visual C++ Redistributable
- Spróbuj uruchomić jako Administrator

### Frontend nie ładuje się
- Upewnij się że `npm run build` zakończył się sukcesem
- Sprawdź folder `frontend/build/`

## Optymalizacja rozmiaru

Aby zmniejszyć rozmiar pliku wykonywalnego:

1. W `app.spec` zmień `upx=True` na `upx=False` jeśli UPX powoduje problemy
2. Usuń niepotrzebne biblioteki z `hiddenimports`
3. Użyj `--exclude-module` dla nieużywanych modułów

## Wsparcie

Jeśli masz problemy:
1. Sprawdź logi w konsoli
2. Upewnij się że wszystkie zależności są zainstalowane
3. Spróbuj uruchomić w trybie deweloperskim: `python backend/app.py`
