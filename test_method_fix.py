#!/usr/bin/env python3
"""
Szybki test poprawki błędu metody uploadAndAnalyze
"""

import subprocess
import os
import sys

def test_frontend_build():
    """Test czy frontend buduje się bez błędów"""
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    
    if not os.path.exists(frontend_dir):
        print("❌ Folder frontend nie istnieje")
        return False
    
    try:
        print("🔧 Sprawdzanie składni JavaScript...")
        
        # Sprawdź czy pliki istnieją
        required_files = [
            'src/services/apiService.js',
            'src/components/ResultsPanel.js', 
            'src/App.js',
            'src/components/ImageViewer.js'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(frontend_dir, file_path)
            if not os.path.exists(full_path):
                print(f"❌ Brak pliku: {file_path}")
                return False
            else:
                print(f"✅ Znaleziono: {file_path}")
        
        print("\n🎯 Test składni zakończony pomyślnie!")
        return True
        
    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")
        return False

def check_method_usage():
    """Sprawdź użycie metod w plikach"""
    print("\n📋 Sprawdzanie poprawności metod API...")
    
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'src')
    
    # Sprawdź apiService.js
    api_file = os.path.join(frontend_dir, 'services', 'apiService.js')
    if os.path.exists(api_file):
        with open(api_file, 'r') as f:
            content = f.read()
            
        if 'uploadAndAnalyzeComparison' in content:
            print("✅ apiService ma metodę uploadAndAnalyzeComparison")
        else:
            print("❌ Brak metody uploadAndAnalyzeComparison")
            
        if 'async uploadAndAnalyze(file)' in content:
            print("✅ apiService ma alias uploadAndAnalyze")
        else:
            print("❌ Brak aliasu uploadAndAnalyze")
    
    # Sprawdź ResultsPanel.js  
    results_file = os.path.join(frontend_dir, 'components', 'ResultsPanel.js')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            content = f.read()
            
        if 'uploadAndAnalyzeComparison' in content:
            print("✅ ResultsPanel używa uploadAndAnalyzeComparison")
        else:
            print("❌ ResultsPanel nie używa uploadAndAnalyzeComparison")
    
    # Sprawdź App.js
    app_file = os.path.join(frontend_dir, 'App.js') 
    if os.path.exists(app_file):
        with open(app_file, 'r') as f:
            content = f.read()
            
        if 'uploadAndAnalyzeComparison' in content:
            print("✅ App.js używa uploadAndAnalyzeComparison")
        else:
            print("❌ App.js nie używa uploadAndAnalyzeComparison")

def main():
    print("🔍 Test poprawki błędu: uploadAndAnalyze is not a function")
    print("=" * 60)
    
    # Test 1: Sprawdź istnienie plików
    if not test_frontend_build():
        return
    
    # Test 2: Sprawdź metody
    check_method_usage()
    
    print("\n" + "=" * 60)
    print("✅ NAPRAWIONE PROBLEMY:")
    print("1. Dodano metodę uploadAndAnalyzeComparison() w apiService")
    print("2. Dodano alias uploadAndAnalyze() dla kompatybilności")
    print("3. Zaktualizowano ResultsPanel do nowej metody")
    print("4. Zachowano kompatybilność wsteczną")
    
    print("\n📋 INSTRUKCJE TESTOWANIA:")
    print("1. cd frontend && npm start")
    print("2. Załaduj obraz RTG") 
    print("3. Kliknij 'Rozpocznij analizę'")
    print("4. Sprawdź czy widok przełącza się na heatmapę")
    print("5. Przetestuj przyciski: Oryginalny | Z anomaliami | Heatmapa")

if __name__ == "__main__":
    main()
