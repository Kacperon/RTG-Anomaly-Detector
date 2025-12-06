# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path
import glob

# Dodaj ścieżkę backend do sys.path
backend_path = os.path.join(os.getcwd(), 'backend')
sys.path.insert(0, backend_path)

# Definicje ścieżek
current_dir = os.getcwd()
backend_dir = os.path.join(current_dir, 'backend')
frontend_build = os.path.join(current_dir, 'frontend', 'build')
data_dir = os.path.join(current_dir, 'data')

# Przygotuj listę plików danych
datas = []

# Frontend build (tylko jeśli istnieje)
if os.path.exists(frontend_build):
    for root, dirs, files in os.walk(frontend_build):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, frontend_build)
            datas.append((full_path, os.path.join('frontend', 'build', rel_path)))

# Modele YOLO
for pt_file in glob.glob(os.path.join(backend_dir, '*.pt')):
    datas.append((pt_file, '.'))

# Pliki modelv1
modelv1_dir = os.path.join(backend_dir, 'modelv1')
if os.path.exists(modelv1_dir):
    for ext in ['*.yaml', '*.py']:
        for file in glob.glob(os.path.join(modelv1_dir, ext)):
            datas.append((file, 'modelv1'))

# Pliki modelv2
modelv2_dir = os.path.join(backend_dir, 'modelv2')
if os.path.exists(modelv2_dir):
    for file in glob.glob(os.path.join(modelv2_dir, '*.py')):
        datas.append((file, 'modelv2'))

# Katalogi danych
for folder in ['uploads', 'results', 'anomaly_reports']:
    folder_path = os.path.join(data_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    gitkeep_path = os.path.join(folder_path, '.gitkeep')
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w') as f:
            f.write('')
    datas.append((gitkeep_path, f'data/{folder}'))
# Pliki ukryte importy (potrzebne dla niektórych bibliotek)
hiddenimports = [
    'flask',
    'flask_cors',
    'cv2',
    'PIL',
    'PIL.Image',
    'numpy',
    'ultralytics',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.models.yolo.detect',
    'ultralytics.utils',
    'torch',
    'torchvision',
    'scipy',
    'scipy.ndimage',
    'skimage',
    'skimage.metrics',
    'skimage.feature',
    'reportlab',
    'reportlab.pdfgen',
    'reportlab.lib',
    'yaml',
    'pkg_resources.py2_warn',
    'modelv1.detector',
    'modelv2.detector',
    'anomaly_detector'
]

# Analiza głównego pliku
a = Analysis(
    [os.path.join(backend_dir, 'app.py')],
    pathex=[backend_dir, current_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.backends._backend_tk'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Usuń niepotrzebne pliki dla zmniejszenia rozmiaru
a.binaries = [x for x in a.binaries if not x[0].startswith('tk')]
a.binaries = [x for x in a.binaries if not x[0].startswith('tcl')]

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RTGAnomalyDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Kompresja UPX dla zmniejszenia rozmiaru
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Pokaż konsolę dla debugowania - można zmienić na False dla Windows bez konsoli
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Możesz dodać plik .ico dla Windows
    version=None,  # Możesz dodać informacje o wersji dla Windows
)
