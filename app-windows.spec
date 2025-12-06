# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Windows-specific configuration
block_cipher = None

# Detect platform
is_windows = sys.platform.startswith('win')

# Dodaj ścieżkę backend do sys.path
current_dir = os.path.abspath(os.getcwd())
backend_path = os.path.join(current_dir, 'backend')
sys.path.insert(0, backend_path)

# Definicje ścieżek - kompatybilne z Windows i Linux
backend_dir = os.path.join(current_dir, 'backend')
frontend_build = os.path.join(current_dir, 'frontend', 'build')
data_dir = os.path.join(current_dir, 'data')

# Pliki ukryte importy - rozszerzone dla Windows
hiddenimports = [
    'flask',
    'flask_cors',
    'cv2',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
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
    'pkg_resources',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements'
]

# Windows-specific hidden imports
if is_windows:
    hiddenimports.extend([
        'win32api',
        'win32con',
        'pywintypes',
        'win32file'
    ])

# Przygotuj dane do dołączenia
datas = []

# Frontend build
if os.path.exists(frontend_build):
    datas.append((frontend_build, 'frontend/build'))

# Modele YOLO
yolo_models = [
    os.path.join(backend_dir, 'yolov8n.pt'),
    os.path.join(backend_dir, 'yolov8s.pt')
]
for model in yolo_models:
    if os.path.exists(model):
        datas.append((model, '.'))

# ModelV1 files
modelv1_dir = os.path.join(backend_dir, 'modelv1')
if os.path.exists(modelv1_dir):
    for ext in ['*.yaml', '*.py']:
        for file in Path(modelv1_dir).glob(ext):
            datas.append((str(file), 'modelv1'))

# ModelV2 files
modelv2_dir = os.path.join(backend_dir, 'modelv2')
if os.path.exists(modelv2_dir):
    for file in Path(modelv2_dir).glob('*.py'):
        datas.append((str(file), 'modelv2'))

# Katalogi danych - stwórz jeśli nie istnieją
data_dirs = ['uploads', 'results', 'anomaly_reports']
for data_subdir in data_dirs:
    full_path = os.path.join(data_dir, data_subdir)
    os.makedirs(full_path, exist_ok=True)
    gitkeep_path = os.path.join(full_path, '.gitkeep')
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w') as f:
            f.write('')
    datas.append((gitkeep_path, f'data/{data_subdir}'))

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
    excludes=[
        'tkinter', 
        'matplotlib.backends._backend_tk',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Usuń niepotrzebne pliki dla Windows
if is_windows:
    a.binaries = [x for x in a.binaries if not x[0].lower().startswith('tk')]
    a.binaries = [x for x in a.binaries if not x[0].lower().startswith('tcl')]
    a.binaries = [x for x in a.binaries if not x[0].lower().startswith('_tk')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Konfiguracja EXE
exe_name = 'RTGAnomalyDetector'
if is_windows:
    exe_name += '.exe'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=exe_name.replace('.exe', ''),  # PyInstaller automatycznie dodaje .exe na Windows
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=not is_windows,  # UPX może powodować problemy na Windows
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # True dla debugowania, False dla produkcji bez konsoli
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Możesz dodać plik .ico dla Windows
)
