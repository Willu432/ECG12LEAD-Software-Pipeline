# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

block_cipher = None

# Kumpulkan semua data files
xgboost_datas = collect_data_files('xgboost')
sklearn_datas = collect_data_files('sklearn')
pywt_datas = collect_data_files('pywt')
matplotlib_datas = collect_data_files('matplotlib')
neurokit2_datas = collect_data_files('neurokit2')
numpy_datas = collect_data_files('numpy')
scipy_datas = collect_data_files('scipy')
joblib_datas = collect_data_files('joblib')

# Kumpulkan binary libraries
xgboost_binaries = collect_dynamic_libs('xgboost')
scipy_binaries = collect_dynamic_libs('scipy')
numpy_binaries = collect_dynamic_libs('numpy')

# Hidden imports LENGKAP
hiddenimports = [
    # Core GUI
    'PIL._tkinter_finder',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    
    # Matplotlib - UPDATED UNTUK LIVE PLOT
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_tk',  # BARU
    'matplotlib.figure',
    'matplotlib.pyplot',
    'matplotlib.ticker',
    'matplotlib.animation',  # BARU - PENTING untuk FuncAnimation
    
    # NumPy
    'numpy',
    'numpy.core',
    'numpy.core._multiarray_umath',
    'numpy._core',
    'numpy._core.multiarray',
    'numpy.lib',
    'numpy.linalg',
    'numpy.fft',
    'numpy.random',
    
    # XGBoost
    'xgboost',
    'xgboost.core',
    'xgboost.sklearn',
    
    # Scikit-learn
    'sklearn',
    'sklearn.ensemble',
    'sklearn.tree',
    'sklearn.utils',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    
    # SciPy
    'scipy',
    'scipy.signal',
    'scipy.signal.windows',
    'scipy.stats',
    'scipy.special',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'scipy.sparse.linalg',
    'scipy.integrate',
    'scipy.interpolate',
    'scipy.linalg',
    'scipy._lib',
    
    # Pandas
    'pandas',
    'pandas.core',
    'pandas.core.arrays',
    'pandas.io',
    'pandas.io.excel',
    
    # Signal processing
    'neurokit2',
    'pywt',
    'pywt._extensions._cwt',
    'pywt._extensions._dwt',
    'pywt._extensions._swt',
    
    # Joblib
    'joblib',
    'joblib.externals',
    'joblib.externals.loky',
    
    # Image processing
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    
    # Utilities
    'threading',
    'queue',
    'socket',
    'struct',
    'time',
    'math',
    'textwrap',
    'collections',  # BARU - untuk deque
]

# Tambahkan submodules
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('scipy.signal')
hiddenimports += collect_submodules('neurokit2')
hiddenimports += collect_submodules('xgboost')
hiddenimports += collect_submodules('pywt')
hiddenimports += collect_submodules('joblib')
hiddenimports += collect_submodules('matplotlib')  # BARU
hiddenimports += collect_submodules('matplotlib.backends')  # BARU

# Hapus duplikat
hiddenimports = list(set(hiddenimports))

a = Analysis(
    ['gui_main.py'],
    pathex=[],
    
    binaries=xgboost_binaries + scipy_binaries + numpy_binaries,
    
    datas=[
        ('logo_unpad.png', '.'),
        ('Model Final 2', 'Model Final 2'),
        ('ekg_processor.py', '.'),
        ('ekg_ekstrak_data.py', '.'),
    ] + xgboost_datas + sklearn_datas + pywt_datas + matplotlib_datas + neurokit2_datas + numpy_datas + scipy_datas + joblib_datas,
    
    hiddenimports=hiddenimports,
    
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'test', 'tests', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ONE FILE MODE - PRODUCTION READY
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AplikasiEKG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # NO CONSOLE - tampilan bersih untuk user
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ECG12LEAD_ICON.ico'
)