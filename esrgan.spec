# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Real-ESRGAN Windows executable.
Build command: pyinstaller esrgan.spec
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(SPECPATH).resolve() if SPECPATH else Path('.').resolve()

# ── Collect hidden imports ─────────────────────────────────────────────
# basicsr uses dynamic imports extensively — list everything
basicsr_imports = [
    'basicsr',
    'basicsr.archs',
    'basicsr.archs.rrdbnet_arch',
    'basicsr.archs.srvgg_arch',
    'basicsr.archs.stylegan2_arch',
    'basicsr.utils',
    'basicsr.utils.download_util',
    'basicsr.utils.registry',
    'basicsr.utils.options',
    'basicsr.data',
    'basicsr.models',
    'basicsr.losses',
    'basicsr.metrics',
    'basicsr.train',
]

realesrgan_imports = [
    'realesrgan',
    'realesrgan.archs',
    'realesrgan.archs.srvgg_arch',
    'realesrgan.archs.discriminator_arch',
    'realesrgan.utils',
    'realesrgan.version',
]

face_imports = [
    'facexlib',
    'facexlib.detection',
    'facexlib.parsing',
    'facexlib.alignment',
    'facexlib.utils',
    'gfpgan',
    'gfpgan.archs',
    'gfpgan.utils',
]

torch_imports = [
    'torch',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
]

cv_imports = [
    'cv2',
    'cv2.data',
]

hiddenimports = (
    basicsr_imports
    + realesrgan_imports
    + face_imports
    + torch_imports
    + cv_imports
    + [
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'PIL',
        'PIL.Image',
        'tqdm',
        'yaml',
        'addict',
        'scipy',
        'scipy.io',
        'scipy.ndimage',
        'skimage',
        'skimage.metrics',
        'skimage.color',
        'collections.abc',
        'importlib.metadata',
        'tkinterdnd2',
        'tkinterdnd2.TkinterDnD',
    ]
)

# Try to locate tkinterdnd2 package data (Tcl/Tk runtime libraries)
_tkdnd_datas = []
try:
    import tkinterdnd2
    _dnd_dir = Path(tkinterdnd2.__file__).parent
    _tkdnd_dir = _dnd_dir / 'tkdnd'
    if _tkdnd_dir.exists():
        _tkdnd_datas.append((str(_tkdnd_dir), 'tkinterdnd2/tkdnd'))
except Exception:
    pass

# ── Exclude heavy/irrelevant modules ──────────────────────────────────
exclude_modules = [
    # Unused ML / training
    'tensorflow',
    'tensorboard',
    'torch.utils.tensorboard',
    'wandb',
    'mlflow',
    # Unused data science
    'pandas',
    'matplotlib',
    'seaborn',
    'plotly',
    # Unused dev tools
    'IPython',
    'ipykernel',
    'jupyter',
    'jupyter_client',
    'jupyter_core',
    'notebook',
    'nbformat',
    'nbconvert',
    # CUDA (keep only what CPU needs)
    'torch.cuda',
    'torch.distributed',
    'torch.distributions',
    # Testing
    'pytest',
    'unittest',
    'nose',
    # Training-only realesrgan modules
    'realesrgan.data',
    'realesrgan.models',
    'realesrgan.train',
]

# ── Data files to bundle ───────────────────────────────────────────────
datas = []

# Bundle default model weights if they exist
default_weight = ROOT / 'weights' / 'RealESRGAN_x4plus.pth'
if default_weight.exists():
    datas.append((str(default_weight), 'weights'))

# Bundle package data
datas.append((str(ROOT / 'realesrgan' / 'version.py'), 'realesrgan'))

# Bundle tkinterdnd2 runtime data
datas.extend(_tkdnd_datas)

# ── Analysis ──────────────────────────────────────────────────────────
a = Analysis(
    [str(ROOT / 'esrgan_gui.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=exclude_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# ── Filter out CUDA DLLs if present ───────────────────────────────────
# In CPU-only builds, torch may still pull in some CUDA stubs — strip them.
filtered_binaries = []
for name, path, _ in a.binaries:
    lower = name.lower()
    if any(kw in lower for kw in ('cuda', 'cudnn', 'cublas', 'cufft', 'curand', 'cusparse', 'nccl', 'nvtx')):
        continue
    filtered_binaries.append((name, path, 'BINARY'))
a.binaries = filtered_binaries

# ── PYZ ────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ── EXE ────────────────────────────────────────────────────────────────
# Try to find an icon; skip if not available
icon_path = ROOT / 'assets' / 'realesrgan.ico'
icon_str = str(icon_path) if icon_path.exists() else None

exe_kwargs = dict(
    pyz=pyz,
    a_scripts=a.scripts,
    exclude_binaries=True,
    name='Real-ESRGAN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,               # Windows GUI mode: no terminal
    disable_windowed_traceback_ref=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if icon_str:
    exe_kwargs['icon'] = icon_str

exe = EXE(**exe_kwargs)

# ── COLLECT (onedir) ───────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Real-ESRGAN',
)
