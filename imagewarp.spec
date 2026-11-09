# -*- mode: python -*-
# PyInstaller spec for ImageWarp — single Windows x64 .exe with bundled models.
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os

import onnxruntime
import cv2

ONNX_ROOT = os.path.dirname(onnxruntime.__file__)
CV2_ROOT = os.path.dirname(cv2.__file__)

# Modules the app actually needs: tkinter (GUI), cv2 (OpenCV), numpy, onnxruntime.
# Everything else is pulled in transitively by PyInstaller's hook system because it
# happens to be installed in the environment. Exclude the heavy unused packages to
# keep the one-file exe lean (target: well under 200 MB instead of multi-GB).
EXCLUDES = [
    # --- Heavy ML / DL frameworks (not used; CPU-only ONNX Runtime) ---
    "torch", "torchvision", "torchaudio",
    "tensorflow", "tensorflow.python", "keras",
    "torch_directml", "torch_cpp_extension",
    # --- Scientific / data stacks (not used) ---
    "scipy", "scipy.*",
    "sklearn", "sklearn.*",
    "pandas", "pandas.*",
    # --- Image / plotting libs (PIL included for preview rendering; matplotlib unused) ---
    "matplotlib", "matplotlib.*",
    "pygments",
    # --- Transformers / NLP ecosystem (not used) ---
    "transformers", "pydantic", "timm", "bitsandbytes",
    "sentencepiece", "sacremoses",
    "google", "google.api_core", "google.auth", "google.cloud",
    "requests_oauthlib", "oauthlib",
    # --- Audio / video / media (not used; OpenCV handles video) ---
    "av", "av.*",
    "torchaudio", "soundfile", "librosa", "resampy",
    "pyav", "pydub",
    # --- Serialization / storage (not used) ---
    "pyarrow", "pyarrow.*",
    "fsspec", "fsspec.*",
    "h5py", "h5py.*",
    "tables",
    "datasets", "datasets.*",
    # --- Scientific / geo / viz (not used) ---
    "skimage", "skimage.*",
    "numba", "numba.*",
    "llvmlite",
    "xarray", "netcdf4",
    "cartopy", "basemap",
    "geopandas", "shapely",
    "folium", "bokeh", "plotly", "dash",
    "streamlit",
    # --- Web frameworks / servers (not used) ---
    "flask", "django",
    "fastapi", "uvicorn", "starlette",
    "celery", "redis",
    # --- Testing / docs (not used) ---
    "pytest", "unittest", "mock",
    "sphinx", "docutils", "hypothesis",
    # --- Misc heavy hooks pulled in by PyInstaller ---
    "triton", "triton.*",
    "win32com", "pythoncom", "pywintypes",
    "pycparser", "cypawn",
    "jinja2", "markupsafe",
    "dateutil",
    "certifi", "charset_normalizer",
    "cloudpickle", "urllib3",
    "sqlite3",
    "backports", "more_itertools",
    "importlib_metadata", "tomli", "wheel", "platformdirs",
    "pkg_resources", "setuptools", "distutils",
    "jaraco", "jaraco.*",
    "ipython", "ipython.*",
    "jedi", "parso",
    "shelve",
    "websockets", "orjson", "anyio", "zoneinfo", "pytz",
    "requests", "requests.*",  # app uses urllib via onnxruntime; requests not needed
]


def collect_package_dlls(root):
    """Return binaries list [(src, dest)] for every .dll under `root`,
    dest preserved relative to root so DLLs land next to their dependents
    inside the one-file bundle (_MEIPATH)."""
    bins = []
    if not os.path.isdir(root):
        return bins
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".dll"):
                src = os.path.join(dirpath, fn)
                dest = os.path.relpath(src, root).replace(os.sep, "/")
                bins.append((src, dest))
    return bins


onnx_bins = collect_package_dlls(ONNX_ROOT)
cv_bins = collect_package_dlls(CV2_ROOT)

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=onnx_bins + cv_bins,
    datas=[
        # Bundle only the models actually used (keeps the exe lean).
        ("models/optical_flow_estimation_raft_2023aug.onnx", "models"),
        ("models/RIFE_fp32.onnx", "models"),
    ],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi",
        "cv2",
        "numpy",
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
    ],
    hookspath=["."],
    hooksdirs=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    cache=None,
    python_version=None,
)

pyz = PYZ(a.pure, block_size=4096, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    block_size=4096,
    cipher=None,
    debug=False,
    bootloader_hook=None,
    icon=None,
    onefile=True,
    name="ImageWarp",
    description="ImageWarp - Warp Transition Generator (RAFT/RIFE)",
    disableconsole=False,
    uac=False,
)
