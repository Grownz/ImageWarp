# -*- mode: python -*-
# PyInstaller spec for ImageWarp — single Windows x64 .exe with bundled models.
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os

import onnxruntime
import cv2

ONNX_ROOT = os.path.dirname(onnxruntime.__file__)
CV2_ROOT = os.path.dirname(cv2.__file__)


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
    excludes=[],
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
    description="ImageWarp - Warp-Transition-Erzeuger (RAFT/RIFE)",
    disableconsole=False,
    uac=False,
)
