# ImageWarp — Warp-Transition-Erzeuger

Lean Windows x64 GUI app that creates warp transitions between two images or two videos using slim ONNX models.

## Technique

- **Image mode (RAFT):** dense optical flow A→B via [RAFT](https://github.com/google/RAFT) (ONNX). Intermediate frame `t` warps A by `t·flow` and B by `-(1−t)·flow`, then crossfades.
- **Video mode (RIFE):** frame interpolation via [RIFE](https://github.com/hzwer/RIFE) (ONNX) between the **last frame of video A** and the **first frame of video B**.

## Stack

- Python 3.10 + tkinter (GUI)
- ONNX Runtime 1.23.2 (inference)
- OpenCV 4.10.0 (image/video processing, warping, remap)
- PyInstaller 6.22.3 (single `.exe`)

## Models

The models are **not** part of the repository (large binaries). Download them with:

```bash
python download_models.py
```

This fetches:
- `models/optical_flow_estimation_raft_2023aug.onnx` (full RAFT, ~64 MB) from `opencv/optical_flow_estimation_raft`
- `models/RIFE_fp32.onnx` (~22 MB) from `FuryTMP/RIFE_fp32`

> The quantized RAFT `int8bq` variant is **not** used (quantization bug in ONNX Runtime); the full non-quantized model works with opset 21.

## Build

```bash
pyinstaller imagewarp.spec
```
Produces a single-file `dist/ImageWarp.exe` (Windows x64).

## Usage

1. Launch `ImageWarp.exe`.
2. Choose source (A) and target (B) — images or videos.
3. Set the number of transition frames.
4. **Preview** the transition, then **export** as MP4/AVI video.

The GUI is in German; model output / inference logs are in English.

## Files

- `app.py` — main GUI + RAFT/RIFE backends
- `imagewarp.spec` — PyInstaller spec
- `download_models.py` — model download script
- `test_raft.py` — RAFT sanity test (shifted-image verification)
