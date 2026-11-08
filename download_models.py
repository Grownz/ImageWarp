"""Download the ONNX models used by ImageWarp.

- RAFT (full, non-quantized): opencv/optical_flow_estimation_raft
  The int8bq variant is skipped because of a known ONNX Runtime
  quantization bug; the full model works with opset 21.
- RIFE: FuryTMP/RIFE_fp32
"""
import os
import requests

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def download(url, out_path):
    print("Downloading from:", url)
    with requests.get(url, stream=True, timeout=60) as r:
        print("status:", r.status_code, "content-length:", r.headers.get("content-length"))
        r.raise_for_status()
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
                    print("  downloaded", total, "bytes", end="\r")
    print()
    print("Done:", out_path, "size:", os.path.getsize(out_path), "bytes")


# RAFT - full non-quantized model (opset 21)
download(
    "https://huggingface.co/opencv/optical_flow_estimation_raft/resolve/main/"
    "optical_flow_estimation_raft_2023aug.onnx",
    os.path.join(MODELS_DIR, "optical_flow_estimation_raft_2023aug.onnx"),
)

# RIFE - fp32
download(
    "https://huggingface.co/FuryTMP/RIFE_fp32/resolve/main/RIFE_fp32.onnx",
    os.path.join(MODELS_DIR, "RIFE_fp32.onnx"),
)

print("All models downloaded.")
