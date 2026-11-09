#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageWarp — Lean Windows x64 GUI for warp transitions between two images/videos.

Image mode (RAFT): dense optical flow A->B; intermediate frame t warps A by t*flow
and B by -(1-t)*flow, then crossfades.
Video mode (RIFE): frame interpolation between last frame of video A and first
frame of video B.

Backend: tkinter (GUI) + ONNX Runtime (inference) + OpenCV (image/video).
"""
import os
import sys
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths (works both as script and as PyInstaller one-file executable)
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    MEIPATH = getattr(sys, "_MEIPATH", None)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MEIPATH = None


def _model_path(name):
    """Locate a model file: bundled inside one-file exe (_MEIPATH) first,
    then next to the exe/script (for dev or manual placement)."""
    if MEIPATH:
        p = os.path.join(MEIPATH, "models", name)
        if os.path.exists(p):
            return p
    p = os.path.join(BASE_DIR, "models", name)
    if os.path.exists(p):
        return p
    return p  # fallback (will raise a clear error on load)


RAFT_MODEL = _model_path("optical_flow_estimation_raft_2023aug.onnx")
RIFE_MODEL = _model_path("RIFE_fp32.onnx")

# RAFT fixed input resolution (H x W) as trained.
RAFT_H, RAFT_W = 360, 480
# Output frame size (W x H) for preview/export (2x RAFT resolution, aspect kept).
OUT_W, OUT_H = 960, 720

# ---------------------------------------------------------------------------
# Inference backends (singleton sessions)
# ---------------------------------------------------------------------------
class Backend:
    _raft_sess = None
    _rif_e_sess = None

    @classmethod
    def raft(cls):
        if cls._raft_sess is None:
            cls._raft_sess = ort.InferenceSession(
                RAFT_MODEL, providers=["CPUExecutionProvider"]
            )
        return cls._raft_sess

    @classmethod
    def rife(cls):
        if cls._rif_e_sess is None:
            cls._rif_e_sess = ort.InferenceSession(
                RIFE_MODEL, providers=["CPUExecutionProvider"]
            )
        return cls._rif_e_sess

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def resize_to(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def to_float_chw(img_bgr):
    """HWC uint8 BGR -> CHW float32 [0,1] with batch dim -> [1,3,H,W]."""
    img = img_bgr.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0).astype(np.float32)

def chw_to_hwc_uint8(arr):
    """[1,3,H,W] or [3,H,W] float [0,1] -> HWC uint8 BGR."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr

# ---------------------------------------------------------------------------
# RAFT optical flow
# ---------------------------------------------------------------------------
def raft_flow(backend, img_a, img_b):
    """Dense optical flow [H,W,2] from img_a to img_b (both HWC uint8 BGR)."""
    sess = backend.raft()
    a = resize_to(img_a, RAFT_W, RAFT_H)
    b = resize_to(img_b, RAFT_W, RAFT_H)
    inp1 = to_float_chw(a)
    inp2 = to_float_chw(b)
    outs = sess.run(None, {"0": inp1, "1": inp2})
    # last output is the upsampled flow [1,2,H,W]
    flow = outs[-1]
    flow = np.squeeze(flow).transpose(1, 2, 0)  # [H,W,2] (u,v)
    return flow

def warp(img, disp):
    """output(q) = img(q - disp). disp [H,W,2] (u,v)."""
    h, w = img.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (grid_x - disp[..., 0]).astype(np.float32)
    map_y = (grid_y - disp[..., 1]).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

def raft_transition_frames(img_a, img_b, num_frames):
    """List of num_frames frames (HWC uint8 BGR) morphing img_a -> img_b via RAFT."""
    # RAFT operates at its fixed resolution; resize inputs there first.
    a = resize_to(img_a, RAFT_W, RAFT_H)
    b = resize_to(img_b, RAFT_W, RAFT_H)
    flow = raft_flow(Backend, a, b)
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0.0
        wa = warp(a, flow * t)
        wb = warp(b, flow * (-(1.0 - t)))
        out = cv2.addWeighted(wa, 1.0 - t, wb, t, 0)
        frames.append(out)
    return frames


# ---------------------------------------------------------------------------
# RIFE frame interpolation
# ---------------------------------------------------------------------------
def rife_midpoint(backend, f0, f1):
    """RIFE: given two frames (HWC uint8 BGR), return interpolated midpoint frame."""
    sess = backend.rife()
    h, w = f0.shape[:2]
    a = resize_to(f0, w, h)
    b = resize_to(f1, w, h)
    x = np.concatenate(
        [to_float_chw(a)[0], to_float_chw(b)[0]], axis=0
    )  # [6,H,W]
    x = np.expand_dims(x, axis=0)  # [1,6,H,W]
    out = sess.run(None, {"input": x})[0]
    return chw_to_hwc_uint8(out)


def rife_transition_frames(f0, f1, num_frames):
    """List of num_frames frames (HWC uint8 BGR) interpolating f0 -> f1 via RIFE.

    Uses iterative dyadic bisection: at each level every current interval is
    split by one RIFE midpoint call. After enough levels we have dyadic frames
    at t = k/2^d, which are then linearly resampled to the requested even grid.
    """
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [f0]
    frames = {0.0: f0, 1.0: f1}
    # intervals to split: (t0, t1, frame_at_t0, frame_at_t1)
    intervals = [(0.0, 1.0, f0, f1)]
    while len(frames) < num_frames:
        nxt = []
        for t0, t1, ft0, ft1 in intervals:
            tm = 0.5 * (t0 + t1)
            fm = rife_midpoint(Backend, ft0, ft1)
            frames[tm] = fm
            nxt.append((t0, tm, ft0, fm))
            nxt.append((tm, t1, fm, ft1))
        intervals = nxt
        if len(frames) >= num_frames:
            break
    items = sorted(frames.items())  # list of (t, frame)
    out = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        # bracket t between adjacent dyadic points
        lo = items[0]
        hi = items[-1]
        for k in range(len(items) - 1):
            t0_, f0_ = items[k]
            t1_, f1_ = items[k + 1]
            if t0_ <= t <= t1_:
                lo, hi = (t0_, f0_), (t1_, f1_)
                break
        if hi[0] == lo[0]:
            out.append(lo[1])
        else:
            alpha = (t - lo[0]) / (hi[0] - lo[0])
            blended = cv2.addWeighted(lo[1], 1.0 - alpha, hi[1], alpha, 0)
            out.append(blended)
    return out


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------
def read_video_last_frame(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    last = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last = frame
    cap.release()
    if last is None:
        raise RuntimeError(f"No frames in video: {path}")
    return last


def read_video_first_frame(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"No frames in video: {path}")
    return frame


def is_image(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_video(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webp", ".mpeg", ".mpg"}


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return img


# ---------------------------------------------------------------------------
# Frame -> output size resize
# ---------------------------------------------------------------------------
def fit_to_output(frame):
    """Resize a frame to OUT_W x OUT_H keeping aspect (letterbox black)."""
    h, w = frame.shape[:2]
    scale = min(OUT_W / w, OUT_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    x = (OUT_W - nw) // 2
    y = (OUT_H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ImageWarp — Warp-Transition-Erzeuger")
        self.root.resizable(False, False)

        self.path_a = tk.StringVar()
        self.path_b = tk.StringVar()
        self.mode = tk.StringVar(value="auto")  # auto | raft | rife
        self.num_frames = tk.IntVar(value=30)

        self._build_ui()

        # Backend state
        self.frames = []  # list of frames already resized to output size
        self.preview_thread = None
        self.preview_stop = threading.Event()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        f = ttk.Frame(self.root, padding=12)
        f.pack(fill=tk.BOTH, expand=True)

        # File A
        row = ttk.Frame(f); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Quelldatei (A):").pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=self.path_a, width=50)
        ent.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(row, text="Durchsuchen…", command=self.pick_a).pack(side=tk.LEFT)

        # File B
        row = ttk.Frame(f); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Zieldatei (B):").pack(side=tk.LEFT)
        ent = ttk.Entry(row, textvariable=self.path_b, width=50)
        ent.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(row, text="Durchsuchen…", command=self.pick_b).pack(side=tk.LEFT)

        # Mode
        row = ttk.Frame(f); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Modus:").pack(side=tk.LEFT)
        cb = ttk.Combobox(
            row, textvariable=self.mode, width=12,
            values=["auto", "Bild (RAFT)", "Video (RIFE)"]
        )
        cb.pack(side=tk.LEFT, padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())

        # Transition length
        row = ttk.Frame(f); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Übergangsrahmen:").pack(side=tk.LEFT)
        sp = ttk.Spinbox(row, from_=2, to=120, textvariable=self.num_frames, width=6)
        sp.pack(side=tk.LEFT, padx=6)

        # Buttons
        row = ttk.Frame(f); row.pack(fill=tk.X, pady=8)
        ttk.Button(row, text="Vorschau", command=self.preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Als Video exportieren…", command=self.export).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Beenden", command=self.root.quit).pack(side=tk.LEFT, padx=4)

        # Status
        self.status = ttk.Label(f, text="Bereit", anchor=tk.W)
        self.status.pack(fill=tk.X, pady=2)

        # Preview canvas (embedded)
        self.canvas = tk.Canvas(f, width=OUT_W, height=OUT_H, bg="black")
        self.canvas.pack(pady=6)
        self.canvas_img = None

    def _on_mode_change(self):
        # nothing dynamic needed; mode resolved at run time
        pass

    # ------------------------------------------------------------------ helpers
    def pick_a(self):
        path = filedialog.askopenfilename(
            title="Quelldatei (A) wählen",
            filetypes=[("Alle Dateien", "*.*"), ("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.webp"),
                       ("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv *.mpeg *.mpg")]
        )
        if path:
            self.path_a.set(path)

    def pick_b(self):
        path = filedialog.askopenfilename(
            title="Zieldatei (B) wählen",
            filetypes=[("Alle Dateien", "*.*"), ("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.webp"),
                       ("Videos", "*.mp4 *.avi *.mkv *.mov *.wmv *.mpeg *.mpg")]
        )
        if path:
            self.path_b.set(path)

    def _resolve_mode(self):
        m = self.mode.get()
        if m == "auto":
            pa, pb = self.path_a.get(), self.path_b.get()
            if is_video(pa) or is_video(pb):
                return "rife"
            return "raft"
        if m.startswith("Bild"):
            return "raft"
        return "rife"

    def _status(self, text):
        self.status.config(text=text)
        self.root.title(f"ImageWarp — {text}")

    # ------------------------------------------------------------------ core
    def _prepare_inputs(self):
        pa, pb = self.path_a.get(), self.path_b.get()
        if not pa or not pb:
            raise ValueError("Bitte beide Dateien wählen.")
        if not os.path.exists(pa) or not os.path.exists(pb):
            raise ValueError("Eine der Dateien existiert nicht.")
        mode = self._resolve_mode()
        if mode == "raft":
            if not (is_image(pa) and is_image(pb)):
                raise ValueError("RAFT-Modus benötigt zwei Bilddateien.")
            img_a = load_image(pa)
            img_b = load_image(pb)
            return img_a, img_b, "raft"
        else:
            # video mode: last frame of A, first frame of B
            if is_image(pa):
                f_a = load_image(pa)
            else:
                f_a = read_video_last_frame(pa)
            if is_image(pb):
                f_b = load_image(pb)
            else:
                f_b = read_video_first_frame(pb)
            return f_a, f_b, "rife"

    def _generate(self):
        mode, f_a, f_b = None, None, None
        pa, pb = self.path_a.get(), self.path_b.get()
        mode = self._resolve_mode()
        if mode == "raft":
            img_a = load_image(pa)
            img_b = load_image(pb)
            # resize inputs to RAFT work size for flow; keep aspect by resize to RAFT_W x RAFT_H
            a = resize_to(img_a, RAFT_W, RAFT_H)
            b = resize_to(img_b, RAFT_W, RAFT_H)
            frames = raft_transition_frames(a, b, self.num_frames.get())
        else:
            if is_image(pa):
                f_a = load_image(pa)
            else:
                f_a = read_video_last_frame(pa)
            if is_image(pb):
                f_b = load_image(pb)
            else:
                f_b = read_video_first_frame(pb)
            # resize to a common work size for RIFE
            f_a = resize_to(f_a, RAFT_W, RAFT_H)
            f_b = resize_to(f_b, RAFT_W, RAFT_H)
            frames = rife_transition_frames(f_a, f_b, self.num_frames.get())
        # resize all frames to output size
        self.frames = [fit_to_output(fr) for fr in frames]
        return mode

    def _run_generate(self, callback, *extra_args):
        self._status("Generiere Frames…")
        try:
            mode = self._generate()
            callback(mode, None, *extra_args)
        except Exception as e:
            callback(None, str(e), *extra_args)

    # ------------------------------------------------------------------ preview
    def preview(self):
        if not self.path_a.get() or not self.path_b.get():
            messagebox.showwarning(self.root.title, "Bitte beide Dateien wählen.")
            return
        self.preview_stop.set()
        # stop any running preview
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=0.5)
        self.preview_stop.clear()
        self.frames = []
        self._status("Generiere Vorschau…")
        self.canvas.delete("all")
        self._run_generate(self._preview_ready)

    def _preview_ready(self, mode, err):
        if err:
            messagebox.showerror("Fehler", err)
            self._status("Fehler: " + err)
            return
        if not self.frames:
            self._status("Keine Frames erzeugt.")
            return
        self._status(f"Vorschau ({mode}) — {len(self.frames)} Frames")
        self._run_preview()

    def _run_preview(self):
        if self.preview_stop.is_set():
            return
        if not self.frames:
            return
        idx = 0
        n = len(self.frames)

        def step():
            nonlocal idx
            if self.preview_stop.is_set():
                return
            fr = self.frames[idx]
            self._show_frame(fr)
            idx += 1
            if idx >= n:
                idx = 0
            self.root.after(40, step)

        self.root.after(40, step)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = None
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb)
        except Exception:
            pil = None
        if pil is not None:
            from PIL import Tkinter as TkPIL
            img = TkPIL.PhotoImage(image=pil)
            self.canvas_img = img  # keep reference
            self.canvas.delete("all")
            self.canvas.create_image(OUT_W // 2, OUT_H // 2, image=img)
        else:
            self.canvas.delete("all")
            self.canvas.create_text(
                OUT_W // 2, OUT_H // 2, text="Preview", fill="white", font=("Arial", 16)
            )

    # ------------------------------------------------------------------ export
    def export(self):
        if not self.path_a.get() or not self.path_b.get():
            messagebox.showwarning(self.root.title, "Bitte beide Dateien wählen.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportieren als Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")]
        )
        if not path:
            return
        self._status("Exportiere…")
        self._run_generate(self._export_ready, path)

    def _export_ready(self, mode, err, path=None):
        if err:
            messagebox.showerror("Fehler", err)
            self._status("Fehler: " + err)
            return
        if not self.frames:
            messagebox.showerror("Fehler", "Keine Frames erzeugt.")
            self._status("Keine Frames.")
            return
        try:
            self._write_video(path, self.frames)
            self._status(f"Exportiert: {os.path.basename(path)} ({len(self.frames)} Frames)")
            messagebox.showinfo(self.root.title, f"Video gespeichert:\n{path}")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self._status("Export-Fehler: " + str(e))

    def _write_video(self, path, frames):
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 30, (OUT_W, OUT_H))
        if not writer.isOpened():
            # fallback: AVI with XVID
            if not path.lower().endswith(".avi"):
                path = path[:-4] + ".avi"
            fourcc = cv2.VideoWriter.fourcc(*"XVID")
            writer = cv2.VideoWriter(path, fourcc, 30, (OUT_W, OUT_H))
        for fr in frames:
            writer.write(fr)
        writer.release()

    # ------------------------------------------------------------------ run
    def run(self):
        self.root.mainloop()


def main():
    root = tk.Tk()
    App(root).run()


if __name__ == "__main__":
    main()


