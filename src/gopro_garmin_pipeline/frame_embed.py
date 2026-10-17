"""Image embeddings of candidate frames, for "do these two clips look alike".

The composer's redundancy term needs a visual similarity between two
candidates. The vision model's rubric scores (light, composition, ...) can't
supply one: they say how *good* a frame is, not what is in it, so a cliff
descent and a bridge walkway that both score "8 scenery" read as twins.
DINOv2-small embeddings of the actual frames separate look-alikes from
different shots far better (AUC 0.98 vs 0.68 on 96 hand-judged pairs from
the 2026-10-10 ride).

Each candidate is embedded from three frames (anchor -1.5s, anchor,
anchor +1.5s) read from the ``.LRV`` proxy when present. A pair's
similarity is the best cosine over the 3x3 frame pairs. Raw cosine runs
uniformly high on POV footage (bars, road and horizon in every frame), so
the composer rescales it against the ride's own pairwise spread.

The ONNX model (88 MB, too big to vendor) downloads once to
``~/.cache/gopro-garmin/models`` from a pinned Hugging Face revision and
is checked against its SHA-256. Per-candidate embeddings cache under
``<date>/.embed_cache/``. Any failure leaves candidates unembedded and the
composer falls back to the rubric.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .utils import gopro_lrv_proxy

_MODEL_REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
_MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small/resolve/"
    f"{_MODEL_REVISION}/onnx/model.onnx"
)
_MODEL_SHA256 = "f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
_MODEL_DIR = Path.home() / ".cache" / "gopro-garmin" / "models"
_MODEL_PATH = _MODEL_DIR / f"dinov2-small-{_MODEL_REVISION[:8]}.onnx"

_CACHE_DIR_NAME = ".embed_cache"
_CACHE_VERSION = "v1"
_FRAME_OFFSETS = (-1.5, 0.0, 1.5)
_DECODE_HEIGHT = 480          # LRV native; full-res clips are scaled to match
_RESIZE_SHORT = 256           # DINOv2 preprocessing: resize, then centre crop
_CROP = 224
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_BATCH = 16

_session = None


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("  Frame embeddings: downloading DINOv2-small (88 MB, one-time)...")
    with tempfile.NamedTemporaryFile(dir=_MODEL_DIR, delete=False) as tmp:
        with urllib.request.urlopen(_MODEL_URL, timeout=120) as resp:
            h = hashlib.sha256()
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                tmp.write(chunk)
    if h.hexdigest() != _MODEL_SHA256:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError("DINOv2 model download failed its SHA-256 check")
    Path(tmp.name).replace(_MODEL_PATH)
    return _MODEL_PATH


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        _session = ort.InferenceSession(
            str(_ensure_model()), providers=["CPUExecutionProvider"])
    return _session


def _frame_source(video_dir: Path, clip_name: str) -> Path | None:
    """The GoPro .LRV proxy if present (fast decode), else the clip itself."""
    clip = video_dir / clip_name
    return gopro_lrv_proxy(clip) or (clip if clip.exists() else None)


def _read_frame(source: Path, t: float) -> np.ndarray | None:
    """One RGB frame at ``t`` seconds, decoded at _DECODE_HEIGHT."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(source),
           "-frames:v", "1", "-vf", f"scale=-2:{_DECODE_HEIGHT}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    if not raw or len(raw) % (_DECODE_HEIGHT * 3):
        return None
    width = len(raw) // (_DECODE_HEIGHT * 3)
    return np.frombuffer(raw, np.uint8).reshape(_DECODE_HEIGHT, width, 3)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(frame)
    w, h = im.size
    s = _RESIZE_SHORT / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = im.size
    left, top = (w - _CROP) // 2, (h - _CROP) // 2
    im = im.crop((left, top, left + _CROP, top + _CROP))
    x = (np.asarray(im, np.float32) / 255.0 - _MEAN) / _STD
    return x.transpose(2, 0, 1)


def _embed(pixels: np.ndarray) -> np.ndarray:
    """L2-normalised CLS embeddings for a (N, 3, 224, 224) batch."""
    sess = _get_session()
    out = [sess.run(None, {"pixel_values": pixels[i:i + _BATCH]})[0][:, 0]
           for i in range(0, len(pixels), _BATCH)]
    e = np.concatenate(out).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def _cache_path(video_dir: Path, clip_name: str, anchor: float) -> Path:
    key = f"{Path(clip_name).stem}@{anchor:.1f}"
    return video_dir / _CACHE_DIR_NAME / _CACHE_VERSION / f"{key}.npy"


def embed_candidates(video_dir: Path, segments: list) -> dict[int, np.ndarray]:
    """Map ``id(seg)`` → (k, 384) frame embeddings, k ≤ 3.

    Candidates whose frames can't be read are left out of the map.
    """
    video_dir = Path(video_dir)
    result: dict[int, np.ndarray] = {}
    todo = []
    for seg in segments:
        path = _cache_path(video_dir, seg.clip_name, seg.anchor_video_secs)
        if path.exists():
            try:
                result[id(seg)] = np.load(path)
                continue
            except (OSError, ValueError):
                pass
        src = _frame_source(video_dir, seg.clip_name)
        if src is not None:
            todo.append((seg, src, path))
    if not todo:
        return result

    jobs = [(i, src, seg.anchor_video_secs + off)
            for i, (seg, src, _) in enumerate(todo) for off in _FRAME_OFFSETS]
    with ThreadPoolExecutor(8) as pool:
        frames = list(pool.map(lambda j: _read_frame(j[1], j[2]), jobs))

    owners, pixels = [], []
    for (i, _, _), frame in zip(jobs, frames):
        if frame is not None:
            owners.append(i)
            pixels.append(_preprocess(frame))
    if not pixels:
        return result
    emb = _embed(np.stack(pixels))

    for i, (seg, _, path) in enumerate(todo):
        rows = emb[[k for k, o in enumerate(owners) if o == i]]
        if not len(rows):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, rows)
        result[id(seg)] = rows
    return result
