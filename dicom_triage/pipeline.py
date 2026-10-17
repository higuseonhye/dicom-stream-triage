"""Turn an uncompressed explicit-VR DICOM into a windowed tensor.

The baseline is pydicom: parse the dataset, materialize pixel_array, window.
The fast path finds the pixel-data tag, views the uint16 frame with
np.frombuffer (no integer copy), windows in NumPy, and does that over a
thread pool. The float tensor is still allocated. Zero-copy names the
integer frame only.

JPEG 2000 is not decoded. pydicom would need an OpenJPEG plugin for that
transfer syntax, and this process does not load one. RLE is not the measured
path either. Both are named in the report as the hospital formats this
bench does not time.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .generator import WINDOW_CENTER, WINDOW_WIDTH, read_pixels_pydicom, render_study, sample_kinds, write_dicom

PIXEL_TAG = b"\xe0\x7f\x10\x00"
ROWS_TAG = b"\x28\x00\x10\x00"
COLS_TAG = b"\x28\x00\x11\x00"
LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN", b"UR", b"UT", b"UV"}


def window_numpy(pixels: np.ndarray, center: float = WINDOW_CENTER, width: float = WINDOW_WIDTH) -> np.ndarray:
    if width <= 0:
        raise ValueError("window width must be positive")
    lower = center - width / 2.0
    scaled = (pixels.astype(np.float32) - np.float32(lower)) / np.float32(width)
    return np.clip(scaled, 0.0, 1.0)


def _tag_value(blob: bytes, tag: bytes) -> tuple[int, int, bytes]:
    found = blob.find(tag)
    if found < 0 or found + 8 > len(blob):
        raise ValueError("DICOM tag missing")
    vr = blob[found + 4 : found + 6]
    if vr in LONG_VRS:
        length = int.from_bytes(blob[found + 8 : found + 12], "little")
        start = found + 12
    else:
        length = int.from_bytes(blob[found + 6 : found + 8], "little")
        start = found + 8
    if length < 0 or start + length > len(blob):
        raise ValueError("DICOM value runs past the file")
    return start, length, vr


def zerocopy_pixels(blob: bytes) -> np.ndarray:
    """Uint16 view of uncompressed pixel data. The returned array aliases blob."""
    if blob[128:132] != b"DICM":
        raise ValueError("missing DICM prefix")
    row_at, row_len, _ = _tag_value(blob, ROWS_TAG)
    col_at, col_len, _ = _tag_value(blob, COLS_TAG)
    if row_len < 2 or col_len < 2:
        raise ValueError("matrix size is not a short")
    rows = int.from_bytes(blob[row_at : row_at + 2], "little")
    cols = int.from_bytes(blob[col_at : col_at + 2], "little")
    start, length, vr = _tag_value(blob, PIXEL_TAG)
    if vr not in (b"OW", b"OB"):
        raise ValueError(f"pixel VR {vr!r} is not an uncompressed frame")
    need = rows * cols * 2
    if length < need:
        raise ValueError("pixel data is shorter than rows*cols*2")
    if start % 2 != 0:
        raise ValueError("pixel data is not aligned for uint16")
    view = np.frombuffer(blob, dtype="<u2", count=rows * cols, offset=start)
    return view.reshape(rows, cols)


def _decode_pydicom(path: Path) -> np.ndarray:
    return window_numpy(read_pixels_pydicom(path))


def _decode_zerocopy(path: Path) -> np.ndarray:
    return window_numpy(zerocopy_pixels(path.read_bytes()))


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }


def _time_each(paths: list[Path], decode) -> tuple[list[float], float]:
    samples = []
    started = time.perf_counter()
    for path in paths:
        tick = time.perf_counter()
        decode(path)
        samples.append((time.perf_counter() - tick) * 1000.0)
    return samples, time.perf_counter() - started


def benchmark_pipeline(
    n_studies: int = 8,
    size: int = 2048,
    workers: int | None = None,
    seed: int = 0,
) -> dict:
    if n_studies < 4:
        raise ValueError("n_studies must be at least 4")
    if workers is None:
        workers = max(2, min(8, os.cpu_count() or 2))
    rng = np.random.default_rng(seed)
    kinds = sample_kinds(n_studies, rng, "balanced")
    root = Path(tempfile.mkdtemp(prefix="dicom-triage-"))
    try:
        paths = []
        for index, kind in enumerate(kinds):
            study = render_study(size, int(kind), rng)
            path = root / f"study_{index:03d}.dcm"
            write_dicom(path, study.pixels)
            paths.append(path)
        reference = _decode_pydicom(paths[0])
        viewed = _decode_zerocopy(paths[0])
        max_abs = float(np.max(np.abs(reference - viewed)))
        if max_abs > 1e-5:
            raise RuntimeError(f"zero-copy window disagrees with pydicom by {max_abs}")
        # Touch every file once so the timed section is not the first disk fault.
        for path in paths:
            path.read_bytes()
        pydicom_ms, pydicom_wall = _time_each(paths, _decode_pydicom)
        copy_ms, copy_wall = _time_each(paths, _decode_zerocopy)
        tick = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_decode_zerocopy, paths))
        thread_wall = time.perf_counter() - tick
    finally:
        shutil.rmtree(root, ignore_errors=True)
    pydicom_stats = _percentiles(pydicom_ms)
    copy_stats = _percentiles(copy_ms)
    return {
        "n_studies": n_studies,
        "frame": size,
        "workers": int(workers),
        "transfer_syntax": "1.2.840.10008.1.2.1",
        "max_abs_diff": max_abs,
        "note": (
            "Uncompressed explicit VR. Both paths read the file, decode the frame, and window it. "
            "np.frombuffer aliases the uint16 bytes. The float window is a new allocation. "
            "JPEG 2000 and RLE are not in this timing."
        ),
        "pydicom_sequential": {
            **pydicom_stats,
            "fps": float(n_studies / pydicom_wall),
            "wall_s": float(pydicom_wall),
        },
        "zerocopy_sequential": {
            **copy_stats,
            "fps": float(n_studies / copy_wall),
            "wall_s": float(copy_wall),
        },
        "zerocopy_threaded": {
            "mean_ms": float(thread_wall * 1000.0 / n_studies),
            "fps": float(n_studies / thread_wall),
            "wall_s": float(thread_wall),
        },
    }
