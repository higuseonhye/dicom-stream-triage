"""Synthetic chest radiographs and the DICOM files that store them.

The picture is two lung ellipses, a mediastinum, ribs, a few vessels, and
one optional finding: a compact nodule, an elongated vessel crossing, or an
apical pneumothorax. Vessel crossings are not nodules. They exist so a
sensitive score can false-alarm on them.

No hospital study and no vendor checkpoint is read. A mammogram would be the
same uncompressed frame with a larger matrix. This generator does not build
that matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

KIND_NORMAL = 0
KIND_NODULE = 1
KIND_VESSEL = 2
KIND_PNEUMOTHORAX = 3
KIND_NAMES = ("normal", "nodule", "vessel", "pneumothorax")

PIXEL_OFFSET = 200.0
PIXEL_SCALE = 3500.0
WINDOW_CENTER = PIXEL_OFFSET + PIXEL_SCALE / 2.0
WINDOW_WIDTH = PIXEL_SCALE


@dataclass
class Study:
    pixels: np.ndarray
    image: np.ndarray
    lung: np.ndarray
    kind: int
    boxes: np.ndarray

    @property
    def nodule(self) -> bool:
        return self.kind == KIND_NODULE

    @property
    def emergency(self) -> bool:
        return self.kind == KIND_PNEUMOTHORAX


def _lungs(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    yy, xx = np.mgrid[0:size, 0:size]
    cy = size * 0.50
    cx = size * 0.50
    left = ((yy - cy) / (size * 0.36)) ** 2 + ((xx - (cx - size * 0.18)) / (size * 0.15)) ** 2 <= 1.0
    right = ((yy - cy) / (size * 0.36)) ** 2 + ((xx - (cx + size * 0.18)) / (size * 0.15)) ** 2 <= 1.0
    return yy, xx, left | right, cy, cx


def _paint_ribs(image: np.ndarray, lung: np.ndarray, yy: np.ndarray, xx: np.ndarray, cx: float) -> None:
    size = image.shape[0]
    for center in (cx - size * 0.18, cx + size * 0.18):
        for index in range(6):
            y0 = size * (0.30 + 0.09 * index)
            curve = y0 + size * 0.012 * np.sin((xx - center) / (size * 0.055))
            rib = (np.abs(yy - curve) < max(1.2, size * 0.007)) & lung & (np.abs(xx - center) < size * 0.14)
            image[rib] = np.maximum(image[rib], 0.40)


def _paint_vessels(image: np.ndarray, lung: np.ndarray, cx: float, rng: np.random.Generator) -> None:
    size = image.shape[0]
    for center in (cx - size * 0.18, cx + size * 0.18):
        for _ in range(3):
            angle = float(rng.uniform(-0.8, 0.8))
            length = int(size * 0.16)
            y = int(rng.integers(int(size * 0.36), int(size * 0.70)))
            x = int(center)
            steps = np.arange(length)
            ys = np.clip((y + steps * np.sin(angle)).astype(np.int64), 0, size - 1)
            xs = np.clip((x + steps * np.cos(angle)).astype(np.int64), 0, size - 1)
            inside = lung[ys, xs]
            image[ys[inside], xs[inside]] = np.maximum(image[ys[inside], xs[inside]], 0.38)


def _lung_point(size: int, rng: np.random.Generator) -> tuple[int, int]:
    """A point that sits inside one ellipse, away from the mediastinum and the border."""
    for _ in range(40):
        side = -1.0 if rng.random() < 0.5 else 1.0
        ny = int(rng.integers(int(size * 0.38), int(size * 0.64)))
        nx = int(size * (0.50 + side * float(rng.uniform(0.10, 0.24))))
        if 24 <= ny < size - 24 and 24 <= nx < size - 24:
            return ny, nx
    return int(size * 0.48), int(size * 0.32)


def render_study(size: int, kind: int, rng: np.random.Generator) -> Study:
    if size < 64:
        raise ValueError("size must be at least 64")
    if kind not in (KIND_NORMAL, KIND_NODULE, KIND_VESSEL, KIND_PNEUMOTHORAX):
        raise ValueError(f"unknown kind {kind}")
    yy, xx, lung, _cy, cx = _lungs(size)
    image = np.full((size, size), 0.06, dtype=np.float64)
    image[lung] = 0.30
    mediastinum = (np.abs(xx - cx) < size * 0.045) & (yy > size * 0.16) & (yy < size * 0.90)
    image[mediastinum] = 0.74
    _paint_ribs(image, lung, yy, xx, cx)
    _paint_vessels(image, lung, cx, rng)
    boxes: list[list[float]] = []
    if kind == KIND_NODULE:
        sigma = size * 0.020
        ny, nx = _lung_point(size, rng)
        blob = np.exp(-((yy - ny) ** 2 + (xx - nx) ** 2) / (2.0 * sigma ** 2))
        image = image + 0.95 * blob
        radius = 2.6 * sigma
        boxes.append([nx - radius, ny - radius, nx + radius, ny + radius])
    elif kind == KIND_VESSEL:
        ny, nx = _lung_point(size, rng)
        horizontal = np.exp(-((yy - ny) ** 2) / (2.0 * 0.95 ** 2)) * np.exp(-((xx - nx) ** 2) / (2.0 * (size * 0.11) ** 2))
        vertical = np.exp(-((xx - nx) ** 2) / (2.0 * 0.95 ** 2)) * np.exp(-((yy - ny) ** 2) / (2.0 * (size * 0.11) ** 2))
        image = image + 0.84 * np.maximum(horizontal, vertical)
    elif kind == KIND_PNEUMOTHORAX:
        side = -1.0 if rng.random() < 0.5 else 1.0
        center = cx + side * size * 0.18
        on_side = lung & (np.abs(xx - center) < size * 0.16)
        line = size * 0.34 + size * 0.02 * np.sin((xx - center) / (size * 0.08))
        air = on_side & (yy < line)
        image[air] = 0.04
        pleural = on_side & (np.abs(yy - line) <= 1.15)
        image[pleural] = 0.93
    image = np.clip(image + rng.normal(0.0, 0.01, size=image.shape), 0.0, 1.0)
    pixels = np.clip(image * PIXEL_SCALE + PIXEL_OFFSET, 0, 4095).astype(np.uint16)
    return Study(
        pixels=pixels,
        image=image.astype(np.float32),
        lung=lung,
        kind=kind,
        boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
    )


def sample_kinds(n: int, rng: np.random.Generator, mix: str) -> np.ndarray:
    if mix == "detection":
        cycle = np.array([KIND_NODULE, KIND_VESSEL, KIND_NORMAL], dtype=np.int64)
        kinds = np.resize(cycle, n).copy()
        rng.shuffle(kinds)
        return kinds
    if mix == "worklist":
        roll = rng.random(n)
        kinds = np.full(n, KIND_NORMAL, dtype=np.int64)
        kinds[roll < 0.10] = KIND_PNEUMOTHORAX
        kinds[(roll >= 0.10) & (roll < 0.22)] = KIND_NODULE
        kinds[(roll >= 0.22) & (roll < 0.30)] = KIND_VESSEL
        return kinds
    if mix == "balanced":
        cycle = np.array([KIND_NORMAL, KIND_NODULE, KIND_VESSEL, KIND_PNEUMOTHORAX], dtype=np.int64)
        kinds = np.resize(cycle, n).copy()
        rng.shuffle(kinds)
        return kinds
    raise ValueError("mix must be detection, worklist, or balanced")


def write_dicom(path: Path, pixels: np.ndarray) -> None:
    """Write one uncompressed secondary-capture DICOM. Explicit VR little endian."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "CR"
    dataset.PatientName = "SYNTHETIC"
    dataset.PatientID = "SYNTH"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows = int(pixels.shape[0])
    dataset.Columns = int(pixels.shape[1])
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.WindowCenter = WINDOW_CENTER
    dataset.WindowWidth = WINDOW_WIDTH
    dataset.PixelData = np.ascontiguousarray(pixels, dtype="<u2").tobytes()
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.save_as(path, enforce_file_format=True)


def read_pixels_pydicom(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    if pixels.dtype != np.uint16:
        pixels = pixels.astype(np.uint16)
    return pixels
