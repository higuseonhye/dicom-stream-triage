"""Sensitive blob proposals and a small ConvNeXt chest head.

Proposals are local peaks of a wide-blur residual inside the lung. The score
is a fixed calibration of that peak, so a vessel crossing can clear 0.5.
The network is a separate head: two heatmap logits (nodule, pneumothorax)
and a 3-class study logit. It does not decide the box filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from torch import nn

from .generator import (
    KIND_NODULE,
    KIND_PNEUMOTHORAX,
    Study,
    render_study,
    sample_kinds,
)

INPUT_SIZE = 256
OUTPUT_STRIDE = 4
RESIDUAL_SIGMA = 6.0
SCORE_SCALE = 0.35
PEAK_FLOOR = 0.05


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.pointwise(self.norm(self.depthwise(hidden)))


class PatchDetector(nn.Module):
    """Stride-4 heatmaps plus a study classifier. Width 16 then 32."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.stage1 = ConvNeXtBlock(16)
        self.down = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.stage2 = ConvNeXtBlock(32)
        self.heatmap = nn.Conv2d(32, 2, kernel_size=1)
        self.classifier = nn.Linear(32 + 2, 3)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.stage2(self.down(self.stage1(self.stem(image))))
        heat = self.heatmap(hidden)
        pooled = hidden.amax(dim=(2, 3))
        peak = heat.amax(dim=(2, 3))
        logits = self.classifier(torch.cat((pooled, peak), dim=1))
        return heat, logits


@dataclass
class Candidate:
    y: int
    x: int
    score: float
    entropy: float
    box: np.ndarray


def residual_map(image: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(image.astype(np.float64), sigma=RESIDUAL_SIGMA)
    return np.clip(image.astype(np.float64) - blurred, 0.0, None)


def peak_confidence(peak: float, scale: float = SCORE_SCALE) -> float:
    if scale <= 0:
        raise ValueError("score scale must be positive")
    return float(peak / (peak + scale))


def propose(
    image: np.ndarray,
    lung: np.ndarray,
    entropy_fn,
    limit: int = 8,
    half: int = 18,
    box_half: int = 12,
) -> list[Candidate]:
    """Lung-restricted residual peaks. entropy_fn scores one residual patch."""
    residual = residual_map(image)
    eligible = lung.copy()
    eligible[:half, :] = False
    eligible[-half:, :] = False
    eligible[:, :half] = False
    eligible[:, -half:] = False
    residual = np.where(eligible, residual, 0.0)
    neighborhood = maximum_filter(residual, size=15)
    peaks = (residual == neighborhood) & (residual >= PEAK_FLOOR)
    ys, xs = np.nonzero(peaks)
    if ys.size == 0:
        return []
    order = np.argsort(residual[ys, xs])[::-1]
    chosen: list[Candidate] = []
    for index in order:
        y = int(ys[index])
        x = int(xs[index])
        if any((y - item.y) ** 2 + (x - item.x) ** 2 < 14 ** 2 for item in chosen):
            continue
        patch = residual[y - half : y + half, x - half : x + half]
        chosen.append(
            Candidate(
                y=y,
                x=x,
                score=peak_confidence(float(residual[y, x])),
                entropy=float(entropy_fn(patch)),
                box=np.asarray([x - box_half, y - box_half, x + box_half, y + box_half], dtype=np.float64),
            )
        )
        if len(chosen) >= limit:
            break
    return chosen


def _heatmap_target(size: int, boxes: np.ndarray, stride: int = OUTPUT_STRIDE) -> np.ndarray:
    grid = size // stride
    target = np.zeros((grid, grid), dtype=np.float32)
    if boxes.size == 0:
        return target
    yy, xx = np.mgrid[0:grid, 0:grid]
    for box in boxes:
        cy = 0.5 * (box[1] + box[3]) / stride
        cx = 0.5 * (box[0] + box[2]) / stride
        target = np.maximum(target, np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 0.85 ** 2)))
    return target.astype(np.float32)


def _ptx_target(image: np.ndarray, kind: int, stride: int = OUTPUT_STRIDE) -> np.ndarray:
    grid = image.shape[0] // stride
    if kind != KIND_PNEUMOTHORAX:
        return np.zeros((grid, grid), dtype=np.float32)
    # The pleural line is the brightest thin structure; splat its grid cells.
    line = image >= 0.85
    pooled = line.reshape(grid, stride, grid, stride).any(axis=(1, 3))
    yy, xx = np.mgrid[0:grid, 0:grid]
    ys, xs = np.nonzero(pooled)
    target = np.zeros((grid, grid), dtype=np.float32)
    for y, x in zip(ys, xs):
        target = np.maximum(target, np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * 0.7 ** 2)))
    return target


def _class_index(kind: int) -> int:
    if kind == KIND_NODULE:
        return 1
    if kind == KIND_PNEUMOTHORAX:
        return 2
    return 0


@dataclass
class CohortTensors:
    images: np.ndarray
    nodule_heat: np.ndarray
    ptx_heat: np.ndarray
    labels: np.ndarray
    kinds: np.ndarray


def build_cohort(n: int, mix: str, seed: int) -> tuple[CohortTensors, list[Study]]:
    rng = np.random.default_rng(seed)
    kinds = sample_kinds(n, rng, mix)
    studies = [render_study(INPUT_SIZE, int(kind), rng) for kind in kinds]
    images = np.empty((n, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    nodule_heat = np.empty((n, INPUT_SIZE // OUTPUT_STRIDE, INPUT_SIZE // OUTPUT_STRIDE), dtype=np.float32)
    ptx_heat = np.empty_like(nodule_heat)
    labels = np.empty(n, dtype=np.int64)
    for index, study in enumerate(studies):
        images[index, 0] = study.image
        nodule_heat[index] = _heatmap_target(INPUT_SIZE, study.boxes)
        ptx_heat[index] = _ptx_target(study.image, study.kind)
        labels[index] = _class_index(study.kind)
    packed = CohortTensors(
        images=images,
        nodule_heat=nodule_heat,
        ptx_heat=ptx_heat,
        labels=labels,
        kinds=np.asarray(kinds),
    )
    return packed, studies


@dataclass
class FitResult:
    model: PatchDetector
    train_loss: float


def fit_detector(n_studies: int = 160, epochs: int = 16, batch_size: int = 32, seed: int = 0) -> FitResult:
    cohort, _studies = build_cohort(n_studies, "balanced", seed)
    torch.manual_seed(seed)
    model = PatchDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    images = torch.from_numpy(np.ascontiguousarray(cohort.images))
    nodule = torch.from_numpy(np.ascontiguousarray(cohort.nodule_heat))
    ptx = torch.from_numpy(np.ascontiguousarray(cohort.ptx_heat))
    labels = torch.from_numpy(np.ascontiguousarray(cohort.labels))
    positive = torch.tensor(12.0)
    shuffle = np.random.default_rng(seed + 7)
    model.train()
    train_loss = 0.0
    for _epoch in range(epochs):
        order = shuffle.permutation(n_studies)
        total = 0.0
        seen = 0
        for start in range(0, n_studies, batch_size):
            batch = torch.from_numpy(order[start : start + batch_size].astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            heat, logits = model(images[batch])
            loss_nodule = nn.functional.binary_cross_entropy_with_logits(heat[:, 0], nodule[batch], pos_weight=positive)
            loss_ptx = nn.functional.binary_cross_entropy_with_logits(heat[:, 1], ptx[batch], pos_weight=positive)
            loss_class = nn.functional.cross_entropy(logits, labels[batch])
            loss = loss_nodule + loss_ptx + loss_class
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(batch.shape[0])
            seen += int(batch.shape[0])
        train_loss = total / max(seen, 1)
    model.eval()
    return FitResult(model=model, train_loss=train_loss)


def predict(model: PatchDetector, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """images is (N, H, W). Returns softmax (N, 3) and nodule heatmap (N, h, w)."""
    tensor = torch.from_numpy(np.ascontiguousarray(images[:, None, :, :], dtype=np.float32))
    model.eval()
    chunks_prob = []
    chunks_heat = []
    with torch.inference_mode():
        for start in range(0, tensor.shape[0], 32):
            heat, logits = model(tensor[start : start + 32])
            chunks_prob.append(torch.softmax(logits, dim=-1).numpy())
            chunks_heat.append(torch.sigmoid(heat[:, 0]).numpy())
    return np.concatenate(chunks_prob, axis=0), np.concatenate(chunks_heat, axis=0)
