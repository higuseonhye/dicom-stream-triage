"""Spatial-entropy box filter and a risk-uncertainty worklist.

Spatial entropy is the Shannon entropy, in nats, of the residual mass inside
a candidate window after that window is pooled onto a 6 by 6 grid. A compact
nodule lands in one or two cells. A vessel crossing spreads along a row and
a column, so its entropy is higher.

The worklist uses a different entropy: the entropy of a fused 3-class
distribution. Pneumothorax mass is the study head. Nodule mass is the
residual-peak score, shifted so a normal peak contributes nothing. Low
entropy and low lesion probability auto-pass. Everything else is ordered by
pneumothorax probability, nodule probability, and that entropy. The queue
never sees the generator label.
"""

from __future__ import annotations

import numpy as np

from .generator import KIND_NAMES, KIND_NODULE, KIND_NORMAL, KIND_PNEUMOTHORAX

IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.5
NODULE_PASS = 0.35
PTX_PASS = 0.25
PRED_ENTROPY_PASS = 0.50
PTX_WEIGHT = 5.0
NODULE_WEIGHT = 2.0
ENTROPY_WEIGHT = 1.2
READ_SECONDS = 45.0


def spatial_entropy(mass: np.ndarray) -> float:
    values = np.clip(mass, 0.0, None).astype(np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probability = (values / total).ravel()
    probability = probability[probability > 0.0]
    return float(-(probability * np.log(probability)).sum())


def core_entropy(patch: np.ndarray, cells: int = 6) -> float:
    """Entropy of residual mass after pooling the window into a cells-by-cells grid.

    A compact nodule fills one or two cells. A vessel crossing fills a row and
    a column, so the same mass is spread and the entropy is higher. Per-pixel
    entropy does not show that: both shapes occupy a similar number of bright
    pixels.
    """
    if patch.ndim != 2 or patch.size == 0:
        return 0.0
    height, width = patch.shape
    height = height - (height % cells)
    width = width - (width % cells)
    if height < cells or width < cells:
        return spatial_entropy(patch)
    view = np.clip(patch[:height, :width], 0.0, None)
    block_y = height // cells
    block_x = width // cells
    pooled = view.reshape(cells, block_y, cells, block_x).sum(axis=(1, 3))
    return spatial_entropy(pooled)


def softmax_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    clipped = clipped / clipped.sum(axis=-1, keepdims=True)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.zeros(0, dtype=np.float64)
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2])
    y1 = np.minimum(box[3], others[:, 3])
    intersection = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    area_box = max(float(box[2] - box[0]) * float(box[3] - box[1]), 0.0)
    area_other = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(others[:, 3] - others[:, 1], 0.0, None)
    union = np.clip(area_box + area_other - intersection, 1e-8, None)
    return intersection / union


def match_image(boxes: np.ndarray, scores: np.ndarray, ground_truth: np.ndarray, iou_threshold: float = IOU_THRESHOLD) -> tuple[int, int, int]:
    """Greedy score order. Returns true positives, false positives, false negatives."""
    if ground_truth.size == 0:
        return 0, int(boxes.shape[0]), 0
    if boxes.size == 0:
        return 0, 0, int(ground_truth.shape[0])
    order = np.argsort(scores)[::-1]
    used = np.zeros(ground_truth.shape[0], dtype=bool)
    true_positive = 0
    false_positive = 0
    for index in order:
        overlaps = box_iou(boxes[index], ground_truth)
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not used[best]:
            used[best] = True
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive, int((~used).sum())


def _rates(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int | None]:
    precision = None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = None if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
    }


def collect_candidates(studies, limit: int = 8) -> list[dict]:
    from .detector import propose

    rows = []
    for image_index, study in enumerate(studies):
        for candidate in propose(study.image, study.lung, core_entropy, limit=limit):
            rows.append(
                {
                    "image": image_index,
                    "score": candidate.score,
                    "entropy": candidate.entropy,
                    "box": candidate.box,
                    "y": candidate.y,
                    "x": candidate.x,
                }
            )
    return rows


def score_policy(candidates: list[dict], studies, score_threshold: float, entropy_threshold: float | None) -> dict:
    per_image: dict[int, list[dict]] = {index: [] for index in range(len(studies))}
    for row in candidates:
        keep = row["score"] >= score_threshold
        if entropy_threshold is not None:
            keep = keep and row["entropy"] <= entropy_threshold
        if keep:
            per_image[row["image"]].append(row)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for index, study in enumerate(studies):
        chosen = per_image[index]
        if chosen:
            boxes = np.stack([row["box"] for row in chosen])
            scores = np.asarray([row["score"] for row in chosen], dtype=np.float64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float64)
            scores = np.zeros(0, dtype=np.float64)
        tp, fp, fn = match_image(boxes, scores, study.boxes)
        true_positive += tp
        false_positive += fp
        false_negative += fn
    return _rates(true_positive, false_positive, false_negative)


def evaluate_saved(
    cand_image: np.ndarray,
    cand_score: np.ndarray,
    cand_entropy: np.ndarray,
    cand_box: np.ndarray,
    gt_image: np.ndarray,
    gt_box: np.ndarray,
    n_images: int,
    score_threshold: float,
    entropy_threshold: float | None,
) -> dict:
    """Same counts as score_policy, from the arrays stored for the board."""
    studies = []
    for index in range(n_images):
        chosen = gt_box[gt_image == index] if gt_image.size else np.zeros((0, 4))
        studies.append(type("Boxes", (), {"boxes": chosen.reshape(-1, 4)})())
    candidates = [
        {"image": int(cand_image[i]), "score": float(cand_score[i]), "entropy": float(cand_entropy[i]), "box": cand_box[i]}
        for i in range(cand_image.shape[0])
    ]
    return score_policy(candidates, studies, score_threshold, entropy_threshold)


def reduction(baseline_fp: int, filtered_fp: int) -> float | None:
    if baseline_fp <= 0:
        return None
    return float((baseline_fp - filtered_fp) / baseline_fp)


def select_entropy_threshold(studies, candidates: list[dict], score_threshold: float = SCORE_THRESHOLD) -> dict:
    """Pick the highest-FP-reduction cut that does not give precision back."""
    entropies = np.asarray([row["entropy"] for row in candidates], dtype=np.float64)
    if entropies.size == 0:
        raise RuntimeError("no candidates to tune")
    grid = np.unique(np.quantile(entropies, np.linspace(0.05, 0.95, 19)))
    baseline = score_policy(candidates, studies, score_threshold, None)
    best = None
    for tau in grid:
        filtered = score_policy(candidates, studies, score_threshold, float(tau))
        fp_drop = reduction(int(baseline["fp"]), int(filtered["fp"]))
        precision_ok = (
            baseline["precision"] is not None
            and filtered["precision"] is not None
            and filtered["precision"] + 1e-9 >= baseline["precision"]
        )
        recall_ok = (
            baseline["recall"] is None
            or filtered["recall"] is None
            or filtered["recall"] + 1e-9 >= 0.85 * baseline["recall"]
        )
        record = {
            "tau": float(tau),
            "fp_reduction": fp_drop,
            "precision_ok": bool(precision_ok),
            "recall_ok": bool(recall_ok),
            "filtered": filtered,
        }
        eligible = precision_ok and recall_ok and fp_drop is not None
        rank = (
            1 if eligible else 0,
            -1.0 if fp_drop is None else fp_drop,
            -1.0 if filtered["precision"] is None else filtered["precision"],
        )
        if best is None or rank > best[0]:
            best = (rank, record)
    assert best is not None
    return {"baseline": baseline, "chosen": best[1], "grid_size": int(grid.size)}


def nodule_evidence(scores: np.ndarray) -> np.ndarray:
    """Map a residual-peak score onto nodule evidence in [0, 1].

    Normal studies peak near 0.39 and nodule or vessel crossings near 0.61
    with the current calibration (scale 0.35). Scores at or below 0.46 add no
    nodule evidence. A peak of 0.71 saturates.
    """
    return np.clip((np.asarray(scores, dtype=np.float64) - 0.46) / 0.25, 0.0, 1.0)


def fuse_study(class_probabilities: np.ndarray, proposal_scores: np.ndarray) -> np.ndarray:
    """Pneumothorax mass comes from the study head. Nodule mass comes from the proposal score.

    The head separates an apical pneumothorax. It does not separate a 5 px nodule
    from rib texture, so that evidence stays with the residual peak that the box
    filter already uses. The remainder is the normal mass. Rows are renormalized.
    """
    if class_probabilities.shape[0] != np.asarray(proposal_scores).shape[0]:
        raise ValueError("class probabilities and proposal scores must share the batch")
    ptx = class_probabilities[:, 2]
    nodule = nodule_evidence(proposal_scores)
    normal = np.clip(1.0 - ptx - nodule, 0.0, None)
    stacked = np.stack((normal, nodule, ptx), axis=1)
    return stacked / np.clip(stacked.sum(axis=1, keepdims=True), 1e-6, None)


def queue_flags(
    probabilities: np.ndarray,
    nodule_cut: float = NODULE_PASS,
    ptx_cut: float = PTX_PASS,
    entropy_cut: float = PRED_ENTROPY_PASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entropy = softmax_entropy(probabilities)
    nodule = probabilities[:, 1]
    ptx = probabilities[:, 2]
    auto_pass = (ptx < ptx_cut) & (nodule < nodule_cut) & (entropy < entropy_cut)
    priority = PTX_WEIGHT * ptx + NODULE_WEIGHT * nodule + ENTROPY_WEIGHT * entropy
    return auto_pass, priority, entropy


def _mean_wait(positions: np.ndarray) -> float | None:
    if positions.size == 0:
        return None
    return float(positions.mean() * READ_SECONDS)


def evaluate_worklist(
    probabilities: np.ndarray,
    kinds: np.ndarray,
    nodule_cut: float = NODULE_PASS,
    ptx_cut: float = PTX_PASS,
    entropy_cut: float = PRED_ENTROPY_PASS,
) -> dict:
    if probabilities.shape[0] != kinds.shape[0]:
        raise ValueError("probabilities and kinds must share the batch")
    auto_pass, priority, entropy = queue_flags(probabilities, nodule_cut, ptx_cut, entropy_cut)
    arrival = np.arange(kinds.shape[0])
    emergency = kinds == KIND_PNEUMOTHORAX
    normal = kinds == KIND_NORMAL
    nodule = kinds == KIND_NODULE
    active = np.flatnonzero(~auto_pass)
    priority_order = active[np.lexsort((arrival[active], -priority[active]))]
    fifo_rank = arrival
    priority_rank = np.full(kinds.shape[0], -1, dtype=np.int64)
    priority_rank[priority_order] = np.arange(priority_order.size)

    missed = emergency & auto_pass
    queued_emergency = emergency & ~auto_pass
    fifo_wait = _mean_wait(fifo_rank[emergency].astype(np.float64))
    priority_wait = _mean_wait(priority_rank[queued_emergency].astype(np.float64))
    if fifo_wait is None or priority_wait is None or fifo_wait <= 0:
        wait_reduction = None
    else:
        wait_reduction = float(1.0 - priority_wait / fifo_wait)
    by_kind = []
    for kind, name in enumerate(KIND_NAMES):
        chosen = kinds == kind
        if int(chosen.sum()) == 0:
            continue
        by_kind.append(
            {
                "kind": name,
                "studies": int(chosen.sum()),
                "mean_nodule": float(probabilities[chosen, 1].mean()),
                "mean_ptx": float(probabilities[chosen, 2].mean()),
                "mean_entropy": float(entropy[chosen].mean()),
                "auto_pass_rate": float(auto_pass[chosen].mean()),
                "mean_priority": float(priority[chosen].mean()),
            }
        )
    return {
        "studies": int(kinds.size),
        "read_seconds": READ_SECONDS,
        "auto_pass_count": int(auto_pass.sum()),
        "normal_auto_pass_rate": float(auto_pass[normal].mean()) if int(normal.sum()) else None,
        "nodule_review_rate": float((~auto_pass)[nodule].mean()) if int(nodule.sum()) else None,
        "emergencies": int(emergency.sum()),
        "emergency_misses": int(missed.sum()),
        "emergency_miss_rate": float(missed.sum() / max(int(emergency.sum()), 1)),
        "fifo_emergency_wait_s": fifo_wait,
        "priority_emergency_wait_s": priority_wait,
        "wait_reduction": wait_reduction,
        "cuts": {"nodule": nodule_cut, "ptx": ptx_cut, "entropy": entropy_cut},
        "by_kind": by_kind,
        "auto_pass": auto_pass,
        "priority": priority,
        "entropy": entropy,
        "priority_order": priority_order,
        "nodule_kind_code": KIND_NODULE,
    }


def self_check() -> dict[str, float]:
    """A compact gaussian core must sit below an elongated cross."""
    yy, xx = np.mgrid[0:36, 0:36]
    nodule = np.exp(-((yy - 18) ** 2 + (xx - 18) ** 2) / (2.0 * 2.0 ** 2))
    horizontal = np.exp(-((yy - 18) ** 2) / (2.0 * 0.9 ** 2))
    vertical = np.exp(-((xx - 18) ** 2) / (2.0 * 0.9 ** 2))
    cross = np.maximum(horizontal, vertical)
    nodule_entropy = core_entropy(nodule)
    cross_entropy = core_entropy(cross)
    if not nodule_entropy + 0.15 < cross_entropy:
        raise RuntimeError(f"spatial entropy did not separate nodule {nodule_entropy:.3f} from cross {cross_entropy:.3f}")
    if nodule_evidence(np.asarray([0.39]))[0] != 0.0 or nodule_evidence(np.asarray([0.61]))[0] < 0.35:
        raise RuntimeError("nodule evidence no longer separates a normal peak from a nodule peak")
    return {"nodule_nats": nodule_entropy, "cross_nats": cross_entropy}
