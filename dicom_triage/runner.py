"""Run the three synthetic DICOM benchmarks and write the report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .detector import INPUT_SIZE, build_cohort, fit_detector, predict, propose
from .generator import render_study
from .pipeline import benchmark_pipeline
from .worklist_engine import (
    IOU_THRESHOLD,
    SCORE_THRESHOLD,
    collect_candidates,
    core_entropy,
    evaluate_worklist,
    fuse_study,
    reduction,
    score_policy,
    select_entropy_threshold,
    self_check,
)

ROOT = Path(__file__).resolve().parents[1]
FP_TARGET = 0.40
WAIT_TARGET = 0.80


def collect_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def render_markdown(report: dict) -> str:
    pipe = report["pipeline"]
    boxes = report["boxes"]
    work = report["worklist"]
    targets = report["targets"]
    base = boxes["eval_baseline"]
    kept = boxes["eval_filtered"]
    kind_rows = [
        [
            row["kind"],
            str(row["studies"]),
            _num(row["mean_nodule"]),
            _num(row["mean_ptx"]),
            _num(row["mean_entropy"]),
            _pct(row["auto_pass_rate"]),
        ]
        for row in work["by_kind"]
    ]
    lines = [
        "# DICOM stream triage — benchmark results",
        "",
        "One CPU run on synthetic chest studies. No hospital archive, no vendor weights, and no JPEG 2000 or RLE file is read.",
        "",
        f"- Seed {report['seed']}. Pipeline frame {pipe['frame']} px, {pipe['n_studies']} studies, {pipe['workers']} threads.",
        f"- Box counts use {boxes['eval_studies']} held-out studies, separate from the {boxes['tune_studies']} used to choose τ. IoU threshold {boxes['iou_threshold']}.",
        f"- Worklist of {work['studies']} studies. One read is a simulated slot of {work['read_seconds']:.0f} seconds.",
        "",
        "## 1. DICOM decode",
        "",
        pipe["note"],
        "",
        _md_table(
            ["Path", "Mean ms/study", "P50 ms", "FPS"],
            [
                [
                    "pydicom sequential",
                    _num(pipe["pydicom_sequential"]["mean_ms"], 2),
                    _num(pipe["pydicom_sequential"]["p50_ms"], 2),
                    _num(pipe["pydicom_sequential"]["fps"], 2),
                ],
                [
                    "zero-copy sequential",
                    _num(pipe["zerocopy_sequential"]["mean_ms"], 2),
                    _num(pipe["zerocopy_sequential"]["p50_ms"], 2),
                    _num(pipe["zerocopy_sequential"]["fps"], 2),
                ],
                [
                    f"zero-copy threads {pipe['workers']}",
                    _num(pipe["zerocopy_threaded"]["mean_ms"], 2),
                    "n/a",
                    _num(pipe["zerocopy_threaded"]["fps"], 2),
                ],
            ],
        ),
        "",
        (
            f"Threaded FPS / pydicom FPS = {_num(targets['decode_speedup'], 2)}. "
            f"Max absolute window error {_num(pipe['max_abs_diff'], 6)}. "
            f"Throughput target (threads faster than sequential pydicom): {'met' if targets['decode_faster'] else 'not met'}."
        ),
        "The threaded milliseconds per study are the batch wall time divided by the study count. A single clicked study is the sequential zero-copy row.",
        "",
        "## 2. False-positive boxes",
        "",
        "The baseline keeps lung residual peaks with confidence ≥ 0.5. The filter also requires spatial entropy ≤ τ. Entropy is the Shannon entropy, in nats, of the residual mass inside a 36 px window after pooling onto a 6×6 grid. τ is chosen on the tuning split as the cut with the largest false-positive drop that does not lower precision and keeps at least 85% of baseline recall. The table freezes that cut on the held-out split.",
        "",
        _md_table(
            ["Policy", "TP", "FP", "FN", "Precision", "Recall"],
            [
                ["confidence ≥ 0.5", str(base["tp"]), str(base["fp"]), str(base["fn"]), _num(base["precision"]), _num(base["recall"])],
                [
                    f"confidence ≥ 0.5 and H ≤ {_num(boxes['tau'], 3)}",
                    str(kept["tp"]),
                    str(kept["fp"]),
                    str(kept["fn"]),
                    _num(kept["precision"]),
                    _num(kept["recall"]),
                ],
            ],
        ),
        "",
        (
            f"False-positive reduction {_pct(boxes['fp_reduction'])} (40% target {'met' if targets['fp_reduction'] else 'not met'}). "
            f"Precision defense {'met' if targets['precision_defense'] else 'not met'}."
        ),
        "",
        "## 3. Worklist",
        "",
        "The study distribution is fused. Pneumothorax mass is the third class of the ConvNeXt study head. Nodule mass is the residual-peak score shifted by 0.46 and divided by 0.25, and scores at or below 0.46 contribute 0. The remainder is normal mass, and the three values are renormalized. Auto-pass requires pneumothorax probability, nodule probability, and the entropy of that distribution all below their cuts. Everything else is ordered by 5 × pneumothorax + 2 × nodule + 1.2 × entropy. Emergency wait is the time until a study whose label is pneumothorax is opened. An auto-passed emergency is a miss and is left out of the wait average.",
        "",
        _md_table(
            ["Kind", "Studies", "Mean nodule", "Mean pneumothorax", "Mean entropy", "Auto-pass"],
            kind_rows,
        ),
        "",
        (
            f"FIFO emergency wait {_num(work['fifo_emergency_wait_s'], 1)} s, "
            f"priority wait {_num(work['priority_emergency_wait_s'], 1)} s, "
            f"reduction {_pct(work['wait_reduction'])} (80% target {'met' if targets['wait_reduction'] else 'not met'})."
        ),
        (
            f"Normal auto-pass {_pct(work['normal_auto_pass_rate'])}. "
            f"Nodule review {_pct(work['nodule_review_rate'])}. "
            f"Emergency misses {work['emergency_misses']} / {work['emergencies']}."
        ),
        "",
    ]
    return "\n".join(lines)


def _stack_candidates(candidates: list[dict]) -> dict[str, np.ndarray]:
    if not candidates:
        return {
            "image": np.zeros(0, dtype=np.int64),
            "score": np.zeros(0, dtype=np.float64),
            "entropy": np.zeros(0, dtype=np.float64),
            "box": np.zeros((0, 4), dtype=np.float64),
        }
    return {
        "image": np.asarray([row["image"] for row in candidates], dtype=np.int64),
        "score": np.asarray([row["score"] for row in candidates], dtype=np.float64),
        "entropy": np.asarray([row["entropy"] for row in candidates], dtype=np.float64),
        "box": np.stack([row["box"] for row in candidates]),
    }


def _stack_gt(studies) -> tuple[np.ndarray, np.ndarray]:
    images = []
    boxes = []
    for index, study in enumerate(studies):
        for box in study.boxes:
            images.append(index)
            boxes.append(box)
    if not boxes:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    return np.asarray(images, dtype=np.int64), np.stack(boxes)


def _proposal_scores(studies) -> np.ndarray:
    scores = np.zeros(len(studies), dtype=np.float64)
    for index, study in enumerate(studies):
        found = propose(study.image, study.lung, core_entropy, limit=1)
        if found:
            scores[index] = found[0].score
    return scores


def _gallery(model, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    kinds = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    studies = [render_study(INPUT_SIZE, int(kind), rng) for kind in kinds]
    images = np.stack([study.image for study in studies])
    head, heat = predict(model, images)
    probabilities = fuse_study(head, _proposal_scores(studies))
    box = np.zeros((len(studies), 6, 4), dtype=np.float32)
    score = np.zeros((len(studies), 6), dtype=np.float32)
    entropy = np.zeros((len(studies), 6), dtype=np.float32)
    count = np.zeros(len(studies), dtype=np.int64)
    gt = np.zeros((len(studies), 2, 4), dtype=np.float32)
    gt_count = np.zeros(len(studies), dtype=np.int64)
    for index, study in enumerate(studies):
        found = propose(study.image, study.lung, core_entropy, limit=6)
        count[index] = len(found)
        for slot, candidate in enumerate(found):
            box[index, slot] = candidate.box
            score[index, slot] = candidate.score
            entropy[index, slot] = candidate.entropy
        gt_count[index] = study.boxes.shape[0]
        if study.boxes.size:
            gt[index, : study.boxes.shape[0]] = study.boxes
    return {
        "image": images.astype(np.float32),
        "heatmap": heat.astype(np.float32),
        "kind": kinds,
        "prob": probabilities.astype(np.float32),
        "det_box": box,
        "det_score": score,
        "det_entropy": entropy,
        "det_count": count,
        "gt_box": gt,
        "gt_count": gt_count,
    }


def run_benchmark(
    n_pipeline: int = 8,
    frame: int = 2048,
    n_train: int = 160,
    epochs: int = 16,
    n_tune: int = 90,
    n_eval: int = 90,
    n_worklist: int = 180,
    seed: int = 0,
) -> tuple[dict, dict[str, np.ndarray]]:
    separation = self_check()
    pipeline = benchmark_pipeline(n_studies=n_pipeline, size=frame, seed=seed)
    tune_cohort, tune_studies = build_cohort(n_tune, "detection", seed + 11)
    eval_cohort, eval_studies = build_cohort(n_eval, "detection", seed + 29)
    del tune_cohort, eval_cohort
    tune_candidates = collect_candidates(tune_studies)
    eval_candidates = collect_candidates(eval_studies)
    tuned = select_entropy_threshold(tune_studies, tune_candidates, SCORE_THRESHOLD)
    tau = float(tuned["chosen"]["tau"])
    eval_baseline = score_policy(eval_candidates, eval_studies, SCORE_THRESHOLD, None)
    eval_filtered = score_policy(eval_candidates, eval_studies, SCORE_THRESHOLD, tau)
    fp_drop = reduction(int(eval_baseline["fp"]), int(eval_filtered["fp"]))
    precision_ok = (
        eval_baseline["precision"] is not None
        and eval_filtered["precision"] is not None
        and eval_filtered["precision"] + 1e-9 >= eval_baseline["precision"]
    )
    fit = fit_detector(n_studies=n_train, epochs=epochs, seed=seed)
    work_cohort, work_studies = build_cohort(n_worklist, "worklist", seed + 41)
    head, _heat = predict(fit.model, work_cohort.images[:, 0])
    probabilities = fuse_study(head, _proposal_scores(work_studies))
    worklist = evaluate_worklist(probabilities, work_cohort.kinds)
    speedup = pipeline["zerocopy_threaded"]["fps"] / pipeline["pydicom_sequential"]["fps"]
    wait_ok = (
        worklist["wait_reduction"] is not None
        and worklist["wait_reduction"] >= WAIT_TARGET
        and worklist["emergency_misses"] == 0
    )
    report = {
        "suite": "dicom-stream-triage",
        "version": __version__,
        "synthetic": True,
        "seed": seed,
        "environment": collect_environment(),
        "separation": separation,
        "pipeline": pipeline,
        "boxes": {
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
            "tau": tau,
            "tune_studies": n_tune,
            "eval_studies": n_eval,
            "tune_baseline": tuned["baseline"],
            "eval_baseline": eval_baseline,
            "eval_filtered": eval_filtered,
            "fp_reduction": fp_drop,
        },
        "worklist": {key: value for key, value in worklist.items() if key not in {"auto_pass", "priority", "entropy", "priority_order"}},
        "train_loss": fit.train_loss,
        "targets": {
            "fp_reduction_target": FP_TARGET,
            "wait_reduction_target": WAIT_TARGET,
            "decode_faster": bool(speedup > 1.0),
            "decode_speedup": float(speedup),
            "fp_reduction": bool(fp_drop is not None and fp_drop >= FP_TARGET),
            "precision_defense": bool(precision_ok),
            "wait_reduction": bool(wait_ok),
        },
    }
    board = _gallery(fit.model, seed + 3)
    flat = _stack_candidates(eval_candidates)
    gt_image, gt_box = _stack_gt(eval_studies)
    board.update(
        {
            "eval_image": flat["image"],
            "eval_score": flat["score"],
            "eval_entropy": flat["entropy"],
            "eval_box": flat["box"],
            "eval_gt_image": gt_image,
            "eval_gt_box": gt_box,
            "eval_n": np.asarray([n_eval], dtype=np.int64),
            "wl_prob": probabilities.astype(np.float32),
            "wl_kind": work_cohort.kinds.astype(np.int64),
            "tau": np.asarray([tau], dtype=np.float64),
        }
    )
    return report, board


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic DICOM decode, box filter, and worklist benchmark.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--pipeline-studies", type=int, default=8)
    parser.add_argument("--frame", type=int, default=2048)
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--tune", type=int, default=90)
    parser.add_argument("--eval", type=int, default=90)
    parser.add_argument("--worklist", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "results" / "benchmark_report.json"))
    parser.add_argument("--report", default=str(ROOT / "BENCHMARK_RESULTS.md"))
    parser.add_argument("--dashboard", default=str(ROOT / "results" / "dashboard.npz"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.synthetic:
        print("Pass --synthetic. This suite does not read a PACS.", file=sys.stderr)
        return 2
    print(f"dicom_triage {__version__}: frame {args.frame}, train {args.train}, epochs {args.epochs}.", flush=True)
    report, board = run_benchmark(
        args.pipeline_studies,
        args.frame,
        args.train,
        args.epochs,
        args.tune,
        args.eval,
        args.worklist,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    Path(args.report).write_text(render_markdown(report), encoding="utf-8")
    np.savez_compressed(args.dashboard, **board)
    pipe = report["pipeline"]
    boxes = report["boxes"]
    work = report["worklist"]
    print(
        f"Decode threaded {pipe['zerocopy_threaded']['fps']:.2f} fps vs pydicom {pipe['pydicom_sequential']['fps']:.2f} fps",
        flush=True,
    )
    print(
        f"Boxes FP reduction {_pct(boxes['fp_reduction'])} at tau {boxes['tau']:.3f}",
        flush=True,
    )
    print(
        f"Worklist wait reduction {_pct(work['wait_reduction'])}, normal auto-pass {_pct(work['normal_auto_pass_rate'])}",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
