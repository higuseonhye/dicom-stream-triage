"""Local board for one synthetic chest worklist. Serves on 127.0.0.1:8504."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from dicom_triage.generator import KIND_NAMES
from dicom_triage.worklist_engine import (
    NODULE_PASS,
    PRED_ENTROPY_PASS,
    PTX_PASS,
    SCORE_THRESHOLD,
    evaluate_saved,
    evaluate_worklist,
    reduction,
)

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "results" / "benchmark_report.json"
DASHBOARD_PATH = ROOT / "results" / "dashboard.npz"

st.set_page_config(page_title="DICOM Stream Triage", layout="wide")


def _font() -> None:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic",):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_font()


@st.cache_resource
def load_board():
    if not REPORT_PATH.exists() or not DASHBOARD_PATH.exists():
        return None
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with np.load(DASHBOARD_PATH) as packed:
        board = {key: packed[key] for key in packed.files}
    return report, board


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _draw_case(board: dict, index: int, score_cut: float, entropy_cut: float) -> plt.Figure:
    image = board["image"][index]
    heat = board["heatmap"][index]
    figure, axis = plt.subplots(figsize=(4.8, 4.8))
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    axis.imshow(heat, cmap="magma", alpha=0.35, vmin=0, vmax=1, extent=(0, image.shape[1], image.shape[0], 0))
    for slot in range(int(board["gt_count"][index])):
        x0, y0, x1, y1 = board["gt_box"][index, slot]
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e5e7eb", linewidth=1.2))
    for slot in range(int(board["det_count"][index])):
        score = float(board["det_score"][index, slot])
        entropy = float(board["det_entropy"][index, slot])
        if score < score_cut:
            continue
        x0, y0, x1, y1 = board["det_box"][index, slot]
        kept = entropy <= entropy_cut
        color = "#0f766e" if kept else "#ea580c"
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.4))
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def _draw_queue(kinds: np.ndarray, order: np.ndarray) -> plt.Figure:
    emergency = kinds == 3
    figure, axis = plt.subplots(figsize=(7.2, 3.2))
    positions = np.arange(kinds.size)
    axis.scatter(positions, np.zeros_like(positions), s=18, color="#d1d5db", label="FIFO")
    ranked = np.full(kinds.size, np.nan)
    ranked[order] = np.arange(order.size)
    axis.scatter(np.flatnonzero(emergency), np.zeros(int(emergency.sum())), s=36, color="#b91c1c", label="FIFO emergency", zorder=3)
    queued = emergency & np.isfinite(ranked)
    axis.scatter(ranked[queued], np.full(int(queued.sum()), 1.0), s=36, color="#0f766e", label="Priority emergency", zorder=3)
    axis.set_yticks([0, 1], ["FIFO", "Priority"])
    axis.set_xlabel("Read order")
    axis.set_ylim(-0.6, 1.6)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    return figure


def main() -> None:
    st.title("DICOM Stream Triage")
    st.caption("Synthetic chest studies. Not for clinical use. No hospital archive and no vendor weights.")
    loaded = load_board()
    if loaded is None:
        st.warning("No result files. From the repository root, run `python -m dicom_triage.runner --synthetic` first.")
        return
    report, board = loaded
    targets = report["targets"]
    pipe = report["pipeline"]
    boxes = report["boxes"]
    work = report["worklist"]
    left, middle, right = st.columns(3)
    left.metric(
        "Threaded decode",
        f"{pipe['zerocopy_threaded']['fps']:.1f} fps",
        f"pydicom {pipe['pydicom_sequential']['fps']:.1f} fps · {'met' if targets['decode_faster'] else 'not met'}",
    )
    middle.metric(
        "False-positive reduction",
        _pct(boxes["fp_reduction"]),
        f"40% target {'met' if targets['fp_reduction'] else 'not met'}",
    )
    right.metric(
        "Emergency wait reduction",
        _pct(work["wait_reduction"]),
        f"80% target {'met' if targets['wait_reduction'] else 'not met'}",
    )
    st.caption(
        f"Precision defense {'met' if targets['precision_defense'] else 'not met'} · "
        f"normal auto-pass {_pct(work['normal_auto_pass_rate'])} · "
        f"emergency misses {work['emergency_misses']} · seed {report['seed']}"
    )

    labels = [f"{index + 1:02d} · {KIND_NAMES[int(board['kind'][index])]}" for index in range(board["image"].shape[0])]
    case = st.selectbox("Gallery study", options=list(range(len(labels))), format_func=lambda index: labels[index])
    score_col, entropy_col = st.columns(2)
    with score_col:
        score_cut = st.slider("Box confidence", 0.20, 0.90, SCORE_THRESHOLD, 0.05)
    with entropy_col:
        entropy_default = round(float(board["tau"][0]), 2)
        entropy_cut = st.slider("Spatial entropy cap (nat)", 1.50, 6.00, entropy_default, 0.01)

    chart, facts = st.columns((1.1, 1.0))
    with chart:
        st.pyplot(_draw_case(board, int(case), score_cut, entropy_cut), clear_figure=True)
    probs = board["prob"][int(case)]
    with facts:
        st.markdown(
            "White is the nodule ground truth. Teal boxes pass the entropy cap. "
            "Orange boxes clear confidence and fail the cap. The overlay is the ConvNeXt nodule heatmap."
        )
        one, two, three = st.columns(3)
        one.metric("Normal", f"{probs[0]:.2f}")
        two.metric("Nodule", f"{probs[1]:.2f}")
        three.metric("Pneumothorax", f"{probs[2]:.2f}")
        kept = 0
        shown = 0
        for slot in range(int(board["det_count"][int(case)])):
            if float(board["det_score"][int(case), slot]) < score_cut:
                continue
            shown += 1
            if float(board["det_entropy"][int(case), slot]) <= entropy_cut:
                kept += 1
        st.caption(f"This study has {shown} boxes above the confidence cut, of which {kept} also pass the entropy cap.")

    live = evaluate_saved(
        board["eval_image"],
        board["eval_score"],
        board["eval_entropy"],
        board["eval_box"],
        board["eval_gt_image"],
        board["eval_gt_box"],
        int(board["eval_n"][0]),
        score_cut,
        entropy_cut,
    )
    baseline = evaluate_saved(
        board["eval_image"],
        board["eval_score"],
        board["eval_entropy"],
        board["eval_box"],
        board["eval_gt_image"],
        board["eval_gt_box"],
        int(board["eval_n"][0]),
        score_cut,
        None,
    )
    official = abs(score_cut - SCORE_THRESHOLD) < 1e-6 and abs(entropy_cut - float(board["tau"][0])) < 0.011
    st.subheader("Held-out boxes" + (" · recorded cuts" if official else " · slider cuts"))
    a, b, c, d = st.columns(4)
    a.metric("Baseline FP", baseline["fp"])
    b.metric("Filtered FP", live["fp"])
    c.metric("False-positive reduction", _pct(reduction(int(baseline["fp"]), int(live["fp"]))))
    d.metric("Filtered precision", "n/a" if live["precision"] is None else f"{live['precision']:.3f}")

    st.subheader("Worklist")
    nodule_cut, ptx_cut, pred_cut = st.columns(3)
    with nodule_cut:
        nodule_threshold = st.slider("Auto-pass nodule cap", 0.05, 0.80, NODULE_PASS, 0.05)
    with ptx_cut:
        ptx_threshold = st.slider("Auto-pass pneumothorax cap", 0.05, 0.80, PTX_PASS, 0.05)
    with pred_cut:
        pred_threshold = st.slider("Auto-pass entropy cap (nat)", 0.05, 1.20, PRED_ENTROPY_PASS, 0.05)
    queued = evaluate_worklist(board["wl_prob"], board["wl_kind"], nodule_threshold, ptx_threshold, pred_threshold)
    st.pyplot(_draw_queue(board["wl_kind"], queued["priority_order"]), clear_figure=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("FIFO emergency wait", f"{queued['fifo_emergency_wait_s']:.0f} s" if queued["fifo_emergency_wait_s"] is not None else "n/a")
    q2.metric("Priority emergency wait", f"{queued['priority_emergency_wait_s']:.0f} s" if queued["priority_emergency_wait_s"] is not None else "n/a")
    q3.metric("Wait reduction", _pct(queued["wait_reduction"]))
    q4.metric("Normal auto-pass", _pct(queued["normal_auto_pass_rate"]))
    preview = []
    for rank, index in enumerate(queued["priority_order"][:12]):
        preview.append(
            {
                "Order": rank + 1,
                "Arrival": int(index) + 1,
                "Kind": KIND_NAMES[int(board["wl_kind"][index])],
                "Nodule": round(float(board["wl_prob"][index, 1]), 3),
                "Pneumothorax": round(float(board["wl_prob"][index, 2]), 3),
                "Entropy": round(float(queued["entropy"][index]), 3),
            }
        )
    st.table(preview, hide_index=True, width="stretch")


main()
