from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from motionscore.inference.scoring import PredictionResult
from motionscore.utils import ensure_parent


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _panel_indices(depth: int, max_panels: int) -> list[int]:
    max_panels = int(max(1, min(9, max_panels)))
    if depth <= 0:
        return [0]
    idx = np.linspace(0, depth - 1, num=min(depth, max_panels), dtype=int).tolist()
    return sorted(set(int(v) for v in idx))


def write_prediction_preview_png(
    volume_xyz: np.ndarray,
    prediction: PredictionResult,
    output_path: str | Path,
    max_panels: int = 3,
) -> Path:
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume [x,y,z], got {volume_xyz.shape}")

    z_count = int(volume_xyz.shape[2])
    panel_indices = _panel_indices(z_count, max_panels=max_panels)

    panel_size = 256
    gap = 8
    header_height = 36

    panels: list[Image.Image] = []
    for z in panel_indices:
        z_safe = int(max(0, min(z_count - 1, z))) if z_count > 0 else 0
        gray = _normalize_to_uint8(volume_xyz[:, :, z_safe] if z_count > 0 else np.zeros((panel_size, panel_size)))
        panel = Image.fromarray(gray, mode="L").convert("RGB").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(panel)

        grade = prediction.slice_grades[z_safe] if z_safe < len(prediction.slice_grades) else prediction.automatic_grade
        conf = prediction.slice_confidences[z_safe] if z_safe < len(prediction.slice_confidences) else prediction.mean_confidence
        conf_pct = int(round(float(conf) * 100.0)) if float(conf) <= 1.0 else int(round(float(conf)))
        label = f"z={z_safe} g={grade} c={conf_pct}%"

        draw.rectangle([(0, panel_size - 24), (panel_size, panel_size)], fill=(0, 0, 0))
        draw.text((6, panel_size - 20), label, fill=(255, 255, 255))
        panels.append(panel)

    canvas_w = len(panels) * panel_size + max(0, len(panels) - 1) * gap
    canvas_h = header_height + panel_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))

    draw = ImageDraw.Draw(canvas)
    header = f"MotionScore preview | automatic_grade={prediction.automatic_grade} | confidence={prediction.automatic_confidence}%"
    draw.text((8, 10), header, fill=(240, 240, 240))

    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_height))
        x += panel_size + gap

    output_path = Path(output_path)
    ensure_parent(output_path)
    canvas.save(output_path, format="PNG")
    return output_path


def write_slice_profile_png(
    prediction: PredictionResult,
    output_path: str | Path,
) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to export slice profile plots") from exc

    grades = np.asarray(prediction.slice_grades, dtype=np.int32)
    conf = np.asarray(prediction.slice_confidences, dtype=np.float32)
    if grades.size == 0:
        raise ValueError("prediction.slice_grades is empty")
    if conf.size != grades.size:
        raise ValueError("slice_confidences length must match slice_grades length")

    z = np.arange(grades.size, dtype=np.int32)
    conf_pct = conf * 100.0 if float(np.nanmax(conf)) <= 1.0 else conf

    votes = np.asarray(prediction.votes, dtype=np.float32)
    if votes.ndim != 2 or votes.shape[0] != grades.size or votes.shape[1] != 5:
        # Fallback: synthesize one-hot vote bars from slice grades.
        votes = np.zeros((grades.size, 5), dtype=np.float32)
        safe_idx = np.clip(grades - 1, 0, 4)
        votes[np.arange(grades.size), safe_idx] = 1.0

    row_sums = np.sum(votes, axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    votes = votes / row_sums

    # Keep grade colors aligned with the Slicer review UI:
    # Grade 1-2 blue shades, Grade 3 yellow, Grade 4-5 red shades.
    grade_colors = ["#125FAF", "#4B97E0", "#F2C94C", "#E56A5D", "#B73A31"]
    grade_labels = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]

    fig, ax_stack = plt.subplots(figsize=(10, 3.4), dpi=150)
    ax_conf = ax_stack.twinx()

    bottom = np.zeros(grades.size, dtype=np.float32)
    for i in range(5):
        heights = votes[:, i]
        ax_stack.bar(
            z,
            heights,
            bottom=bottom,
            width=1.0,
            color=grade_colors[i],
            edgecolor="none",
            align="center",
            label=grade_labels[i],
        )
        bottom += heights

    ax_conf.plot(z, conf_pct, color="#111111", linewidth=1.25, alpha=0.9, label="Confidence")
    ax_conf.axhline(float(prediction.automatic_confidence), color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_conf.set_ylabel("Confidence (%)", color="#111111")
    ax_conf.set_ylim(0.0, 100.0)

    ax_stack.set_xlabel("Slice index (z)")
    ax_stack.set_ylabel("Vote proportion")
    ax_stack.set_ylim(0.0, 1.0)
    ax_stack.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_stack.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax_stack.set_title(
        f"MotionScore per-slice votes | automatic grade={prediction.automatic_grade} | confidence={prediction.automatic_confidence}%"
    )
    ax_stack.legend(loc="upper left", ncol=5, fontsize=7, frameon=False)

    output_path = Path(output_path)
    ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path
