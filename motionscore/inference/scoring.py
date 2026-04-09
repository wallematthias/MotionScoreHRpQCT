from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from skimage.filters import gaussian

from motionscore.inference.preprocessing import PreprocessInfo, preprocess_slice

if TYPE_CHECKING:
    from motionscore.inference.model import ModelEnsemble


@dataclass(slots=True)
class PredictionResult:
    automatic_grade: int
    automatic_confidence: int
    mean_confidence: float
    slice_grades: list[int]
    slice_confidences: list[float]
    stack_grades: list[float]
    stack_confidences: list[float]
    stack_ranges: list[tuple[int, int]]
    preprocessed_scan: np.ndarray | None
    preprocess_infos: list[PreprocessInfo]
    votes: np.ndarray


def compute_stack_ranges(depth: int, stackheight: int, on_incomplete_stack: str = "keep_last") -> list[tuple[int, int]]:
    if stackheight <= 0:
        raise ValueError("stackheight must be > 0")

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < depth:
        end = min(depth, start + stackheight)
        is_incomplete = (end - start) < stackheight

        if is_incomplete and on_incomplete_stack == "drop_last":
            break
        if is_incomplete and on_incomplete_stack == "error":
            raise ValueError(
                f"Scan depth {depth} is not divisible by stackheight={stackheight}"
            )

        ranges.append((start, end))
        start += stackheight

    if not ranges:
        ranges = [(0, depth)]
    return ranges


def _vote_predictions(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_models, n_slices, n_classes = predictions.shape

    result_binary = np.zeros((n_models, n_slices, n_classes), dtype=np.float32)
    result_matrix = np.zeros((n_models, n_slices), dtype=np.int32)

    for i in range(n_models):
        max_per_slice = np.max(predictions[i], axis=1, keepdims=True)
        safe = np.where(max_per_slice <= 0, 1.0, max_per_slice)
        result_binary[i] = np.floor(predictions[i] / safe)
        result_matrix[i] = np.argmax(result_binary[i], axis=1)

    votes = np.zeros((n_slices, n_classes), dtype=np.float32)
    for s in range(n_slices):
        bins = np.bincount(result_matrix[:, s].astype(np.int32), minlength=n_classes)
        votes[s] = bins / float(n_models)

    slice_grades = np.argmax(votes, axis=1) + 1
    slice_conf = np.max(votes, axis=1)
    return votes, slice_grades, slice_conf


def predict_scan(
    volume_xyz: np.ndarray,
    ensemble: ModelEnsemble,
    stackheight: int = 168,
    on_incomplete_stack: str = "keep_last",
    slice_batch_size: int = 64,
    slice_step: int = 1,
    retain_preprocessed: bool = False,
) -> PredictionResult:
    if volume_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume [x,y,z], got {volume_xyz.shape}")

    depth = volume_xyz.shape[2]
    stackheight = min(max(1, int(stackheight)), depth)

    filtered = gaussian(volume_xyz, sigma=0.8, truncate=1.25)
    if int(slice_batch_size) <= 0:
        raise ValueError("slice_batch_size must be > 0")
    batch_size = int(slice_batch_size)
    if int(slice_step) <= 0:
        raise ValueError("slice_step must be > 0")
    step = int(slice_step)
    if retain_preprocessed and step != 1:
        raise ValueError("slice_step must be 1 when retain_preprocessed=True")

    all_infos: list[PreprocessInfo] = []
    retained_scan: list[np.ndarray] | None = [] if retain_preprocessed else None
    pending_batch: list[np.ndarray] = []
    pending_indices: list[int] = []
    predicted_indices: list[int] = []
    votes_chunks: list[np.ndarray] = []
    slice_grades_chunks: list[np.ndarray] = []
    slice_conf_chunks: list[np.ndarray] = []

    slice_indices = list(range(0, depth, step))
    for idx_pos, i in enumerate(slice_indices):
        arr, info = preprocess_slice(filtered[:, :, i])
        if retain_preprocessed and retained_scan is not None:
            retained_scan.append(arr)
            all_infos.append(info)
        elif retain_preprocessed:
            # Defensive branch; should not happen due initialization above.
            retained_scan = [arr]
            all_infos = [info]

        pending_batch.append(arr)
        pending_indices.append(i)
        if len(pending_batch) < batch_size and idx_pos < (len(slice_indices) - 1):
            continue

        batch = np.asarray(pending_batch, dtype=np.float32) / 255.0
        predictions = ensemble.predict(batch)
        chunk_votes, chunk_grades, chunk_conf = _vote_predictions(predictions)
        votes_chunks.append(chunk_votes)
        slice_grades_chunks.append(chunk_grades)
        slice_conf_chunks.append(chunk_conf)
        predicted_indices.extend(pending_indices)
        pending_batch.clear()
        pending_indices.clear()

    if not votes_chunks:
        raise RuntimeError("No slice predictions were generated")

    votes_sparse = np.concatenate(votes_chunks, axis=0)
    slice_grades_sparse = np.concatenate(slice_grades_chunks, axis=0)
    slice_conf_sparse = np.concatenate(slice_conf_chunks, axis=0)

    pred_idx = np.asarray(predicted_indices, dtype=np.int32)
    votes = np.zeros((depth, votes_sparse.shape[1]), dtype=np.float32)
    slice_grades = np.zeros(depth, dtype=np.int32)
    slice_conf = np.full(depth, -1.0, dtype=np.float32)
    votes[pred_idx] = votes_sparse
    slice_grades[pred_idx] = slice_grades_sparse
    slice_conf[pred_idx] = slice_conf_sparse

    automatic_grade = int(np.round(np.mean(slice_grades_sparse), 0))
    mean_conf = float(np.round(np.mean(slice_conf_sparse), 2))
    automatic_conf = int(np.round(mean_conf * 100, 0))

    ranges = compute_stack_ranges(depth, stackheight, on_incomplete_stack=on_incomplete_stack)
    stack_grades = []
    stack_conf = []
    for start, end in ranges:
        in_stack = (pred_idx >= int(start)) & (pred_idx < int(end))
        if not np.any(in_stack):
            stack_grades.append(0.0)
            stack_conf.append(-1.0)
            continue
        stack_grades.append(float(np.round(np.mean(slice_grades_sparse[in_stack]), 0)))
        stack_conf.append(float(np.round(np.mean(slice_conf_sparse[in_stack]), 2)))

    if retain_preprocessed and retained_scan is not None:
        preprocessed_scan = np.asarray(retained_scan, dtype=np.float32)
    else:
        preprocessed_scan = None
        all_infos = []

    return PredictionResult(
        automatic_grade=automatic_grade,
        automatic_confidence=automatic_conf,
        mean_confidence=mean_conf,
        slice_grades=[int(v) for v in slice_grades.tolist()],
        slice_confidences=[float(v) for v in slice_conf.tolist()],
        stack_grades=stack_grades,
        stack_confidences=stack_conf,
        stack_ranges=ranges,
        preprocessed_scan=preprocessed_scan,
        preprocess_infos=all_infos,
        votes=votes,
    )
