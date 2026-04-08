from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class PreprocessInfo:
    orig_h: int
    orig_w: int
    square_size: int
    top_pad: int
    bottom_pad: int
    left_pad: int
    right_pad: int


def pad_to_square(image: np.ndarray) -> tuple[np.ndarray, PreprocessInfo]:
    height, width = image.shape[:2]
    size = max(height, width)
    top_pad = (size - height) // 2
    bottom_pad = size - height - top_pad
    left_pad = (size - width) // 2
    right_pad = size - width - left_pad
    padded = cv2.copyMakeBorder(
        image,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    info = PreprocessInfo(
        orig_h=height,
        orig_w=width,
        square_size=size,
        top_pad=top_pad,
        bottom_pad=bottom_pad,
        left_pad=left_pad,
        right_pad=right_pad,
    )
    return padded, info


def preprocess_slice(image: np.ndarray) -> tuple[np.ndarray, PreprocessInfo]:
    padded, info = pad_to_square(image)
    normalized = cv2.normalize(padded, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    pil_image = Image.fromarray(normalized)
    resized = pil_image.resize((512, 512))
    arr = np.asarray(resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr, info


def inverse_resize_heatmap(heatmap_512: np.ndarray, info: PreprocessInfo) -> np.ndarray:
    if heatmap_512.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got {heatmap_512.shape}")

    square = cv2.resize(heatmap_512, (info.square_size, info.square_size), interpolation=cv2.INTER_LINEAR)

    h_start = info.top_pad
    h_stop = info.square_size - info.bottom_pad
    w_start = info.left_pad
    w_stop = info.square_size - info.right_pad

    cropped = square[h_start:h_stop, w_start:w_stop]
    if cropped.shape != (info.orig_h, info.orig_w):
        cropped = cv2.resize(cropped, (info.orig_w, info.orig_h), interpolation=cv2.INTER_LINEAR)
    return cropped
