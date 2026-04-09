from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import gaussian

from motionscore.inference.torch_model import build_torch_model
from motionscore.inference.preprocessing import preprocess_slice
from motionscore.io.aim import read_aim
from motionscore.utils import read_tsv, utc_now_iso, write_json


@dataclass(slots=True)
class TrainConfig:
    manifest_path: Path
    init_model_dir: Path
    output_model_dir: Path
    device: str = "auto"
    scaling: str = "native"
    batch_size: int = 24
    epochs_head: int = 10
    epochs_finetune: int = 50
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_rotate: bool = False
    aug_crop: bool = False
    max_cache_scans: int = 64
    num_workers: int = 0
    seed: int = 13


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for training. Install with pip install -e '.[torch]'.") from exc
    return torch, DataLoader, Dataset


def _resolve_torch_device(requested: str) -> str:
    torch, _, _ = _require_torch()
    device = str(requested).strip().lower()
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, mps, cuda")
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("Requested device=cuda but CUDA is not available")
        return "cuda"
    if device == "mps":
        has_mps = bool(getattr(torch.backends, "mps", None))
        if not has_mps or not bool(torch.backends.mps.is_available()):
            raise RuntimeError("Requested device=mps but MPS is not available")
        return "mps"

    has_mps = bool(getattr(torch.backends, "mps", None))
    if has_mps and bool(torch.backends.mps.is_available()):
        return "mps"
    if bool(torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def _as_int(text: str | int | float | None, default: int = 0) -> int:
    if text is None:
        return default
    if isinstance(text, int):
        return int(text)
    txt = str(text).strip()
    if not txt:
        return default
    return int(float(txt))


def _as_float(text: str | int | float | None, default: float = 0.0) -> float:
    if text is None:
        return default
    if isinstance(text, (float, int)):
        return float(text)
    txt = str(text).strip()
    if not txt:
        return default
    return float(txt)


def _quadratic_weighted_kappa(confusion: np.ndarray) -> float:
    if confusion.size == 0:
        return 0.0
    n = float(np.sum(confusion))
    if n <= 0.0:
        return 0.0

    k = confusion.shape[0]
    obs = confusion / n

    row_marg = np.sum(obs, axis=1, keepdims=True)
    col_marg = np.sum(obs, axis=0, keepdims=True)
    expected = row_marg @ col_marg

    denom = 0.0
    numer = 0.0
    norm = float((k - 1) ** 2) if k > 1 else 1.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / norm
            numer += w * float(obs[i, j])
            denom += w * float(expected[i, j])

    if denom <= 1e-12:
        return 1.0 if numer <= 1e-12 else 0.0
    return float(1.0 - (numer / denom))


def _compute_metrics(y_true: list[int], y_pred: list[int], n_classes: int = 5) -> dict[str, Any]:
    if not y_true:
        return {
            "n": 0,
            "accuracy": 0.0,
            "weighted_kappa": 0.0,
            "confusion": [[0 for _ in range(n_classes)] for _ in range(n_classes)],
        }

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            confusion[t, p] += 1

    correct = int(np.trace(confusion))
    total = int(np.sum(confusion))
    acc = float(correct / total) if total > 0 else 0.0

    return {
        "n": total,
        "accuracy": acc,
        "weighted_kappa": _quadratic_weighted_kappa(confusion.astype(np.float32)),
        "confusion": confusion.tolist(),
    }


def _write_training_plot_png(points: list[dict[str, Any]], output_path: Path, *, title: str) -> Path:
    width = 1024
    height = 520
    margin_left = 78
    margin_right = 88
    margin_top = 56
    margin_bottom = 58
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_w = max(1, plot_right - plot_left)
    plot_h = max(1, plot_bottom - plot_top)

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(plot_left, plot_top), (plot_right, plot_bottom)], outline=(60, 60, 60), width=1)
    draw.text((16, 14), title, fill=(20, 20, 20))

    if not points:
        draw.text((plot_left + 8, plot_top + 8), "waiting for epochs...", fill=(80, 80, 80))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return output_path

    xs = [float(max(1, int(p.get("x", 1)))) for p in points]
    train_loss = [float(p.get("train_loss", 0.0)) for p in points]
    val_loss = [float(p.get("val_loss", 0.0)) for p in points]
    lr_values = [float(p.get("lr", 0.0)) for p in points]

    x_min = min(xs)
    x_max = max(xs)
    if x_max <= x_min:
        x_max = x_min + 1.0

    loss_vals = [v for v in (train_loss + val_loss) if np.isfinite(v)]
    if not loss_vals:
        loss_vals = [0.0, 1.0]
    loss_min = min(loss_vals)
    loss_max = max(loss_vals)
    if loss_max <= loss_min:
        loss_max = loss_min + 1.0

    lr_vals = [v for v in lr_values if np.isfinite(v) and v >= 0.0]
    if not lr_vals:
        lr_vals = [0.0, 1.0]
    lr_min = min(lr_vals)
    lr_max = max(lr_vals)
    if lr_max <= lr_min:
        lr_max = lr_min + 1.0

    def _x_to_px(xv: float) -> int:
        return int(round(plot_left + ((xv - x_min) / (x_max - x_min)) * plot_w))

    def _loss_to_py(yv: float) -> int:
        return int(round(plot_bottom - ((yv - loss_min) / (loss_max - loss_min)) * plot_h))

    def _lr_to_py(yv: float) -> int:
        return int(round(plot_bottom - ((yv - lr_min) / (lr_max - lr_min)) * plot_h))

    # Horizontal guide lines.
    for i in range(5):
        y = int(round(plot_top + (i / 4.0) * plot_h))
        draw.line([(plot_left, y), (plot_right, y)], fill=(236, 236, 236), width=1)

    # Stage transition markers.
    last_stage = str(points[0].get("stage", "")).strip()
    for p in points[1:]:
        stage = str(p.get("stage", "")).strip()
        if stage != last_stage:
            xv = _x_to_px(float(max(1, int(p.get("x", 1)))))
            draw.line([(xv, plot_top), (xv, plot_bottom)], fill=(200, 200, 200), width=1)
            draw.text((xv + 4, plot_top + 4), stage, fill=(120, 120, 120))
            last_stage = stage

    def _draw_series(values: list[float], color: tuple[int, int, int], mapper) -> None:
        series_pts: list[tuple[int, int]] = []
        for x, y in zip(xs, values):
            if not np.isfinite(y):
                continue
            series_pts.append((_x_to_px(float(x)), mapper(float(y))))
        if len(series_pts) >= 2:
            draw.line(series_pts, fill=color, width=3)

    _draw_series(train_loss, (40, 114, 225), _loss_to_py)
    _draw_series(val_loss, (216, 76, 59), _loss_to_py)
    _draw_series(lr_values, (39, 174, 96), _lr_to_py)

    # Axis labels and legend.
    draw.text((plot_left, plot_bottom + 10), "Epoch", fill=(40, 40, 40))
    draw.text((10, plot_top), f"Loss [{loss_min:.4f}..{loss_max:.4f}]", fill=(40, 40, 40))
    draw.text((plot_right + 8, plot_top), f"LR [{lr_min:.6f}..{lr_max:.6f}]", fill=(40, 40, 40))

    legend_y = 34
    draw.rectangle([(plot_left, legend_y), (plot_left + 14, legend_y + 10)], fill=(40, 114, 225))
    draw.text((plot_left + 20, legend_y - 2), "Train loss", fill=(40, 40, 40))
    draw.rectangle([(plot_left + 130, legend_y), (plot_left + 144, legend_y + 10)], fill=(216, 76, 59))
    draw.text((plot_left + 150, legend_y - 2), "Validation loss", fill=(40, 40, 40))
    draw.rectangle([(plot_left + 304, legend_y), (plot_left + 318, legend_y + 10)], fill=(39, 174, 96))
    draw.text((plot_left + 324, legend_y - 2), "Learning rate", fill=(40, 40, 40))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path


class _SliceManifestDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        manifest_parent: Path,
        scaling: str = "native",
        max_cache_scans: int = 64,
    ):
        torch, _, Dataset = _require_torch()
        self._torch = torch
        self.rows = list(rows)
        self.manifest_parent = Path(manifest_parent).resolve()
        self.scaling = scaling
        self.max_cache_scans = int(max_cache_scans)
        self.unlimited_cache = self.max_cache_scans <= 0
        if self.max_cache_scans <= 0:
            self.max_cache_scans = 0
        self._scan_cache: dict[str, np.ndarray] = {}
        self._scan_cache_order: list[str] = []
        self._slice_db_cache: dict[str, np.ndarray] = {}
        self._slice_db_cache_order: list[str] = []
        self.rows_with_cache = 0
        self.rows_without_cache = 0
        for row in self.rows:
            if self._row_cache_target(row) is None:
                self.rows_without_cache += 1
            else:
                self.rows_with_cache += 1

        class _TorchDataset(Dataset):
            pass

        self.dataset_cls = _TorchDataset

    def _touch_lru(self, key: str, order: list[str]) -> None:
        if self.unlimited_cache:
            return
        try:
            order.remove(key)
        except ValueError:
            pass
        order.append(key)

    def _evict_lru(self, cache: dict[str, np.ndarray], order: list[str]) -> None:
        if self.unlimited_cache:
            return
        while len(order) > self.max_cache_scans:
            drop = order.pop(0)
            cache.pop(drop, None)

    def _row_cache_target(self, row: dict[str, str]) -> tuple[Path, int] | None:
        cache_path_txt = str(row.get("cache_npy_path", "")).strip()
        if not cache_path_txt:
            return None
        cache_index_txt = str(row.get("cache_index", "")).strip()
        if not cache_index_txt:
            return None
        cache_path = Path(cache_path_txt)
        if not cache_path.is_absolute():
            cache_path = (self.manifest_parent / cache_path).resolve()
        else:
            cache_path = cache_path.resolve()
        return cache_path, _as_int(cache_index_txt, default=0)

    def _load_scan(self, raw_image_path: str) -> np.ndarray:
        key = str(Path(raw_image_path).resolve())
        if key in self._scan_cache:
            self._touch_lru(key, self._scan_cache_order)
            return self._scan_cache[key]

        aim = read_aim(Path(key), scaling=self.scaling)
        if aim.data.ndim != 3:
            raise ValueError(f"Expected 3D volume for {key}, got {aim.data.shape}")
        filtered = gaussian(aim.data, sigma=0.8, truncate=1.25).astype(np.float32)

        self._scan_cache[key] = filtered
        self._touch_lru(key, self._scan_cache_order)
        self._evict_lru(self._scan_cache, self._scan_cache_order)
        return filtered

    def _load_slice_db(self, cache_path: Path) -> np.ndarray:
        key = str(cache_path.resolve())
        if key in self._slice_db_cache:
            self._touch_lru(key, self._slice_db_cache_order)
            return self._slice_db_cache[key]

        if not cache_path.exists():
            raise FileNotFoundError(f"Cached slice DB not found: {cache_path}")
        data = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if data.ndim != 3:
            raise ValueError(f"Expected cached slice DB shape [N,H,W] in {cache_path}, got {data.shape}")
        self._slice_db_cache[key] = data
        self._touch_lru(key, self._slice_db_cache_order)
        self._evict_lru(self._slice_db_cache, self._slice_db_cache_order)
        return data

    def _getitem(self, idx: int):
        row = self.rows[idx]
        cache_target = self._row_cache_target(row)
        label = _as_int(row.get("label"), default=1)
        label = int(min(5, max(1, label))) - 1
        weight = float(max(0.01, _as_float(row.get("sample_weight"), default=1.0)))
        if cache_target is not None:
            cache_path, cache_index = cache_target
            slices = self._load_slice_db(cache_path)
            local_idx = int(max(0, min(slices.shape[0] - 1, int(cache_index))))
            arr = np.asarray(slices[local_idx], dtype=np.float32)
        else:
            volume = self._load_scan(row["raw_image_path"])
            z = int(max(0, min(volume.shape[2] - 1, _as_int(row.get("slice_index"), default=0))))
            arr, _ = preprocess_slice(volume[:, :, z])
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] > 0:
                arr = arr[:, :, 0]
        arr = arr / 255.0

        x = self._torch.from_numpy(arr).unsqueeze(0)
        y = self._torch.tensor(label, dtype=self._torch.long)
        w = self._torch.tensor(weight, dtype=self._torch.float32)
        return x, y, w

    def to_torch_dataset(self):
        outer = self

        class _ConcreteDataset(self.dataset_cls):
            def __len__(self):
                return len(outer.rows)

            def __getitem__(self, idx):
                return outer._getitem(int(idx))

        return _ConcreteDataset()


def _set_trainable_stage(model: Any, stage: str) -> None:
    stage_norm = str(stage).strip().lower()
    if stage_norm not in {"head", "finetune"}:
        raise ValueError("stage must be one of: head, finetune")

    for p in model.parameters():
        p.requires_grad = stage_norm == "finetune"

    if stage_norm == "head":
        for name in ("fc1", "fc2"):
            if hasattr(model, name):
                for p in getattr(model, name).parameters():
                    p.requires_grad = True


def _weighted_nll_loss(torch, probs, targets, weights):
    p = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-8)
    per_item = -torch.log(p) * weights
    return per_item.sum() / weights.sum().clamp(min=1e-8)


def _augment_batch(torch, x, *, hflip: bool, vflip: bool, rotate: bool, crop: bool):
    if x.ndim != 4:
        return x
    out = x
    if bool(hflip):
        mask = torch.rand((out.shape[0],), device=out.device) < 0.5
        if bool(mask.any()):
            out[mask] = torch.flip(out[mask], dims=[3])
    if bool(vflip):
        mask = torch.rand((out.shape[0],), device=out.device) < 0.5
        if bool(mask.any()):
            out[mask] = torch.flip(out[mask], dims=[2])
    if bool(rotate):
        mask = torch.rand((out.shape[0],), device=out.device) < 0.5
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            for sample_idx in idx.tolist():
                k = int(torch.randint(1, 4, (1,), device=out.device).item())
                out[sample_idx] = torch.rot90(out[sample_idx], k=k, dims=[1, 2])
    if bool(crop):
        import torch.nn.functional as F

        mask = torch.rand((out.shape[0],), device=out.device) < 0.5
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            _, _, height, width = out.shape
            min_scale = 0.85
            for sample_idx in idx.tolist():
                scale = float(torch.empty(1, device=out.device).uniform_(min_scale, 1.0).item())
                crop_h = max(8, int(round(height * scale)))
                crop_w = max(8, int(round(width * scale)))
                max_y = max(0, height - crop_h)
                max_x = max(0, width - crop_w)
                start_y = int(torch.randint(0, max_y + 1, (1,), device=out.device).item()) if max_y > 0 else 0
                start_x = int(torch.randint(0, max_x + 1, (1,), device=out.device).item()) if max_x > 0 else 0
                cropped = out[sample_idx:sample_idx + 1, :, start_y:start_y + crop_h, start_x:start_x + crop_w]
                out[sample_idx:sample_idx + 1] = F.interpolate(
                    cropped,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
    return out


def _run_epoch(
    model,
    loader,
    *,
    device: str,
    optimizer=None,
    aug_hflip: bool = False,
    aug_vflip: bool = False,
    aug_rotate: bool = False,
    aug_crop: bool = False,
):
    torch, _, _ = _require_torch()
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        if train_mode and (bool(aug_hflip) or bool(aug_vflip) or bool(aug_rotate) or bool(aug_crop)):
            x = _augment_batch(
                torch,
                x,
                hflip=bool(aug_hflip),
                vflip=bool(aug_vflip),
                rotate=bool(aug_rotate),
                crop=bool(aug_crop),
            )

        with torch.set_grad_enabled(train_mode):
            probs = model(x)
            loss = _weighted_nll_loss(torch, probs=probs, targets=y, weights=w)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_batches += 1

        pred = torch.argmax(probs.detach(), dim=1)
        y_true.extend([int(v) for v in y.detach().cpu().tolist()])
        y_pred.extend([int(v) for v in pred.cpu().tolist()])

    avg_loss = total_loss / float(max(1, total_batches))
    metrics = _compute_metrics(y_true=y_true, y_pred=y_pred, n_classes=5)
    metrics["loss"] = avg_loss
    return metrics


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    rows = read_tsv(manifest_path)
    if not rows:
        raise FileNotFoundError(f"Manifest has no rows: {manifest_path}")
    return rows


def _split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if str(row.get("split", "")).strip().lower() == "train"]
    val_rows = [row for row in rows if str(row.get("split", "")).strip().lower() == "val"]
    test_rows = [row for row in rows if str(row.get("split", "")).strip().lower() == "test"]
    if not train_rows:
        raise ValueError("Manifest has no train rows")
    if not val_rows:
        raise ValueError("Manifest has no val rows")
    if not test_rows:
        raise ValueError("Manifest has no test rows")
    return train_rows, val_rows, test_rows


def _checkpoint_state_dict(model_path: Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch, _, _ = _require_torch()
    payload = torch.load(str(model_path), map_location=torch.device(map_location))
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload in {model_path}: {type(payload)}")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint state_dict must be a dict in {model_path}")
    return state


def _train_one_model(
    *,
    checkpoint_path: Path,
    model_index: int,
    cfg: TrainConfig,
    train_loader,
    val_loader,
    device: str,
) -> tuple[Path, dict[str, Any]]:
    torch, _, _ = _require_torch()
    model = build_torch_model()
    state = _checkpoint_state_dict(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    history: list[dict[str, Any]] = []
    plot_points: list[dict[str, Any]] = []
    plot_x = 0
    best_state = None
    best_key = float("-inf")
    best_epoch = -1

    # Stage A: train only classifier layers.
    _set_trainable_stage(model, stage="head")
    optim_head = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(cfg.lr_head))
    head_best_val_loss = float("inf")
    head_no_improve = 0
    for epoch in range(int(max(0, cfg.epochs_head))):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optim_head,
            aug_hflip=cfg.aug_hflip,
            aug_vflip=cfg.aug_vflip,
            aug_rotate=cfg.aug_rotate,
            aug_crop=cfg.aug_crop,
        )
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)
        lr = float(optim_head.param_groups[0].get("lr", cfg.lr_head))
        score = float(val_metrics.get("weighted_kappa", 0.0))
        val_loss = float(val_metrics.get("loss", 0.0))
        plot_x += 1
        history.append(
            {
                "stage": "head",
                "stage_label": "classifier",
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        plot_points.append(
            {
                "x": plot_x,
                "stage": "classifier",
                "model_index": int(model_index),
                "train_loss": float(train_metrics.get("loss", 0.0)),
                "val_loss": float(val_metrics.get("loss", 0.0)),
                "lr": lr,
            }
        )
        _write_training_plot_png(
            plot_points,
            cfg.output_model_dir / "training_plot_live.png",
            title=f"Training Progress | model={model_index}",
        )
        print(
            "[train-epoch] "
            f"model={model_index} stage=classifier epoch={epoch + 1} "
            f"train_loss={float(train_metrics.get('loss', 0.0)):.6f} "
            f"val_loss={float(val_metrics.get('loss', 0.0)):.6f} "
            f"lr={lr:.8f}",
            flush=True,
        )
        if score > best_key:
            best_key = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = len(history)
        if val_loss + float(cfg.early_stopping_min_delta) < head_best_val_loss:
            head_best_val_loss = val_loss
            head_no_improve = 0
        else:
            head_no_improve += 1
            if int(cfg.early_stopping_patience) > 0 and head_no_improve >= int(cfg.early_stopping_patience):
                print(
                    "[train-early-stop] "
                    f"model={model_index} stage=classifier epoch={epoch + 1} "
                    f"patience={int(cfg.early_stopping_patience)}",
                    flush=True,
                )
                break

    # Stage B: fine-tune the full model.
    _set_trainable_stage(model, stage="finetune")
    optim_ft = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(cfg.lr_finetune))
    ft_best_val_loss = float("inf")
    ft_no_improve = 0
    for epoch in range(int(max(0, cfg.epochs_finetune))):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optim_ft,
            aug_hflip=cfg.aug_hflip,
            aug_vflip=cfg.aug_vflip,
            aug_rotate=cfg.aug_rotate,
            aug_crop=cfg.aug_crop,
        )
        val_metrics = _run_epoch(model, val_loader, device=device, optimizer=None)
        lr = float(optim_ft.param_groups[0].get("lr", cfg.lr_finetune))
        score = float(val_metrics.get("weighted_kappa", 0.0))
        val_loss = float(val_metrics.get("loss", 0.0))
        plot_x += 1
        history.append(
            {
                "stage": "finetune",
                "stage_label": "full-model",
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        plot_points.append(
            {
                "x": plot_x,
                "stage": "full-model",
                "model_index": int(model_index),
                "train_loss": float(train_metrics.get("loss", 0.0)),
                "val_loss": float(val_metrics.get("loss", 0.0)),
                "lr": lr,
            }
        )
        _write_training_plot_png(
            plot_points,
            cfg.output_model_dir / "training_plot_live.png",
            title=f"Training Progress | model={model_index}",
        )
        print(
            "[train-epoch] "
            f"model={model_index} stage=full-model epoch={epoch + 1} "
            f"train_loss={float(train_metrics.get('loss', 0.0)):.6f} "
            f"val_loss={float(val_metrics.get('loss', 0.0)):.6f} "
            f"lr={lr:.8f}",
            flush=True,
        )
        if score > best_key:
            best_key = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = len(history)
        if val_loss + float(cfg.early_stopping_min_delta) < ft_best_val_loss:
            ft_best_val_loss = val_loss
            ft_no_improve = 0
        else:
            ft_no_improve += 1
            if int(cfg.early_stopping_patience) > 0 and ft_no_improve >= int(cfg.early_stopping_patience):
                print(
                    "[train-early-stop] "
                    f"model={model_index} stage=full-model epoch={epoch + 1} "
                    f"patience={int(cfg.early_stopping_patience)}",
                    flush=True,
                )
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    out_path = cfg.output_model_dir / f"DNN_{model_index}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "meta": {
                "trained_at": utc_now_iso(),
                "init_checkpoint": str(checkpoint_path.name),
                "best_val_weighted_kappa": float(best_key if np.isfinite(best_key) else 0.0),
                "best_epoch": int(best_epoch),
            },
        },
        str(out_path),
    )

    _write_training_plot_png(
        plot_points,
        cfg.output_model_dir / f"training_plot_model_{model_index}.png",
        title=f"Training Curve | model={model_index}",
    )

    return out_path, {
        "init_checkpoint": checkpoint_path.name,
        "output_checkpoint": out_path.name,
        "best_val_weighted_kappa": float(best_key if np.isfinite(best_key) else 0.0),
        "best_epoch": int(best_epoch),
        "history": history,
        "plot_points": plot_points,
    }


def run_transfer_learning(cfg: TrainConfig) -> dict[str, Any]:
    torch, DataLoader, _ = _require_torch()
    torch.manual_seed(int(cfg.seed))
    cfg.output_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] live plot: {cfg.output_model_dir / 'training_plot_live.png'}", flush=True)
    _write_training_plot_png([], cfg.output_model_dir / "training_plot_live.png", title="Training Progress | initializing")

    rows = _load_manifest_rows(cfg.manifest_path)
    train_rows, val_rows, test_rows = _split_rows(rows)
    manifest_parent = cfg.manifest_path.resolve().parent

    train_dataset_wrap = _SliceManifestDataset(
        train_rows,
        manifest_parent=manifest_parent,
        scaling=cfg.scaling,
        max_cache_scans=cfg.max_cache_scans,
    )
    val_dataset_wrap = _SliceManifestDataset(
        val_rows,
        manifest_parent=manifest_parent,
        scaling=cfg.scaling,
        max_cache_scans=cfg.max_cache_scans,
    )
    test_dataset_wrap = _SliceManifestDataset(
        test_rows,
        manifest_parent=manifest_parent,
        scaling=cfg.scaling,
        max_cache_scans=cfg.max_cache_scans,
    )
    train_dataset = train_dataset_wrap.to_torch_dataset()
    val_dataset = val_dataset_wrap.to_torch_dataset()
    test_dataset = test_dataset_wrap.to_torch_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(max(1, cfg.batch_size)),
        shuffle=True,
        num_workers=int(max(0, cfg.num_workers)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(max(1, cfg.batch_size)),
        shuffle=False,
        num_workers=int(max(0, cfg.num_workers)),
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(max(1, cfg.batch_size)),
        shuffle=False,
        num_workers=int(max(0, cfg.num_workers)),
        pin_memory=False,
    )

    init_paths = sorted(cfg.init_model_dir.glob("DNN_*.pt"))
    if not init_paths:
        raise FileNotFoundError(f"No DNN_*.pt checkpoints found in init_model_dir={cfg.init_model_dir}")

    device = _resolve_torch_device(cfg.device)
    print(
        "[train] "
        f"device={device} split(train/val/test)={len(train_rows)}/{len(val_rows)}/{len(test_rows)} "
        f"cache_scans={'unlimited' if cfg.max_cache_scans <= 0 else int(cfg.max_cache_scans)} "
        f"cache_rows(train/val/test)="
        f"{train_dataset_wrap.rows_with_cache}/{val_dataset_wrap.rows_with_cache}/{test_dataset_wrap.rows_with_cache}",
        flush=True,
    )
    per_model: list[dict[str, Any]] = []
    combined_plot_points: list[dict[str, Any]] = []
    combined_x = 0
    for model_index, init_path in enumerate(init_paths):
        out_path, train_report = _train_one_model(
            checkpoint_path=init_path,
            model_index=model_index,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        model = build_torch_model()
        model_state = _checkpoint_state_dict(out_path, map_location="cpu")
        model.load_state_dict(model_state, strict=True)
        model.to(device)
        test_metrics = _run_epoch(model, test_loader, device=device, optimizer=None)
        train_report["test"] = test_metrics
        local_points = list(train_report.get("plot_points", []))
        for row in local_points:
            combined_x += 1
            combined_plot_points.append(
                {
                    "x": combined_x,
                    "stage": row.get("stage", ""),
                    "model_index": int(row.get("model_index", model_index)),
                    "train_loss": float(row.get("train_loss", 0.0)),
                    "val_loss": float(row.get("val_loss", 0.0)),
                    "lr": float(row.get("lr", 0.0)),
                }
            )
        _write_training_plot_png(
            combined_plot_points,
            cfg.output_model_dir / "training_plot_live.png",
            title="Training Progress | all models",
        )
        per_model.append(train_report)

    summary = {
        "manifest_path": str(cfg.manifest_path.resolve()),
        "init_model_dir": str(cfg.init_model_dir.resolve()),
        "output_model_dir": str(cfg.output_model_dir.resolve()),
        "device": device,
        "seed": int(cfg.seed),
        "batch_size": int(cfg.batch_size),
        "epochs_head": int(cfg.epochs_head),
        "epochs_finetune": int(cfg.epochs_finetune),
        "lr_head": float(cfg.lr_head),
        "lr_finetune": float(cfg.lr_finetune),
        "early_stopping_patience": int(cfg.early_stopping_patience),
        "early_stopping_min_delta": float(cfg.early_stopping_min_delta),
        "aug_hflip": bool(cfg.aug_hflip),
        "aug_vflip": bool(cfg.aug_vflip),
        "aug_rotate": bool(cfg.aug_rotate),
        "aug_crop": bool(cfg.aug_crop),
        "n_rows": len(rows),
        "split_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "label_counts": {
            str(k): int(v)
            for k, v in sorted(Counter([_as_int(r.get("label"), default=0) for r in rows]).items())
        },
        "training_plot_live_path": str((cfg.output_model_dir / "training_plot_live.png").resolve()),
        "training_plot_path": str((cfg.output_model_dir / "training_plot.png").resolve()),
        "models": per_model,
        "trained_at": utc_now_iso(),
    }

    metrics_path = cfg.output_model_dir / "training_metrics.json"
    write_json(metrics_path, summary)
    _write_training_plot_png(
        combined_plot_points,
        cfg.output_model_dir / "training_plot.png",
        title="Training Progress | final",
    )
    print(f"[train] final plot: {cfg.output_model_dir / 'training_plot.png'}", flush=True)
    return summary
