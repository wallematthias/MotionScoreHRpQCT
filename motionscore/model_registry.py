from __future__ import annotations

from pathlib import Path
from typing import Any

from motionscore.utils import read_json, utc_now_iso, write_json

MODEL_REGISTRY_FILENAME = "model_registry.json"


def _implicit_root_profile(model_root: Path) -> dict[str, Any] | None:
    checkpoints = sorted(model_root.glob("DNN_*.pt"))
    if not checkpoints:
        return None
    return {
        "model_id": "base-v1",
        "display_name": "Base v1",
        "domain": "hrpqct",
        "version": "v1",
        "description": "Implicit default profile from checkpoints at model root.",
        "relative_dir": ".",
        "checkpoint_count": len(checkpoints),
        "source_model_id": "",
        "training_manifest": "",
        "metrics_path": "",
        "registered_at": "",
    }


def get_registry_path(model_root: str | Path) -> Path:
    return Path(model_root).resolve() / MODEL_REGISTRY_FILENAME


def _validate_registry_payload(payload: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid registry format in {registry_path}: expected object")

    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"Invalid registry format in {registry_path}: models must be a list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in models:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid registry format in {registry_path}: model entry must be object")
        model_id = str(raw.get("model_id", "")).strip()
        rel_dir = str(raw.get("relative_dir", "")).strip()
        if not model_id:
            raise ValueError(f"Invalid registry format in {registry_path}: model_id is required")
        if model_id in seen:
            raise ValueError(f"Invalid registry format in {registry_path}: duplicate model_id '{model_id}'")
        if not rel_dir:
            raise ValueError(
                f"Invalid registry format in {registry_path}: relative_dir is required for '{model_id}'"
            )
        seen.add(model_id)
        entry = dict(raw)
        entry["model_id"] = model_id
        entry["relative_dir"] = rel_dir
        normalized.append(entry)

    default_model_id = str(payload.get("default_model_id", "")).strip()
    if not default_model_id and normalized:
        default_model_id = str(normalized[0].get("model_id", "")).strip()

    if default_model_id and default_model_id not in seen:
        raise ValueError(
            f"Invalid registry format in {registry_path}: default_model_id '{default_model_id}' not found"
        )

    return {
        "schema_version": str(payload.get("schema_version", "1.0") or "1.0"),
        "updated_at": str(payload.get("updated_at", "") or ""),
        "default_model_id": default_model_id,
        "models": normalized,
    }


def load_model_registry(model_root: str | Path) -> dict[str, Any]:
    root = Path(model_root).resolve()
    registry_path = get_registry_path(root)
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found: {registry_path}. "
            "Register a model first with `motionscore model-register ...`."
        )

    payload = read_json(registry_path)
    return _validate_registry_payload(payload, registry_path=registry_path)


def list_model_profiles(model_root: str | Path) -> list[dict[str, Any]]:
    root = Path(model_root).resolve()
    registry_path = get_registry_path(root)
    if not registry_path.exists():
        implicit = _implicit_root_profile(root)
        return [implicit] if implicit is not None else []

    payload = load_model_registry(root)
    return [dict(entry) for entry in payload.get("models", [])]


def _resolve_model_profile(model_root: Path, model_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_model_registry(model_root)
    requested = str(model_id or payload.get("default_model_id", "")).strip()
    if not requested:
        raise ValueError(
            f"Registry {get_registry_path(model_root)} has no default_model_id and no model_id was provided"
        )

    for entry in payload.get("models", []):
        if str(entry.get("model_id", "")).strip() == requested:
            return dict(entry), payload
    raise KeyError(f"Unknown model_id '{requested}' in {get_registry_path(model_root)}")


def resolve_model_dir(model_root: str | Path, model_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    root = Path(model_root).resolve()
    registry_path = get_registry_path(root)
    if not registry_path.exists():
        implicit = _implicit_root_profile(root)
        requested = str(model_id or "base-v1").strip() or "base-v1"
        if implicit is None:
            raise FileNotFoundError(
                f"Model registry not found: {registry_path}. "
                "Register a model first with `motionscore model-register ...`, "
                f"or place DNN_*.pt checkpoints directly in {root}."
            )
        if requested != str(implicit.get("model_id", "")).strip():
            raise KeyError(f"Unknown model_id '{requested}' in implicit root profile for {root}")
        return root, dict(implicit)

    profile, payload = _resolve_model_profile(root, model_id=model_id)

    rel_dir = Path(str(profile.get("relative_dir", "")).strip())
    model_dir = (root / rel_dir).resolve()
    if root not in model_dir.parents and model_dir != root:
        raise ValueError(f"Invalid relative_dir for model_id '{profile.get('model_id')}': {rel_dir}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory for model_id '{profile.get('model_id')}' does not exist: {model_dir}"
        )

    dnn_models = sorted(model_dir.glob("DNN_*.pt"))
    if not dnn_models:
        raise FileNotFoundError(
            f"No DNN_*.pt checkpoints found for model_id '{profile.get('model_id')}' in {model_dir}"
        )

    merged_profile = dict(profile)
    merged_profile["default_model_id"] = payload.get("default_model_id", "")
    return model_dir, merged_profile


def register_model_profile(
    model_root: str | Path,
    model_id: str,
    model_dir: str | Path,
    *,
    display_name: str,
    domain: str,
    version: str,
    description: str = "",
    source_model_id: str = "",
    training_manifest: str = "",
    metrics_path: str = "",
    make_default: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = Path(model_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_id_norm = str(model_id).strip()
    if not model_id_norm:
        raise ValueError("model_id must not be empty")

    model_dir_resolved = Path(model_dir).resolve()
    if root not in model_dir_resolved.parents and model_dir_resolved != root:
        raise ValueError(f"model_dir must be inside model_root. model_root={root} model_dir={model_dir_resolved}")

    dnn_models = sorted(model_dir_resolved.glob("DNN_*.pt"))
    if not dnn_models:
        raise FileNotFoundError(f"No DNN_*.pt checkpoints found in {model_dir_resolved}")

    registry_path = get_registry_path(root)
    if registry_path.exists():
        payload = load_model_registry(root)
    else:
        payload = {
            "schema_version": "1.0",
            "default_model_id": "",
            "models": [],
        }

    rel_dir = model_dir_resolved.relative_to(root)

    new_entry = {
        "model_id": model_id_norm,
        "display_name": str(display_name).strip() or model_id_norm,
        "domain": str(domain).strip() or "custom",
        "version": str(version).strip() or "v1",
        "description": str(description).strip(),
        "relative_dir": str(rel_dir) if str(rel_dir) else ".",
        "checkpoint_count": len(dnn_models),
        "source_model_id": str(source_model_id).strip(),
        "training_manifest": str(training_manifest).strip(),
        "metrics_path": str(metrics_path).strip(),
        "registered_at": utc_now_iso(),
    }

    models = list(payload.get("models", []))
    replaced = False
    for idx, row in enumerate(models):
        if str(row.get("model_id", "")).strip() == model_id_norm:
            models[idx] = new_entry
            replaced = True
            break
    if not replaced:
        models.append(new_entry)

    default_model_id = str(payload.get("default_model_id", "")).strip()
    if make_default or not default_model_id:
        default_model_id = model_id_norm

    out = {
        "schema_version": "1.0",
        "updated_at": utc_now_iso(),
        "default_model_id": default_model_id,
        "models": sorted(models, key=lambda x: str(x.get("model_id", "")).casefold()),
    }
    write_json(registry_path, out)
    return registry_path, new_entry
