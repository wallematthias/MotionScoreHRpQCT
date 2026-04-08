from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def to_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def make_scan_id(subject_id: str, site: str, session_id: str, raw_image_path: Path) -> str:
    key = f"{subject_id}|{site}|{session_id}|{raw_image_path.resolve()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"sub-{subject_id}_site-{site}_ses-{session_id}_{digest}"


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serial = {}
            for key in writer.fieldnames:
                value = row.get(key, "")
                if value is None:
                    serial[key] = ""
                elif isinstance(value, (dict, list)):
                    serial[key] = json.dumps(value, separators=(",", ":"))
                else:
                    serial[key] = str(value)
            writer.writerow(serial)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
