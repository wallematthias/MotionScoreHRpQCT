from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RawSession:
    subject_id: str
    site: str
    session_id: str
    raw_image_path: Path
    stack_index: int | None = None
    output_rel_dir: Path | None = None
    source_session_id: str | None = None
    raw_mask_paths: dict[str, Path] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("subject_id must not be empty")
        if not self.site:
            raise ValueError("site must not be empty")
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.raw_image_path:
            raise ValueError("raw_image_path must be provided")
        if self.stack_index is not None and self.stack_index < 1:
            raise ValueError("stack_index must be >= 1")
