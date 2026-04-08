import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.getenv("MOTIONSCORE_EQUIVALENCE_JSON"),
    reason="Set MOTIONSCORE_EQUIVALENCE_JSON to enable strict legacy/new equivalence gate.",
)
def test_strict_equivalence_gate() -> None:
    payload_path = Path(os.environ["MOTIONSCORE_EQUIVALENCE_JSON"])
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    tol = float(os.getenv("MOTIONSCORE_EQUIVALENCE_TOL", "1e-6"))

    # Expected payload shape:
    # {
    #   "pairs": [
    #     {"scan_id": "...", "legacy_grade": 3, "new_grade": 3,
    #      "legacy_conf": 85.0, "new_conf": 85.0}
    #   ]
    # }
    pairs = payload.get("pairs", [])
    assert pairs, "No equivalence pairs provided"

    for pair in pairs:
        assert int(pair["legacy_grade"]) == int(pair["new_grade"]), pair
        conf_diff = abs(float(pair["legacy_conf"]) - float(pair["new_conf"]))
        assert conf_diff <= tol, f"{pair['scan_id']}: confidence diff {conf_diff} > {tol}"
