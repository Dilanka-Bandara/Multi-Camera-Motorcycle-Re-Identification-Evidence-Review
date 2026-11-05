import ast
import json
import os
from pathlib import Path

import pytest

from app.config import load_config
from app.schemas import EvidenceItem
from app.verify.policy import decide


@pytest.mark.parametrize(
    ("p", "severity", "ambiguous", "want"),
    [
        (0.99, "critical", False, "rejected"),
        (0.99, "high", True, "uncertain"),
        (0.99, None, False, "accepted"),
        (0.1, None, False, "rejected"),
    ],
)
def test_policy(p: float, severity: str | None, ambiguous: bool, want: str) -> None:
    items = (
        [
            EvidenceItem(
                check="test",
                polarity="contradicts",
                severity=severity,
                value=None,
                threshold=None,
                message="example",
            )
        ]
        if severity
        else []
    )
    assert decide(p, items, ambiguous, 1, load_config().verifier) == want


def test_verifier_imports_are_independent() -> None:
    for path in Path("backend/app/verify").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            assert all("matching.candidates" not in name for name in modules)


def test_fixture_review_evidence() -> None:
    root = Path(os.environ.get("MOTO_PHASE5_RUN", "data/user/runs/phase5-validation"))
    if not (root / "queries.json").is_file():
        pytest.skip("Run the full dev fixture to evaluate independent review")
    labels = {
        t["track_id"]: t["raw_track_ids"][0]
        for camera in ("A", "B")
        for t in json.loads((root / f"tracks_{camera}.json").read_text())
    }
    shared_reviewed = 0
    for query in json.loads((root / "queries.json").read_text()):
        source = labels[query["query_track_id"]]
        if source in (9, 10, 11):
            assert query["outcome"] in ("uncertain", "no_reliable_match")
        for c in query["candidates"]:
            v = c["verification"]
            if c["rank"] > load_config("dev_fixture").verifier.review_top_n:
                assert v is None
                continue
            assert v["evidence"] and (root / v["keypoint_viz_path"]).is_file()
            target = labels[c["candidate_track_id"]]
            if source == target and v["status"] in ("accepted", "uncertain"):
                shared_reviewed += 1
            if {source, target} in ({0, 1}, {2, 3}):
                assert v["status"] != "accepted"
    assert shared_reviewed >= 5
