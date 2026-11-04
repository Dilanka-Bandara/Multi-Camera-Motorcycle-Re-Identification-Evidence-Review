"""Validate configuration boundaries, shared contracts and deterministic GT behavior."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import deep_merge, load_config
from app.pipeline.gt_detector import detect_groundtruth
from app.schemas import BBox, Candidate, RunResult, RunStatus
from scripts.make_dev_fixture import generate_fixture, make_schedule


def test_config_merge_isolation_and_precedence() -> None:
    original = {"a": {"b": 1, "c": [1, 2]}}
    result = deep_merge(original, {"a": {"c": [3]}})
    assert result == {"a": {"b": 1, "c": [3]}}
    result["a"]["c"].append(4)
    assert original["a"]["c"] == [1, 2]
    cfg = load_config("dev_fixture", {"topology": {"max_transit_s": 25}})
    assert cfg.topology.min_transit_s == 5
    assert cfg.topology.max_transit_s == 25
    assert cfg.detector.mode == "groundtruth"
    assert cfg.cameras.B.start_time_s == 4
    assert load_config().detector.mode == "yolo"
    assert cfg.model_validate_json(cfg.model_dump_json()) == cfg


@pytest.mark.parametrize("override", [
    {"detector": {"conf": 1.1}}, {"detector": {"conff": 0.5}},
    {"topology": {"min_transit_s": 80}}, {"verifier": {"p_reject": 0.9}},
    {"matching": {"weights": {"visual": 0}}}, {"runtime": {"seed": -1}},
    {"fixture": {"width": 641}}, {"groundtruth": {"drop_rate": -0.1}},
])
def test_invalid_config_rejected(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        load_config(overrides=override)


def test_preset_path_cannot_escape() -> None:
    with pytest.raises(ValueError, match="simple lowercase"):
        load_config("../default")


@pytest.mark.parametrize("bounds", [(-1, 0, 1, 1), (0, 0, 1.1, 1),
                                         (0.5, 0, 0.4, 1), (0, 0, 0, 1)])
def test_bbox_contract(bounds: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValidationError):
        BBox(**dict(zip(("x1", "y1", "x2", "y2"), bounds, strict=True)))


def test_nested_result_roundtrip() -> None:
    candidate = Candidate(
        pair_id="A-0001__B-0001", query_track_id="A-0001", candidate_track_id="B-0001",
        rank=1, scores=dict.fromkeys(["visual", "color", "temporal", "direction", "combined"], 0.8),
        delta_t_s=14, delta_t_adjusted_s=10, ambiguous=False,
        competing_pair_ids=[], assigned=True,
    )
    result = RunResult(
        run_id="test", created_at="2026-11-04T00:00:00Z", config_snapshot={}, videos={},
        clock_offset={"mode": "estimate", "estimated_s": 4, "n_pairs_used": 3, "reliable": True},
        tracks_A=[], tracks_B=[], queries=[{
            "query_track_id": "A-0001", "outcome": "uncertain", "best_pair_id": candidate.pair_id,
            "candidates": [candidate],
        }], metrics=None, timings={},
    )
    assert RunResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        RunStatus(run_id="x", state="done", stage="test", progress=1.1)


def write_gt(path: Path) -> None:
    """Write a sparse-ID tracker fixture without the optional cls column."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "identity_id", "x1", "y1", "x2", "y2"])
        for frame in range(20):
            for identity in (2, 100):
                writer.writerow([frame, identity, 0.1, 0.2, 0.3, 0.4])


def test_gt_detector_drop_fragment_and_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    write_gt(path)
    cfg = load_config()
    baseline = detect_groundtruth(path, fps=10, start_time_s=4, options=cfg.groundtruth, seed=42)
    assert len(baseline) == 40
    assert {d.track_id for d in baseline} == {2, 100}
    assert baseline[-1].t_s == 5.9
    assert all(d.cls == 3 for d in baseline)
    fragmented = cfg.groundtruth.model_copy(update={"fragment_prob": 1})
    split = detect_groundtruth(path, fps=10, start_time_s=4, options=fragmented, seed=42)
    assert len(split) == 36
    assert {d.track_id for d in split} == {2, 100, 101, 102}
    assert split == detect_groundtruth(path, fps=10, start_time_s=4, options=fragmented, seed=42)
    for old, new in ((2, 101), (100, 102)):
        old_end = max(d.frame_idx for d in split if d.track_id == old)
        new_start = min(d.frame_idx for d in split if d.track_id == new)
        assert new_start - old_end == cfg.groundtruth.fragment_gap_frames + 1
    dropped = cfg.groundtruth.model_copy(update={"drop_rate": 1})
    assert detect_groundtruth(path, fps=10, start_time_s=4, options=dropped, seed=42) == []
    partial = cfg.groundtruth.model_copy(update={"drop_rate": 0.5})
    assert 0 < len(detect_groundtruth(path, fps=10, start_time_s=4,
                                    options=partial, seed=42)) < 40


def test_fixture_schedule() -> None:
    cfg = load_config("dev_fixture")
    schedule = make_schedule(cfg)
    assert schedule == make_schedule(cfg)
    assert len(schedule) == 14
    assert sum(i["entry_a_s"] is not None for i in schedule) == 12
    assert sum(i["entry_b_s"] is not None for i in schedule) == 11
    for item in schedule[:9]:
        assert 8 <= item["entry_b_s"] - item["exit_a_s"] <= 15
        assert item["exit_b_s"] < cfg.fixture.duration_s
    for left, right in ((0, 1), (2, 3)):
        assert schedule[left]["body"] == schedule[right]["body"]
        assert schedule[left]["helmet"] == schedule[right]["helmet"]
        assert schedule[left]["torso"] != schedule[right]["torso"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required (available in Docker)")
def test_fixture_encoding_and_normalized_gt(tmp_path: Path) -> None:
    cfg = load_config("dev_fixture", {"fixture": {"width": 64, "height": 36, "fps": 1}})
    generate_fixture(tmp_path, cfg)
    for camera, expected in (("A", set(range(12))), ("B", set(range(9)) | {12, 13})):
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_streams", "-of", "json",
            str(tmp_path / f"camera_{camera}.mp4"),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        assert stream["codec_name"] == "h264"
        assert int(stream["nb_frames"]) == 60
        detections = detect_groundtruth(tmp_path / f"gt_{camera}.csv", fps=cfg.fixture.fps,
                                       start_time_s=0, options=cfg.groundtruth, seed=42)
        assert {d.track_id for d in detections} == expected
    meta = json.loads((tmp_path / "fixture_meta.json").read_text())
    assert meta["true_clock_offset_s"] == 4
    assert len(meta["true_pairs"]) == 9
    with pytest.raises(FileExistsError):
        generate_fixture(tmp_path, cfg)
