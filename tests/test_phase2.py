"""Phase 2 geometry, stitching, quality and real fixture integration checks."""

import ast
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import load_config
from app.pipeline.crops import Observation, select_crops
from app.pipeline.detect_track import detect_track, process_stride, result_detections
from app.pipeline.geometry import iou
from app.pipeline.ingest import ingest_video, probe_video
from app.pipeline.motion import summarize_motion
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.rider_assoc import AssociatedDetection, associate_riders
from app.pipeline.stitching import Fragment, stitch_fragments
from app.pipeline.track_quality import score_track
from app.schemas import BBox, Detection, Track, VideoMeta
from scripts.make_dev_fixture import generate_fixture


def detection(frame: int, raw_id: int = 1, x: float = 0.2, cls: int = 3) -> Detection:
    """Return a normalized synthetic observation."""
    return Detection(
        frame_idx=frame,
        t_s=frame / 15,
        track_id=raw_id,
        cls=cls,
        conf=0.9,
        bbox=BBox(x1=x, y1=0.3, x2=x + 0.2, y2=0.6),
    )


def test_rider_selection_and_missing_parts() -> None:
    cfg = load_config()
    vehicle = detection(0)
    rider = detection(0, 10, cls=0).model_copy(
        update={"bbox": BBox(x1=0.24, y1=0.1, x2=0.36, y2=0.5)}
    )
    unrelated = detection(0, 11, x=0.7, cls=0)
    result = associate_riders([vehicle], [unrelated, rider], cfg.rider_association)[0]
    assert result.rider_box == rider.bbox
    assert result.union_box.y1 == 0.1
    assert result.union_box.y2 == 0.6
    assert associate_riders([vehicle], [], cfg.rider_association)[0].rider_box is None
    assert (
        associate_riders([vehicle.model_copy(update={"cls": 2})], [rider], cfg.rider_association)[
            0
        ].rider_box
        is None
    )


def test_stitching_gates_and_chain_without_mutation() -> None:
    cfg = load_config().tracker.stitching
    fragments = [Fragment([i], [detection(i * 3, i, x=0.2 + i * 0.01)], [1, 0]) for i in range(3)]
    stitched = stitch_fragments(fragments[::-1], cfg)
    assert len(stitched) == 1
    assert stitched[0].raw_ids == [0, 1, 2]
    assert fragments[0].raw_ids == [0]
    for other in [
        Fragment([4], [detection(0, 4)], [1, 0]),
        Fragment([4], [detection(100, 4)], [1, 0]),
        Fragment([4], [detection(3, 4, x=0.7)], [1, 0]),
        Fragment([4], [detection(3, 4)], [0, 1]),
    ]:
        assert len(stitch_fragments([fragments[0], other], cfg)) == 2


def test_motion_and_quality() -> None:
    cfg = load_config()
    boxes = [detection(i, x=0.1 + i * 0.02).bbox for i in range(8)]
    entry, exit_point, vector = summarize_motion(boxes, cfg.motion)
    assert exit_point[0] > entry[0]
    assert vector == pytest.approx((1, 0))
    assert summarize_motion([boxes[0]] * 5, cfg.motion)[2] == (0, 0)
    score, breakdown = score_track(
        n_frames=30,
        mean_conf=1,
        mean_crop_quality=1,
        truncated_fraction=0,
        occluded_fraction=0,
        median_height_px=100,
        config=cfg.track_quality,
    )
    assert score == pytest.approx(1)
    assert set(breakdown) == set(cfg.track_quality.weights)


def test_crop_selection_enforces_diversity_and_limit() -> None:
    cfg = load_config().crops.model_copy(update={"top_k": 2, "min_frame_gap": 3})
    observations = []
    for frame, quality in ((0, 0.8), (1, 1), (4, 0.7), (8, 0.5)):
        d = detection(frame)
        observations.append(
            Observation(
                AssociatedDetection(d, d.bbox, None, d.bbox), quality, 100, False, False, [1]
            )
        )
    assert [o.associated.detection.frame_idx for o in select_crops(observations, 360, cfg)] == [
        1,
        4,
    ]
    assert select_crops(observations, 10, cfg) == []


def test_untracked_results_and_stride() -> None:
    meta = VideoMeta(
        path="x", fps=30, frame_count=60, width=640, height=360, duration_s=2, start_time_s=4
    )
    assert result_detections(SimpleNamespace(boxes=None), 0, meta) == []
    assert result_detections(SimpleNamespace(boxes=SimpleNamespace(id=None)), 0, meta) == []
    assert [process_stride(fps, 15) for fps in (10, 15, 20, 30, 60)] == [1, 1, 2, 2, 4]


def test_pure_modules_have_no_heavy_imports() -> None:
    for name in ("geometry", "rider_assoc", "motion", "stitching", "track_quality"):
        path = Path("backend/app/pipeline") / f"{name}.py"
        tree = ast.parse(path.read_text())
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports += [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert all(not name.startswith(("torch", "cv2")) for name in imports)


def test_fixture_pipeline_recovers_fragmented_identities(tmp_path: Path) -> None:
    cfg = load_config(
        "dev_fixture",
        {"fixture": {"width": 320, "height": 180}, "groundtruth": {"fragment_prob": 0.2}},
    )
    fixture = tmp_path / "fixture"
    generate_fixture(fixture, cfg)
    videos = {c: fixture / f"camera_{c}.mp4" for c in ("A", "B")}
    truth = {c: fixture / f"gt_{c}.csv" for c in ("A", "B")}
    out = tmp_path / "run"
    metadata = run_pipeline(cfg, videos, out, truth)
    assert metadata["state"] == "done"
    merged_count = 0
    for camera, identity_count in (("A", 12), ("B", 11)):
        tracks = [
            Track.model_validate(t) for t in json.loads((out / f"tracks_{camera}.json").read_text())
        ]
        with truth[camera].open() as handle:
            gt = list(csv.DictReader(handle))
        votes: dict[int, list[str]] = {}
        for track in tracks:
            counts: dict[int, int] = {}
            for box in track.boxes:
                for row in gt:
                    if int(row["frame_idx"]) != box.frame_idx:
                        continue
                    target = BBox(**{key: float(row[key]) for key in ("x1", "y1", "x2", "y2")})
                    if iou(box.bbox, target) >= 0.5:
                        identity = int(row["identity_id"])
                        counts[identity] = counts.get(identity, 0) + 1
            winner = max(counts, key=counts.get)
            assert counts[winner] / track.n_frames >= 0.5
            votes.setdefault(winner, []).append(track.track_id)
            merged_count += len(track.raw_track_ids) > 1
            assert 0 <= track.quality <= 1
            assert 0 < len(track.crops) <= cfg.crops.top_k
            assert track.motion_vector == pytest.approx((1, 0), abs=0.01)
            for crop in track.crops:
                assert (out / crop.path_full).is_file()
                assert (out / crop.path_vehicle).is_file()
            frames = sorted(c.frame_idx for c in track.crops)
            assert all(
                b - a >= cfg.crops.min_frame_gap for a, b in zip(frames, frames[1:], strict=False)
            )
        recovered = sum(len(ids) == 1 for ids in votes.values()) / identity_count
        assert recovered >= 0.9
        assert probe_video(out / f"camera_{camera}.mp4", 0).frame_count == 900
    assert merged_count > 0
    from app.matching.similarity import set_similarity
    from app.pipeline.representation import load_representation

    camera_tracks = {
        camera: [
            Track.model_validate(t) for t in json.loads((out / f"tracks_{camera}.json").read_text())
        ]
        for camera in ("A", "B")
    }
    query = next(t for t in camera_tracks["A"] if 0 in t.raw_track_ids)
    same = next(t for t in camera_tracks["B"] if 0 in t.raw_track_ids)
    different = next(t for t in camera_tracks["B"] if 4 in t.raw_track_ids)
    reps = [
        load_representation(out / "representations" / f"{t.track_id}.npz")
        for t in (query, same, different)
    ]
    assert set_similarity(reps[0].emb_full, reps[1].emb_full, 5) > set_similarity(
        reps[0].emb_full, reps[2].emb_full, 5
    )
    with pytest.raises(FileExistsError):
        run_pipeline(cfg, videos, out, truth)


@pytest.mark.slow
def test_real_yolo_two_seconds(tmp_path: Path) -> None:
    weights = Path(os.environ.get("MOTO_TEST_WEIGHTS", "models/yolo26s.pt"))
    source = Path(os.environ.get("MOTO_TEST_VIDEO", "data/fixtures/dev/camera_A.mp4"))
    if not weights.is_file():
        pytest.skip("Local YOLO weights missing; no runtime downloads")
    if not source.is_file():
        pytest.skip("Set MOTO_TEST_VIDEO to an existing local clip")
    import subprocess

    clip = tmp_path / "two_seconds.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            "2",
            "-an",
            "-c:v",
            "libx264",
            str(clip),
        ],
        check=True,
    )
    cfg = load_config(overrides={"detector": {"weights": weights.name}})
    web = tmp_path / "web.mp4"
    meta = ingest_video(clip, web, 0, cfg.detector.max_process_height)
    results, provenance = detect_track(web, meta, cfg, weights.parent, "test-real-yolo")
    assert provenance["weights"] == weights.name
    assert all(0 <= d.frame_idx < meta.frame_count for d in results)
