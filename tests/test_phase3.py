"""Appearance math, local feature archives and optional real DINO checks."""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import load_config
from app.features.color import extract_color
from app.features.embedder import DinoEmbedder, DummyEmbedder
from app.matching.similarity import color_similarity, set_similarity
from app.pipeline.representation import (
    build_representation,
    load_representation,
    save_representation,
)
from app.schemas import Track


def test_similarity_hand_computed_and_missing() -> None:
    assert set_similarity([[1, 0]], [[1, 0]], 5) == pytest.approx(1)
    assert set_similarity([[1, 0]], [[0, 1]], 5) == pytest.approx(0)
    assert set_similarity([[1, 0]], [[-1, 0]], 5) == 0
    assert set_similarity([[1, 0], [0, 1]], [[1, 0]], 1) == pytest.approx((1 + 2**-0.5) / 2)
    assert set_similarity(np.empty((0, 2)), [[1, 0]], 1) == 0
    assert set_similarity([[0, 0]], [[1, 0]], 1) == 0
    with pytest.raises(ValueError):
        set_similarity([[float("nan")]], [[1]], 1)
    with pytest.raises(ValueError):
        set_similarity([[1, 0]], [[1]], 1)
    assert color_similarity([2, 0], [1, 0]) == pytest.approx(1)
    assert color_similarity([1, 0], [0, 1]) == 0


def test_color_and_dummy_features(tmp_path: Path) -> None:
    cfg = load_config("dev_fixture")
    red = np.zeros((80, 70, 3), dtype=np.uint8)
    red[:, :, 2] = 240
    blue = red[:, :, ::-1].copy()
    paths = []
    for name, image in (("red", red), ("red_dim", (red * 0.9).astype(np.uint8)), ("blue", blue)):
        path = tmp_path / f"{name}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    hist, lab = extract_color(red, cfg.color)
    assert sum(hist) == pytest.approx(1)
    assert 0 < lab[0] <= 100
    embedder = DummyEmbedder(cfg.embedding, cfg.color)
    vectors = embedder.embed(paths)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1, 1, 1])
    assert np.array_equal(vectors, embedder.embed(paths))
    assert set_similarity(vectors[:1], vectors[1:2], 1) > set_similarity(
        vectors[:1], vectors[2:], 1
    )


def test_missing_dino_model_never_downloads(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="provisioned locally"):
        DinoEmbedder(tmp_path, 2)


def test_representation_roundtrip_from_fixture(tmp_path: Path) -> None:
    from app.schemas import BBox, CropInfo, FrameBox

    cfg = load_config("dev_fixture")
    box = BBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
    image = np.full((64, 64, 3), (20, 60, 180), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "crop.jpg"), image)
    crop = CropInfo(
        frame_idx=0,
        t_s=0,
        bbox=box,
        quality=1,
        sharpness=10,
        truncated=False,
        occluded=False,
        path_full="crop.jpg",
        path_vehicle="crop.jpg",
    )
    track = Track(
        track_id="A-0000",
        camera="A",
        raw_track_ids=[0],
        first_frame=0,
        last_frame=0,
        t_start_s=0,
        t_end_s=0,
        boxes=[FrameBox(frame_idx=0, bbox=box)],
        n_frames=1,
        mean_conf=1,
        median_box_h_px=64,
        entry_point=(0.3, 0.3),
        exit_point=(0.3, 0.3),
        motion_vector=(0, 0),
        quality=1,
        quality_breakdown={},
        crops=[crop],
        has_rider=False,
    )
    rep = build_representation(track, tmp_path, DummyEmbedder(cfg.embedding, cfg.color), cfg)
    assert rep.emb_rider is None
    destination = tmp_path / "representation.npz"
    save_representation(rep, destination)
    assert load_representation(destination) == rep
    with pytest.raises(FileExistsError):
        save_representation(rep, destination)


@pytest.mark.slow
def test_dino_real_fixture_similarity() -> None:
    model = Path(os.environ.get("MOTO_DINO_MODEL", "models/dinov2-small"))
    run = Path(os.environ.get("MOTO_TEST_RUN", "data/user/runs/phase2-validation"))
    if not (model / "config.json").is_file() or not (run / "tracks_A.json").is_file():
        pytest.skip("Local model and completed fixture run required")
    a = [Track.model_validate(v) for v in json.loads((run / "tracks_A.json").read_text())]
    b = [Track.model_validate(v) for v in json.loads((run / "tracks_B.json").read_text())]
    # Evaluate every shared identity, including lookalikes, rather than choosing
    # a favorable isolated pair. A pretrained model need not rank every sprite correctly.
    left = [next(t for t in a if identity in t.raw_track_ids) for identity in range(9)]
    right = [next(t for t in b if identity in t.raw_track_ids) for identity in range(9)]
    embedder = DinoEmbedder(model, 8, os.environ.get("MOTO_TEST_DEVICE", "cpu"))
    try:
        left_vectors = [embedder.embed([run / c.path_full for c in t.crops]) for t in left]
        right_vectors = [embedder.embed([run / c.path_full for c in t.crops]) for t in right]
        scores = np.array([[set_similarity(x, y, 5) for y in right_vectors] for x in left_vectors])
        same_mean = float(np.diag(scores).mean())
        different_mean = float(scores[~np.eye(len(left), dtype=bool)].mean())
        logging.info("run_id=%s DINO fixture same_mean=%s different_mean=%s", run.name,
                     same_mean, different_mean)
        assert same_mean > different_mean
        if embedder.device.startswith("cuda"):
            assert embedder.peak_memory_bytes() < 3 * 1024**3
    finally:
        embedder.close()
