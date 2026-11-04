"""Build compact per-track appearance archives without retaining heavy models."""

import logging
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.features.color import extract_color
from app.features.embedder import DinoEmbedder, DummyEmbedder, Embedder
from app.schemas import Track, TrackRepresentation

LOGGER = logging.getLogger(__name__)


def build_representation(
    track: Track, run_dir: Path, embedder: Embedder, config: AppConfig
) -> TrackRepresentation:
    """Embed full/vehicle/available rider crops, with mean colour features."""
    if not track.crops:
        raise ValueError("Track representation requires selected crops")
    full = embedder.embed([run_dir / c.path_full for c in track.crops])
    vehicle = embedder.embed([run_dir / c.path_vehicle for c in track.crops])
    rider_paths = [run_dir / c.path_rider for c in track.crops if c.path_rider is not None]
    rider = embedder.embed(rider_paths) if config.embedding.parts and rider_paths else None
    average = full.mean(axis=0)
    norm = np.linalg.norm(average)
    average = average / norm if norm else average
    histograms: list[list[float]] = []
    labs: list[list[float]] = []
    for crop in track.crops:
        image = cv2.imread(str(run_dir / crop.path_vehicle))
        if image is None:
            raise ValueError(f"Cannot decode {crop.path_vehicle}")
        hist, lab = extract_color(image, config.color)
        histograms.append(hist)
        labs.append(lab)
    return TrackRepresentation(
        track_id=track.track_id,
        emb_full=full.tolist(),
        emb_full_mean=average.tolist(),
        emb_vehicle=vehicle.tolist(),
        emb_rider=rider.tolist() if rider is not None else None,
        color_hist=np.mean(histograms, axis=0).tolist(),
        lab_mean=np.mean(labs, axis=0).tolist(),
    )


def save_representation(representation: TrackRepresentation, destination: Path) -> None:
    """Write numeric NPZ arrays; absence of rider features is an empty matrix."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = representation.model_dump()
    rider = values.pop("emb_rider")
    values["emb_rider"] = (
        rider
        if rider is not None
        else np.empty((0, len(representation.emb_full_mean)), dtype=np.float32)
    )
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **values)


def load_representation(path: Path) -> TrackRepresentation:
    """Load an archive with pickle explicitly disabled."""
    with np.load(path, allow_pickle=False) as arrays:
        values = {key: arrays[key].tolist() for key in arrays.files}
    values["emb_rider"] = values["emb_rider"] or None
    return TrackRepresentation.model_validate(values)


def represent_tracks(
    tracks: list[Track], run_dir: Path, config: AppConfig, models_dir: Path
) -> dict[str, int | str]:
    """Process tracks sequentially and release the selected embedder afterward."""
    embedder: Embedder = (
        DummyEmbedder(config.embedding, config.color)
        if (config.embedding.backend == "dummy")
        else DinoEmbedder(
            models_dir / "dinov2-small", config.embedding.batch_size, config.detector.device
        )
    )
    peak = 0
    try:
        for track in tracks:
            representation = build_representation(track, run_dir, embedder, config)
            save_representation(
                representation, run_dir / "representations" / f"{track.track_id}.npz"
            )
        if isinstance(embedder, DinoEmbedder):
            peak = embedder.peak_memory_bytes()
        LOGGER.info("run_id=%s represented=%s peak_cuda_bytes=%s", run_dir.name, len(tracks), peak)
    finally:
        embedder.close()
    return {"backend": config.embedding.backend, "tracks": len(tracks), "peak_cuda_bytes": peak}
