"""Shared JSON contracts. Coordinates are normalized outside detector internals."""

from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from app.config import Probability, StrictModel

Camera = Literal["A", "B"]
NonnegativeInt = Annotated[int, Field(ge=0)]
XY = tuple[Probability, Probability]


class BBox(StrictModel):
    x1: Probability
    y1: Probability
    x2: Probability
    y2: Probability

    @model_validator(mode="after")
    def positive_area(self) -> Self:
        """Reject reversed or zero-area rectangles."""
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Bounding box must have positive width and height")
        return self


class Detection(StrictModel):
    frame_idx: NonnegativeInt
    t_s: float
    bbox: BBox
    conf: Probability
    cls: NonnegativeInt
    track_id: NonnegativeInt


class CropInfo(StrictModel):
    frame_idx: NonnegativeInt
    t_s: float
    bbox: BBox
    quality: Probability
    sharpness: Annotated[float, Field(ge=0)]
    truncated: bool
    occluded: bool
    path_full: str
    path_rider: str | None = None
    path_vehicle: str


class FrameBox(StrictModel):
    frame_idx: NonnegativeInt
    bbox: BBox


class Track(StrictModel):
    track_id: str
    camera: Camera
    raw_track_ids: list[NonnegativeInt]
    first_frame: NonnegativeInt
    last_frame: NonnegativeInt
    t_start_s: float
    t_end_s: float
    boxes: list[FrameBox]
    n_frames: Annotated[int, Field(gt=0)]
    mean_conf: Probability
    median_box_h_px: Annotated[float, Field(gt=0)]
    entry_point: XY
    exit_point: XY
    motion_vector: tuple[float, float]
    quality: Probability
    quality_breakdown: dict[str, float]
    crops: list[CropInfo]
    has_rider: bool

    @model_validator(mode="after")
    def ordered_track(self) -> Self:
        """Validate bounds and a unit (or stationary zero) motion vector."""
        if self.first_frame > self.last_frame or self.t_start_s > self.t_end_s:
            raise ValueError("Track bounds are reversed")
        length_sq = sum(v * v for v in self.motion_vector)
        if length_sq != 0 and abs(length_sq - 1) > 1e-6:
            raise ValueError("Motion vector must be unit length or zero")
        return self


class TrackRepresentation(StrictModel):
    track_id: str
    emb_full: list[list[float]]
    emb_full_mean: list[float]
    emb_rider: list[list[float]] | None
    emb_vehicle: list[list[float]]
    color_hist: list[float]
    lab_mean: tuple[float, float, float]


class EvidenceItem(StrictModel):
    check: str
    polarity: Literal["supports", "contradicts", "neutral", "missing"]
    severity: Literal["low", "medium", "high", "critical"]
    value: float | str | None
    threshold: float | str | None
    message: str


class Verification(StrictModel):
    status: Literal["accepted", "uncertain", "rejected"]
    confidence: Probability
    features: dict[str, float]
    evidence: list[EvidenceItem]
    summary: str
    keypoint_viz_path: str | None = None


class CandidateScores(StrictModel):
    visual: Probability
    color: Probability
    temporal: Probability
    direction: Probability
    combined: Probability


class Candidate(StrictModel):
    pair_id: str
    query_track_id: str
    candidate_track_id: str
    rank: Annotated[int, Field(gt=0)]
    scores: CandidateScores
    delta_t_s: float
    delta_t_adjusted_s: float
    ambiguous: bool
    competing_pair_ids: list[str]
    assigned: bool
    verification: Verification | None = None


class QueryResult(StrictModel):
    query_track_id: str
    outcome: Literal["matched", "uncertain", "no_reliable_match"]
    best_pair_id: str | None
    candidates: list[Candidate]


class RunStatus(StrictModel):
    run_id: str
    state: Literal["queued", "running", "done", "failed"]
    stage: str
    progress: Probability
    error: str | None = None
    timings_s: dict[str, Annotated[float, Field(ge=0)]] = Field(default_factory=dict)
    fps_processed: dict[str, Annotated[float, Field(ge=0)]] = Field(default_factory=dict)


class VideoMeta(StrictModel):
    path: str
    fps: Annotated[float, Field(gt=0)]
    frame_count: NonnegativeInt
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    duration_s: Annotated[float, Field(ge=0)]
    start_time_s: float


class ClockOffset(StrictModel):
    mode: Literal["estimate", "fixed", "none"]
    estimated_s: float
    n_pairs_used: NonnegativeInt
    raw_delta_mode_s: float | None = None
    reliable: bool


class RunResult(StrictModel):
    run_id: str
    created_at: datetime
    config_snapshot: dict[str, Any]
    videos: dict[Camera, VideoMeta]
    clock_offset: ClockOffset
    tracks_A: list[Track]
    tracks_B: list[Track]
    queries: list[QueryResult]
    metrics: dict[str, Any] | None
    timings: dict[str, Annotated[float, Field(ge=0)]]
