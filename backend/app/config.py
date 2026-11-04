"""Validated configuration, with defaults supplied exclusively by YAML files."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]
Count = Annotated[int, Field(gt=0)]
Direction = tuple[float, float]


class StrictModel(BaseModel):
    """Reject misspelled fields and non-finite numeric values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TargetConfig(StrictModel):
    classes: list[Literal["motorcycle", "car", "bus", "truck"]] = Field(min_length=1)
    rider_class: Literal["person"]


class DetectorConfig(StrictModel):
    mode: Literal["yolo", "groundtruth"]
    weights: str
    fallback_weights: str
    imgsz: Count
    conf: Probability
    iou: Probability
    half: bool | Literal["auto"]
    device: str
    max_process_height: Count
    vid_stride: Count | Literal["auto"]


class StitchingConfig(StrictModel):
    enabled: bool
    max_gap_s: Positive
    max_center_jump: Probability
    min_appearance_sim: Probability


class TrackerConfig(StrictModel):
    type: Literal["botsort.yaml", "bytetrack.yaml"]
    min_track_frames: Count
    stitching: StitchingConfig


class CropWeights(StrictModel):
    size: Probability
    conf: Probability
    sharpness: Probability
    aspect: Probability


class CropsConfig(StrictModel):
    top_k: Count
    min_box_height_px: Count
    min_frame_gap: Count
    border_margin: Probability
    occlusion_iou: Probability
    weights: CropWeights


class EmbeddingConfig(StrictModel):
    backend: Literal["dinov2", "dummy"]
    model: str
    batch_size: Count
    parts: bool


class ColorConfig(StrictModel):
    hsv_bins: tuple[Count, Count, Count]


class CameraConfig(StrictModel):
    name: str
    start_time_s: float


class CamerasConfig(StrictModel):
    A: CameraConfig
    B: CameraConfig


class ClockConfig(StrictModel):
    mode: Literal["estimate", "fixed", "none"]
    fixed_s: float
    search_range_s: Positive
    min_confident_pairs: Count
    confident_visual_sim: Probability


class TopologyConfig(StrictModel):
    direction: Literal["A_to_B", "B_to_A", "both"]
    min_transit_s: Annotated[float, Field(ge=0)]
    max_transit_s: Positive
    expected_exit_dir_A: Direction | None
    expected_entry_dir_B: Direction | None
    clock_offset: ClockConfig

    @model_validator(mode="after")
    def ordered_window(self) -> Self:
        """Ensure transit bounds define a nonempty interval."""
        if self.min_transit_s >= self.max_transit_s:
            raise ValueError("min_transit_s must be less than max_transit_s")
        return self


class MatchWeights(StrictModel):
    visual: Probability
    color: Probability
    temporal: Probability
    direction: Probability


class MatchingConfig(StrictModel):
    top_n_candidates: Count
    set_sim_top_m: Count
    weights: MatchWeights
    temporal_soft_tau_s: Positive
    min_match_score: Probability
    ambiguity_margin: Probability


class KeypointConfig(StrictModel):
    backend: Literal["lightglue", "sift"]
    max_keypoints: Count
    pairs_to_check: Count
    min_inliers_pass: Count
    max_inliers_fail: Annotated[int, Field(ge=0)]


class VerifierConfig(StrictModel):
    keypoints: KeypointConfig
    part_disagreement_gap: Probability
    min_track_quality: Probability
    p_accept: Probability
    p_reject: Probability
    review_top_n: Count

    @model_validator(mode="after")
    def ordered_thresholds(self) -> Self:
        """Reject contradictory decision thresholds."""
        if self.p_reject >= self.p_accept:
            raise ValueError("p_reject must be less than p_accept")
        if self.keypoints.max_inliers_fail >= self.keypoints.min_inliers_pass:
            raise ValueError("keypoint fail threshold must be below pass threshold")
        return self


class RuntimeConfig(StrictModel):
    data_dir: Path
    seed: Annotated[int, Field(ge=0)]


class GroundTruthConfig(StrictModel):
    drop_rate: Probability
    fragment_prob: Probability
    fragment_gap_frames: Count
    confidence: Probability


class FixtureConfig(StrictModel):
    width: Count
    height: Count
    fps: Count
    duration_s: Positive
    transit_min_s: Positive
    transit_max_s: Positive
    clock_offset_s: float
    camera_b_scale: Positive
    noise_std: Annotated[float, Field(ge=0)]
    blur_radius: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def valid_video(self) -> Self:
        """Validate encoding dimensions and fixture timing."""
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 yuv420p requires even width and height")
        if self.transit_min_s > self.transit_max_s:
            raise ValueError("fixture transit bounds are reversed")
        if self.duration_s <= 40 + self.transit_max_s:
            raise ValueError("fixture duration must exceed 40 + maximum transit")
        return self


class AppConfig(StrictModel):
    target: TargetConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    crops: CropsConfig
    embedding: EmbeddingConfig
    color: ColorConfig
    cameras: CamerasConfig
    topology: TopologyConfig
    matching: MatchingConfig
    verifier: VerifierConfig
    runtime: RuntimeConfig
    groundtruth: GroundTruthConfig
    fixture: FixtureConfig

    @model_validator(mode="after")
    def normalized_weights(self) -> Self:
        """Require each weighted score to remain on the unit interval."""
        for weights in (self.crops.weights, self.matching.weights):
            if abs(sum(weights.model_dump().values()) - 1) > 1e-9:
                raise ValueError("score weights must sum to one")
        return self


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without mutating inputs; replace lists atomically."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> AppConfig:
    """Load defaults, a named preset, then per-run overrides and validate once."""
    directory = config_dir or Path(os.environ.get(
        "MOTO_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "configs"),
    ))

    def read(path: Path) -> dict[str, Any]:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected YAML mapping in {path}")
        return value

    data = read(directory / "default.yaml")
    if preset is not None:
        if not preset or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in preset):
            raise ValueError("Preset must be a simple lowercase name")
        data = deep_merge(data, read(directory / "presets" / f"{preset}.yaml"))
    return AppConfig.model_validate(deep_merge(data, overrides or {}))
