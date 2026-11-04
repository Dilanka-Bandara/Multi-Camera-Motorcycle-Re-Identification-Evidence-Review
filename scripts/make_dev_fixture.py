"""Generate reproducible two-camera motorcycle footage and normalized truth."""

import argparse
import csv
import json
import logging
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)


def make_schedule(config: AppConfig) -> list[dict[str, Any]]:
    """Define 14 identities; nine shared, three A-only and two B-only."""
    rng = random.Random(config.runtime.seed)
    colors = ["#df4545", "#df4545", "#418cde", "#418cde", "#d8b638",
              "#7d53b5", "#3d9772", "#e28736", "#b7b9c4", "#d15493",
              "#42acb3", "#9d783e", "#6d8753", "#835f9f"]
    identities: list[dict[str, Any]] = []
    for identity in range(14):
        entry_a = 1 + identity * 2.6 if identity < 12 else None
        exit_a = entry_a + 6 if entry_a is not None else None
        transit = rng.uniform(config.fixture.transit_min_s, config.fixture.transit_max_s)
        entry_b = exit_a + transit if identity < 9 else None
        if identity >= 12:
            entry_b = 8 + (identity - 12) * 25
        identities.append({
            "identity_id": identity, "body": colors[identity],
            "helmet": "#eee5c7" if identity < 4 else colors[(identity + 4) % 14],
            "torso": "#e8df82" if identity in (1, 3) else "#313851",
            "shape": 0 if identity < 4 else identity % 3,
            "entry_a_s": entry_a, "exit_a_s": exit_a,
            "entry_b_s": entry_b, "exit_b_s": entry_b + 6 if entry_b is not None else None,
            "transit_s": transit if identity < 9 else None,
        })
    return identities


def draw_bike(
    draw: ImageDraw.ImageDraw, x: float, y: float, scale: float, identity: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Draw the two wheels, body, torso and helmet; return enclosing pixel bounds."""
    def box(a: float, b: float, c: float, d: float) -> tuple[float, float, float, float]:
        return x + a * scale, y + b * scale, x + c * scale, y + d * scale

    for wheel_x in (-25, 25):
        draw.ellipse(box(wheel_x - 10, -10, wheel_x + 10, 10), fill="#171b22")
        draw.ellipse(box(wheel_x - 5, -5, wheel_x + 5, 5), fill="#a0a4ad")
    polygon = [(-30, -13), (-8, -28 - identity["shape"] * 3), (20, -24), (34, -12), (4, -5)]
    draw.polygon([(x + a * scale, y + b * scale) for a, b in polygon], fill=identity["body"])
    draw.rectangle(box(-12, -52, 7, -26), fill=identity["torso"])
    draw.ellipse(box(-13, -70, 9, -48), fill=identity["helmet"])
    draw.line([(x + 5 * scale, y - 40 * scale), (x + 25 * scale, y - 25 * scale)],
              fill=identity["torso"], width=max(1, round(5 * scale)))
    draw.line([(x + 25 * scale, y - 25 * scale), (x + 30 * scale, y - 10 * scale)],
              fill="#232830", width=max(1, round(3 * scale)))
    return box(-35, -70, 35, 10)


def render_camera(
    out: Path, camera: str, identities: list[dict[str, Any]], config: AppConfig,
) -> None:
    """Stream RGB frames into ffmpeg without storing raw frames on disk."""
    cfg = config.fixture
    width, height = cfg.width, cfg.height
    rng = np.random.default_rng(config.runtime.seed + (camera == "B"))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(cfg.fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out / f"camera_{camera}.mp4"),
    ]
    scale = width / 640 * (cfg.camera_b_scale if camera == "B" else 1)
    with (out / f"gt_{camera}.csv").open("x", newline="", encoding="utf-8") as gt_file:
        writer = csv.writer(gt_file)
        writer.writerow(["frame_idx", "identity_id", "x1", "y1", "x2", "y2", "cls"])
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            try:
                for frame in range(round(cfg.fps * cfg.duration_s)):
                    t_s = frame / cfg.fps
                    image = Image.new("RGB", (width, height), (105, 108, 112))
                    draw = ImageDraw.Draw(image)
                    for lane_y in (height * 0.43, height * 0.8):
                        for lane_x in range(0, width, max(1, width // 8)):
                            draw.line([(lane_x, lane_y), (lane_x + width / 16, lane_y)],
                                      fill="#cdc9b7", width=3)
                    for identity in identities:
                        entry = identity[f"entry_{camera.lower()}_s"]
                        if entry is None or not entry <= t_s <= entry + 6:
                            continue
                        x = -35 * scale + (t_s - entry) / 6 * (width + 70 * scale)
                        y = height * (0.61 + (identity["identity_id"] % 2) * 0.18)
                        x1, y1, x2, y2 = draw_bike(draw, x, y, scale, identity)
                        bounds = [max(0, x1) / width, max(0, y1) / height,
                                  min(width, x2) / width, min(height, y2) / height]
                        if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
                            writer.writerow([frame, identity["identity_id"], *bounds, 3])
                    pole_x = width * (0.53 if camera == "A" else 0.63)
                    draw.rectangle((pole_x, height * 0.2, pole_x + width * 0.025, height),
                                   fill="#555b60")
                    if camera == "B":
                        image = image.filter(ImageFilter.GaussianBlur(cfg.blur_radius))
                    pixels = np.asarray(image, dtype=np.float32)
                    if camera == "B":
                        pixels = pixels * np.array([0.88, 0.94, 1.03]) + 5
                    pixels += rng.normal(0, cfg.noise_std, pixels.shape)
                    encoder.stdin.write(np.clip(pixels, 0, 255).astype(np.uint8).tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"ffmpeg failed encoding camera {camera}")
    LOGGER.info("run_id=dev_fixture camera=%s generated", camera)


def generate_fixture(out: Path, config: AppConfig) -> None:
    """Create both videos, GT and evaluation-only metadata without overwriting files."""
    names = ["camera_A.mp4", "camera_B.mp4", "gt_A.csv", "gt_B.csv", "fixture_meta.json"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Fixture outputs already exist; choose a new --out directory")
    out.mkdir(parents=True, exist_ok=True)
    identities = make_schedule(config)
    for camera in ("A", "B"):
        render_camera(out, camera, identities, config)
    metadata = {
        "seed": config.runtime.seed, "true_clock_offset_s": config.fixture.clock_offset_s,
        "camera_start_times_s": {"A": 0, "B": config.fixture.clock_offset_s},
        "true_pairs": [[i, i] for i in range(9)], "a_only": [9, 10, 11], "b_only": [12, 13],
        "lookalike_pairs": [[0, 1], [2, 3]], "identities": identities,
        "video": config.fixture.model_dump(),
        "time_convention": "Frame times are physical; camera B timestamps add its clock offset. "
                           "Transit is B entry minus A exit, before adding that offset.",
    }
    (out / "fixture_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    """Generate the default fixture from repository YAML settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    generate_fixture(args.out, load_config("dev_fixture", config_dir=args.config_dir))


if __name__ == "__main__":
    main()
