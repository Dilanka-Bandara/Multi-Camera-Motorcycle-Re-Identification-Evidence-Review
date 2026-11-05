"""Create paired camera crops and transform VnTraffic MOT-format identities consistently."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.pipeline.ingest import probe_video


def transform_box(
    box: tuple[float, float, float, float], left: int, width: int, height: int
) -> tuple[float, float, float, float] | None:
    x, y, w, h = box
    x1, y1 = max(x, left), max(0, y)
    x2, y2 = min(x + w, left + width), min(y + h, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 - left) / width, y1 / height, (x2 - left) / width, y2 / height)


def make_pair(
    video: Path, gt: Path, out: Path, settings: dict[str, float], preset: Path, sample: Path
) -> None:
    input_directory = out.relative_to(Path("data")).as_posix()
    meta = probe_video(video, 0)
    if out.exists() or preset.exists() or sample.exists():
        raise FileExistsError("Choose fresh output, sample and preset paths")
    out.mkdir(parents=True)
    sample.mkdir(parents=True)
    crop_w = int(meta.width * settings["crop_fraction"]) // 2 * 2
    delay = round(settings["delay_s"] * meta.fps)
    offsets = {"A": 0, "B": meta.width - crop_w}
    durations = {}
    for camera in ("A", "B"):
        scale = settings["scale_b"] if camera == "B" else 1
        width, height = int(crop_w * scale) // 2 * 2, int(meta.height * scale) // 2 * 2
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(meta.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / f"camera_{camera}.mp4"),
        ]
        capture = cv2.VideoCapture(str(video))
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        if encoder.stdin is None:
            raise RuntimeError("Encoder has no input stream")
        frame_count = 0
        try:
            if camera == "B":
                black = np.zeros((height, width, 3), np.uint8).tobytes()
                for _ in range(delay):
                    encoder.stdin.write(black)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                cropped = frame[:, offsets[camera] : offsets[camera] + crop_w]
                if camera == "B":
                    cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
                    values = (cropped.astype(np.float32) / 255) ** settings["gamma"]
                    values *= np.array([settings["blue_gain"], 1, settings["red_gain"]])
                    cropped = np.clip(values * 255, 0, 255).astype(np.uint8)
                    cropped = cv2.GaussianBlur(cropped, (0, 0), settings["blur_sigma"])
                    ok, encoded = cv2.imencode(
                        ".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, int(settings["jpeg_quality"])]
                    )
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    cropped = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                encoder.stdin.write(cropped.tobytes())
                frame_count += 1
        finally:
            capture.release()
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f"Camera {camera} encoding failed")
        durations[camera] = (frame_count + (delay if camera == "B" else 0)) / meta.fps
        with (
            gt.open(newline="") as source,
            (out / f"gt_{camera}.csv").open("x", newline="") as target,
        ):
            writer = csv.writer(target)
            writer.writerow(["frame_idx", "identity_id", "x1", "y1", "x2", "y2"])
            for row in csv.reader(source):
                frame, ident = int(row[0]), row[1]
                bbox = transform_box(
                    tuple(float(v) for v in row[2:6]), offsets[camera], crop_w, meta.height
                )
                if bbox and 0 <= frame < frame_count:
                    writer.writerow([frame + (delay if camera == "B" else 0), ident, *bbox])
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-i",
                str(out / f"camera_{camera}.mp4"),
                "-vf",
                f"scale=-2:{int(settings['sample_height'])}",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                str(sample / f"camera_{camera}.mp4"),
            ],
            check=True,
        )
    config = {
        "inputs": {"directory": input_directory},
        "target": {"classes": ["motorcycle"]},
        "detector": {"mode": "yolo"},
        "cameras": {"B": {"start_time_s": settings["clock_offset_s"]}},
        "topology": {
            "direction": "A_to_B",
            "min_transit_s": 0,
            "max_transit_s": settings["delay_s"] + meta.duration_s,
            "clock_offset": {"mode": "fixed", "fixed_s": settings["clock_offset_s"]},
        },
    }
    preset.parent.mkdir(parents=True, exist_ok=True)
    preset.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    info = {
        "source": str(video),
        "source_duration_s": meta.duration_s,
        "output_duration_s": durations,
        "settings": settings,
        "class_annotation": "unknown; identity boxes cover all vehicle classes",
        "timing": "B frame indices include the synthetic delay. B timestamp adds clock_offset_s.",
        "license": "CC-BY-4.0",
        "source_record": "https://zenodo.org/records/18195750",
    }
    (out / "preparation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (sample / "README.md").write_text(
        "# VnTraffic synthetic camera sample\n\n"
        "Derived from Vehicle Tracking Dataset, Thi-Nhung Le and contributors (2026). "
        "Video source credited by the dataset authors to HTV, Hanoi Radio and Television Station. "
        "Licensed CC BY 4.0. Source and attribution: https://zenodo.org/records/18195750\n\n"
        "Changes include left/right overlapping crops, delayed Camera B, resize, "
        "colour/gamma changes, blur and JPEG degradation, then H.264 downscaling. "
        f"The source duration is {meta.duration_s:.2f} seconds. "
        f"B includes a {delay / meta.fps:.2f}-second blank delay. "
        "The annotations supply identity boxes without class labels. "
        "This pair derives from one view.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/user/datasets/vntraffic_synth"))
    parser.add_argument("--preset", type=Path, default=Path("configs/presets/vntraffic_synth.yaml"))
    parser.add_argument("--sample", type=Path, default=Path("samples/vntraffic_synth"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_preparation.yaml"))
    parser.add_argument("--clock-offset-s", type=float)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())["synthetic"]
    if args.clock_offset_s is not None:
        settings["clock_offset_s"] = args.clock_offset_s
    make_pair(args.video, args.gt, args.out, settings, args.preset, args.sample)


if __name__ == "__main__":
    main()
