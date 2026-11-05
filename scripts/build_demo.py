"""Assemble a captioned 4:20 demo from real browser captures (no fabricated UI)."""

import argparse
import json
import subprocess
import textwrap
from pathlib import Path


def render(source: Path, output: Path, seconds: int, title: str, caption: str) -> None:
    """Encode a captured sequence or still with a separate caption band."""
    title_path = output.with_suffix(".title.txt")
    caption_path = output.with_suffix(".caption.txt")
    title_path.write_text(title, encoding="utf-8")
    caption_path.write_text(textwrap.fill(caption, width=98), encoding="utf-8")
    if source.is_dir():
        manifest = json.loads((source / "frames.json").read_text())
        frames = manifest["frames"]
        lines = []
        for index, frame in enumerate(frames):
            end = frames[index + 1]["at"] if index + 1 < len(frames) else manifest["duration"]
            name = (source / frame["file"]).resolve().as_posix()
            lines.extend([f"file '{name}'", f"duration {max(0.001, end - frame['at']):.6f}"])
        lines.append(f"file '{(source / frames[-1]['file']).resolve().as_posix()}'")
        listing = output.with_suffix(".frames.txt")
        listing.write_text("\n".join(lines), encoding="utf-8")
        inputs = ["-f", "concat", "-safe", "0", "-i", str(listing)]
    else:
        inputs = ["-loop", "1", "-framerate", "15", "-i", str(source)]
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = (
        "fps=15,scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:900:(ow-iw)/2:0:color=0x10191f,"
        f"tpad=stop_mode=clone:stop_duration={seconds},"
        f"drawtext=fontfile={font}:textfile={title_path}:x=32:y=739:fontsize=28:"
        "fontcolor=0x81d5c4:expansion=none,"
        f"drawtext=fontfile={font}:textfile={caption_path}:x=32:y=783:fontsize=22:"
        "line_spacing=8:fontcolor=white:expansion=none"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            *inputs,
            "-vf",
            filters,
            "-t",
            str(seconds),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=Path("data/demo/capture"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    c = args.capture
    e = Path("docs/evidence")
    scenes = [
        (
            c / "home.jpg",
            30,
            "1 / 8  |  MotoReID: two feeds, evidence-based review",
            "Detect and track -> diverse crops and DINO features -> timing and "
            "assignment -> independent verification. The system can abstain. This "
            "captioned walkthrough combines recorded UI playback with still evidence "
            "views.",
        ),
        (
            c / "fixture-progress",
            30,
            "2 / 8  |  An actual fixture job",
            "Recorded processing: the fixture completed in 17.6 seconds. It uses ground- "
            "truth detections and dummy embeddings for repeatable tests. Real YOLO runs "
            "took about nine minutes, including crop processing and geometric "
            "verification.",
        ),
        (
            c / "real-playback-framed",
            12,
            "3 / 8  |  Real-video playback and track overlays",
            "VnTraffic baseline: both cameras play from their selected track positions. "
            "Yellow marks the selected track; teal marks other tracks. Camera B is a "
            "delayed, degraded crop of the same source view.",
        ),
        (
            e / "vntraffic-correct-uncertain-detail.jpg",
            33,
            "3 / 8  |  Correct retrieval is not automatic acceptance",
            "A-0000 and B-0018 both map to identity 39. Forty geometric inliers support "
            "the pairing. Competing candidates and another query's stronger claim keep "
            "it uncertain. The 0.829 probability shown is from the hand-set baseline.",
        ),
        (
            e / "vntraffic-false-top.jpg",
            22,
            "4 / 8  |  A wrong top candidate is withheld",
            "A-0006 is identity 44; B-0018 is identity 39. Visual similarity is high, "
            "but independent geometry has zero inliers. The uncertain verdict avoids a "
            "false accepted match. No true identity appears in this query's top five.",
        ),
        (
            e / "vntraffic-rank-two.jpg",
            23,
            "4 / 8  |  A near-miss in candidate ranking",
            "A-0017 is identity 46. B-0013 is the wrong leader; B-0035 is the correct "
            "identity at rank two. Similar scores, missing rider crops and competing "
            "claims explain uncertainty. Abstention does not fix the Rank-1 error.",
        ),
        (
            c / "no-reliable-match.jpg",
            15,
            "5 / 8  |  No reliable match is a valid outcome",
            "The fixture's A-0010 query has no reliable match. Query outcome and "
            "individual candidate verdict are different. Abstention avoids claiming "
            "that an uncertain candidate establishes an identity.",
        ),
        (
            c / "roundabout-rejected.jpg",
            15,
            "5 / 8  |  Real rejection: timing contradicts appearance",
            "RoundaboutHD baseline A-0000 -> B-0039: 60 geometric inliers, but "
            "adjusted transit is -33.9 seconds, triggering rejection. This demonstrates "
            "the configured transit rule, not a confirmed different identity. "
            "Overlapping views can violate the non-overlap assumption.",
        ),
        (
            c / "clock-fixture.jpg",
            20,
            "6 / 8  |  Clock offset with a stated assumption",
            "The fixture estimates +3.567 seconds against a true +4 seconds using a "
            "travel-time prior. VnTraffic's +6 seconds is configured, not estimated. "
            "Unknown clock offset cannot be separated from travel time without an "
            "assumption.",
        ),
        (
            c / "fixture-metrics.jpg",
            40,
            "7 / 8  |  Training and held-out results",
            "Logistic calibration: 14 tuning pairs (5 positive, 9 negative); neural "
            "weights stay pretrained. Held-out Rank-1: RoundaboutHD 10/14; VnTraffic "
            "2/12. No labelled held-out query was accepted. One unmapped accepted "
            "baseline query is unscored. Screen: fixture metrics.",
        ),
        (
            c / "fixture-metrics.jpg",
            20,
            "8 / 8  |  Exports, reproducibility and credits",
            "JSON, candidate CSV and query CSV downloads were verified. Fresh Docker "
            "inference passed with networking disabled. 57 tests pass. VnTraffic: "
            "Thi-Nhung Le and contributors / HTV, CC BY 4.0. RoundaboutHD: yl4300 "
            "dataset on Hugging Face. See README reports and limitations.",
        ),
    ]
    for index, (source, duration, title, caption) in enumerate(scenes):
        render(source, args.out / f"scene-{index:02}.mp4", duration, title, caption)
        print(f"Rendered scene {index + 1}/{len(scenes)}", flush=True)
    listing = args.out / "scenes.txt"
    listing.write_text(
        "\n".join(f"file 'scene-{i:02}.mp4'" for i in range(len(scenes))), encoding="utf-8"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.out / "moto-reid-demo.mp4"),
        ],
        check=True,
    )
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "duration_s": sum(scene[1] for scene in scenes),
                "format": (
                    "Captioned walkthrough; recorded UI sequences "
                    "plus still evidence views; no audio"
                ),
                "captures": str(c),
                "baseline_ui_run": "5d25ad36e97a46c89684f030f663ee10",
                "fixture_ui_run": "4b54d955c9444324ab89f95b73173ccb",
                "roundabout_ui_run": "142105526f104a17a51334b4d405a06c",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
