"""Convert RoundaboutHD's zero-based multi-camera GT and a selected time interval."""

import argparse
import csv
import itertools
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from app.evaluation.split import tuning_identity
from app.pipeline.ingest import probe_video


def category_summary(root: Path) -> dict[str, object]:
    """Inspect workbook categories without confusing SCT IDs with global IDs."""
    workbooks = [p for p in root.glob("*.xlsx") if p.stem.lower() == "vehicle_statistic"]
    if not workbooks:
        return {"status": "class unknown; statistics workbook absent"}
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    counts: dict[str, int] = defaultdict(int)
    with zipfile.ZipFile(workbooks[0]) as archive:
        strings = [
            "".join(item.itertext())
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)
        ]
        for name in archive.namelist():
            if not (name.startswith("xl/worksheets/sheet") and name.endswith(".xml")):
                continue
            sheet = ET.fromstring(archive.read(name))
            for row in sheet.findall("s:sheetData/s:row", ns)[1:]:
                for cell in row:
                    value = cell.find("s:v", ns)
                    if cell.get("r", "").rstrip("0123456789") == "F" and value is not None:
                        category = strings[int(value.text)] if cell.get("t") == "s" else value.text
                        if category:
                            counts[category] += 1
    return {
        "status": "SCT categories available; global-identity class mapping unverified",
        "category_counts": dict(counts),
        "motorcycle_rows": sum(v for k, v in counts.items() if "motorcycle" in k.lower()),
        "pair_motorcycles": "class unknown; do not equate local SCT IDs with global IDs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cam-a", type=int, default=1)
    parser.add_argument("--cam-b", type=int, default=2)
    parser.add_argument("--start-s", type=float, default=0)
    parser.add_argument("--duration-s", type=float, default=60)
    parser.add_argument("--out", type=Path, default=Path("data/user/datasets/roundabouthd_pair"))
    parser.add_argument(
        "--preset", type=Path, default=Path("configs/presets/roundabouthd_pair.yaml")
    )
    args = parser.parse_args()
    if args.cam_a == args.cam_b or args.start_s < 0 or args.duration_s <= 0:
        parser.error("Choose two different cameras and a positive interval")
    directory = args.out.relative_to(Path("data")).as_posix()
    files = list(args.source.rglob("Multi_CAM_Ground_Truth.txt"))
    files += list(args.source.rglob("Multi_CAM_Ground_Turth.txt"))
    if len(files) != 1:
        parser.error("Expected exactly one Multi_CAM_Ground_Truth.txt")
    root = files[0].parent
    classes = category_summary(root)
    print(json.dumps(classes))
    records: dict[int, list[tuple[str, int, list[float]]]] = defaultdict(list)
    identities: dict[int, set[str]] = defaultdict(set)
    for line in files[0].read_text().splitlines():
        values = line.replace(",", " ").split()
        if not values:
            continue
        camera, identity, frame = int(values[0]), values[1], int(values[2])
        records[camera].append((identity, frame, [float(x) for x in values[3:7]]))
        identities[camera].add(identity)
    for a, b in itertools.combinations(sorted(identities), 2):
        print(
            f"Camera {a}/{b}: {len(identities[a] & identities[b])} shared identities; "
            "motorcycles: class unknown (workbook uses SCT IDs)"
        )
    if args.out.exists() or args.preset.exists():
        raise FileExistsError("Choose fresh output and preset paths")
    args.out.mkdir(parents=True)
    intervals: dict[str, dict[str, list[float]]] = {}
    for label, camera in (("A", args.cam_a), ("B", args.cam_b)):
        video = root / f"imagesc{camera:03d}" / "video.mp4"
        meta = probe_video(video, 0)
        first = round(args.start_s * meta.fps)
        last = min(meta.frame_count, first + round(args.duration_s * meta.fps))
        if first >= last:
            raise ValueError("Requested interval is outside the video")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-i",
                str(video),
                "-vf",
                f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS,scale=-2:1080",
                "-an",
                "-frames:v",
                str(last - first),
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
                str(args.out / f"camera_{label}.mp4"),
            ],
            check=True,
        )
        intervals[label] = defaultdict(list)
        with (args.out / f"gt_{label}.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_idx", "identity_id", "x1", "y1", "x2", "y2"])
            for identity, frame, (x, y, w, h) in records[camera]:
                if not first <= frame < last:
                    continue
                box = (
                    max(0, x / meta.width),
                    max(0, y / meta.height),
                    min(1, (x + w) / meta.width),
                    min(1, (y + h) / meta.height),
                )
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                writer.writerow([frame - first, identity, *box])
                intervals[label][identity].append((frame - first) / meta.fps)
    settings = yaml.safe_load(Path("configs/data_preparation.yaml").read_text())["roundabout"]
    shared = intervals["A"].keys() & intervals["B"].keys()
    deltas = []
    for ident in shared:
        if tuning_identity(ident):
            a, b = intervals["A"][ident], intervals["B"][ident]
            deltas.append(max(min(b) - max(a), min(a) - max(b)))
    positive = [d for d in deltas if d >= 0]
    if not positive:
        raise ValueError("No tuning-split transit pairs; choose another interval")
    lower, upper = np.quantile(positive, [settings["lower_quantile"], settings["upper_quantile"]])
    margin = max(settings["min_window_s"], float(upper - lower)) * settings["window_margin"]
    lower = max(0, float(lower) - margin)
    upper = max(lower + settings["min_window_s"], float(upper) + margin)
    cfg = {
        "inputs": {"directory": directory},
        "target": {"classes": ["car", "motorcycle", "bus", "truck"]},
        "detector": {"mode": "yolo"},
        "topology": {
            "direction": "both",
            "min_transit_s": lower,
            "max_transit_s": upper,
            "clock_offset": {"mode": "fixed", "fixed_s": 0},
        },
    }
    args.preset.parent.mkdir(parents=True, exist_ok=True)
    args.preset.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    info = {
        "camera_pair": [args.cam_a, args.cam_b],
        "start_s": args.start_s,
        "duration_s": args.duration_s,
        "shared_in_segment": len(shared),
        "tuning_transit_pairs": len(positive),
        "class_annotation": "unknown",
        "class_inspection": classes,
        "clock_assumption": "Videos share zero offset; fixed 0 s",
        "transit_window_s": [lower, upper],
        "source": "https://huggingface.co/datasets/yl4300/RoundaboutHD",
    }
    (args.out / "preparation.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
