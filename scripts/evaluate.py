"""Print the metrics table and optionally save a run's evaluation report."""

import argparse
import json
from pathlib import Path

from app.config import AppConfig
from app.evaluation.gt import map_truth
from app.evaluation.metrics import evaluate
from app.schemas import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gt-a", type=Path)
    parser.add_argument("--gt-b", type=Path)
    args = parser.parse_args()
    root = args.run if args.run.is_dir() else Path("data/user/runs") / args.run
    result = RunResult.model_validate_json((root / "results.json").read_bytes())
    if bool(args.gt_a) != bool(args.gt_b):
        parser.error("Provide both --gt-a and --gt-b")
    if args.gt_a:
        cfg = AppConfig.model_validate(result.config_snapshot)
        a, ids_a = map_truth(result.tracks_A, args.gt_a, cfg.evaluation)
        b, ids_b = map_truth(result.tracks_B, args.gt_b, cfg.evaluation)
        metrics = evaluate(
            result.queries,
            a | b,
            {"A": ids_a, "B": ids_b},
            {t.track_id: t.camera for t in result.tracks_A + result.tracks_B},
            cfg.evaluation.calibration_bins,
        )
        result.metrics = {**(result.metrics or {}), **metrics}
    if result.metrics is None:
        parser.error("This run has no evaluation ground truth")
    lines = ["| Metric | Value |", "|---|---|"]
    for key, value in result.metrics.items():
        if not isinstance(value, (list, dict)):
            lines.append(f"| {key} | {value} |")
    for mode, values in result.metrics.get("decisions", {}).items():
        lines.extend(f"| {mode} {key} | {value} |" for key, value in values.items())
    table = "\n".join(lines) + "\n"
    if args.out:
        markdown = args.out.with_suffix(".md")
        if args.out.exists() or markdown.exists():
            raise FileExistsError("Choose fresh report paths")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result.metrics, stream, indent=2)
        with markdown.open("x", encoding="utf-8") as stream:
            stream.write(table)
    print(table)


if __name__ == "__main__":
    main()
