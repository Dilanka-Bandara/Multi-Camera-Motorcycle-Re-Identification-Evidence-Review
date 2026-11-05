"""Print the metrics table and optionally save a run's evaluation report."""

import argparse
import json
from pathlib import Path

from app.schemas import RunResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.run if args.run.is_dir() else Path("data/user/runs") / args.run
    result = RunResult.model_validate_json((root / "results.json").read_bytes())
    if result.metrics is None:
        parser.error("This run has no evaluation ground truth")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result.metrics, stream, indent=2)
    print("| Metric | Value |\n|---|---|")
    for key, value in result.metrics.items():
        if not isinstance(value, (list, dict)):
            print(f"| {key} | {value} |")


if __name__ == "__main__":
    main()
