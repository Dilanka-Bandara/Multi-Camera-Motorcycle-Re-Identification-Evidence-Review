"""Fit the development fixture's verifier calibration on separately seeded fixtures.

The fixture uses a deterministic dummy embedder and SIFT, so its verifier features
are distributed differently from the DINO/LightGlue features behind
configs/calibrator.json. Tuning fixtures share the 14 sprite designs of the baked
fixture but draw different transit times and noise; the baked seed never enters
fitting or threshold selection.
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from app.config import AppConfig, load_config
from app.evaluation.gt import map_truth
from app.evaluation.metrics import evaluate
from app.pipeline.orchestrator import run_pipeline
from app.schemas import QueryResult, RunResult
from app.verify.calibrator import Calibrator
from scripts.fit_calibrator import reviewed_queries
from scripts.make_dev_fixture import generate_fixture

PRESET = "dev_fixture"
Labelled = tuple[list[QueryResult], dict[str, str], dict[str, set[str]], dict[str, str]]


def prepare(seed: int, work: Path, models: Path) -> tuple[Labelled, list[list[str]]]:
    """Generate one tuning fixture, run the pipeline once and attach identities."""
    fixture, run_dir = work / "fixtures" / str(seed), work / "runs" / str(seed)
    config = load_config(PRESET, {"runtime": {"seed": seed}})
    if not fixture.exists():
        generate_fixture(fixture, config)
    truth = {camera: fixture / f"gt_{camera}.csv" for camera in ("A", "B")}
    if not (run_dir / "results.json").is_file():
        videos = {camera: fixture / f"camera_{camera}.mp4" for camera in ("A", "B")}
        run_pipeline(config, videos, run_dir, truth, models)
    run = RunResult.model_validate_json((run_dir / "results.json").read_bytes())
    a, ids_a = map_truth(run.tracks_A, truth["A"], config.evaluation)
    b, ids_b = map_truth(run.tracks_B, truth["B"], config.evaluation)
    cameras = {t.track_id: t.camera for t in run.tracks_A + run.tracks_B}
    meta = json.loads((fixture / "fixture_meta.json").read_text(encoding="utf-8"))
    lookalikes = [[str(i) for i in pair] for pair in meta["lookalike_pairs"]]
    return (run.queries, a | b, {"A": ids_a, "B": ids_b}, cameras), lookalikes


def lookalikes_flagged(
    queries: list[QueryResult], labels: dict[str, str], pairs: list[list[str]]
) -> bool:
    """Phase 4 acceptance: each lookalike pair keeps an ambiguous leading proposal."""
    flagged = {
        labels[q.query_track_id]
        for q in queries
        if q.candidates and q.candidates[0].ambiguous and q.query_track_id in labels
    }
    return all(flagged & set(pair) for pair in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Tuning fixtures and runs")
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    logging.basicConfig(level=logging.WARNING)
    settings = yaml.safe_load(Path("configs/data_preparation.yaml").read_text(encoding="utf-8"))[
        "fixture_calibration"
    ]
    base = load_config(PRESET)
    seeds = settings["tuning_seeds"]
    if base.runtime.seed in seeds:
        raise ValueError("The baked fixture seed must stay out of tuning")
    names = list(Calibrator.load(Path("configs/calibrator.json")).weights)
    data, pairs = {}, {}
    for seed in seeds:
        data[seed], pairs[seed] = prepare(seed, args.work, args.models_dir)
    rows, outcomes = [], []
    for queries, labels, _, _ in data.values():
        for q in queries:
            for c in q.candidates:
                if c.verification and {q.query_track_id, c.candidate_track_id} <= labels.keys():
                    rows.append([c.verification.features[k] for k in names])
                    outcomes.append(
                        int(labels[q.query_track_id] == labels[c.candidate_track_id])
                    )
    best: tuple[float, float, float, float] = (-1.0, -1.0, 0.0, 0.0)
    chosen: tuple[Calibrator, AppConfig, dict[str, int]] | None = None
    grid = list(
        itertools.product(
            settings["min_match_scores"],
            settings["ambiguity_margins"],
            settings["accept_probabilities"],
            settings["reject_probabilities"],
        )
    )
    for strength in settings["regularizations"]:
        model = LogisticRegression(
            C=strength, max_iter=settings["max_iterations"], random_state=settings["seed"]
        ).fit(np.asarray(rows), outcomes)
        calibration = Calibrator(
            intercept=float(model.intercept_[0]),
            weights=dict(zip(names, model.coef_[0].tolist(), strict=True)),
        )
        for minimum, ambiguity, accept, reject in grid:
            if reject >= accept:
                continue
            config = base.model_copy(deep=True)
            config.matching.min_match_score = minimum
            config.matching.ambiguity_margin = ambiguity
            config.verifier.p_accept, config.verifier.p_reject = accept, reject
            tp = fp = positives = 0
            valid = True
            for seed, (queries, labels, gallery, cameras) in data.items():
                predicted = reviewed_queries(queries, calibration, config)
                valid = valid and lookalikes_flagged(predicted, labels, pairs[seed])
                metrics = evaluate(
                    predicted, labels, gallery, cameras, base.evaluation.calibration_bins
                )
                tp += metrics["decisions"]["accepted"]["tp"]
                fp += metrics["decisions"]["accepted"]["fp"]
                positives += metrics["queries_with_match"]
            if not valid:
                continue
            f1 = 2 * tp / (positives + tp + fp) if positives + tp + fp else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            # Ties prefer the wider ambiguity margin, then stronger regularization.
            score = (f1, precision, ambiguity, -strength)
            if score > best:
                best = score
                chosen = (calibration, config, {"tp": tp, "fp": fp, "positives": positives})
    if chosen is None:
        raise RuntimeError("No threshold combination keeps both lookalike pairs flagged")
    calibration, config, counts = chosen
    args.out.mkdir(parents=True)
    calibration.save(args.out / "calibrator.json")
    manifest = {
        "method": "scikit-learn LogisticRegression on separately seeded fixture renderings",
        "preset": PRESET,
        "tuning_seeds": seeds,
        "excluded_baked_seed": base.runtime.seed,
        "training_pairs": len(rows),
        "training_positives": sum(outcomes),
        "features": names,
        "settings": settings,
        "constraint": "every tuning fixture keeps both lookalike pairs flagged ambiguous",
        "selection": "pooled accepted F1, precision, wider ambiguity margin, smaller C",
        "regularization": -best[3],
        "tuning_accepted": {**counts, "f1": best[0], "precision": best[1]},
        "thresholds": {
            "min_match_score": config.matching.min_match_score,
            "ambiguity_margin": config.matching.ambiguity_margin,
            "p_accept": config.verifier.p_accept,
            "p_reject": config.verifier.p_reject,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["thresholds"] | manifest["tuning_accepted"], indent=2))


if __name__ == "__main__":
    main()
