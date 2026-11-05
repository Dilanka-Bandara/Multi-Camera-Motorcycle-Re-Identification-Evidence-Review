"""Fit on identity-separated tuning data and evaluate held-out queries once."""

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from app.config import AppConfig
from app.evaluation.gt import map_truth
from app.evaluation.metrics import evaluate
from app.evaluation.split import tuning_identity, tuning_pair
from app.matching.assignment import assign
from app.schemas import QueryResult, RunResult
from app.verify.calibrator import Calibrator
from app.verify.policy import decide, resolve, summarize


def reviewed_queries(
    queries: list[QueryResult], calibration: Calibrator, config: AppConfig
) -> list[QueryResult]:
    rows = deepcopy(queries)
    assign([q.candidates for q in rows], config.matching)
    claims: dict[str, float] = {}
    for query in rows:
        if query.candidates:
            lead = query.candidates[0]
            claims[lead.candidate_track_id] = max(
                claims.get(lead.candidate_track_id, 0), lead.scores.combined
            )
    results = []
    for q in rows:
        for c in q.candidates:
            v = c.verification
            if v is None:
                continue
            if claims.get(c.candidate_track_id, 0) <= c.scores.combined:
                v.evidence = [e for e in v.evidence if e.check != "assignment_conflict"]
            for e in v.evidence:
                if e.check == "ambiguity":
                    margin = v.features["margin"]
                    e.polarity = (
                        "contradicts"
                        if c.ambiguous
                        else "supports"
                        if margin >= 2 * config.matching.ambiguity_margin
                        else "neutral"
                    )
                    e.severity = "high" if c.ambiguous else "low"
                    e.threshold = config.matching.ambiguity_margin
                    e.message = (
                        "another pairing has similar support"
                        if c.ambiguous
                        else f"ranking margin is {margin:.2f}"
                    )
            v.confidence = calibration.probability(v.features)
            v.status = decide(
                v.confidence,
                v.evidence,
                c.ambiguous,
                v.features["track_quality_min"],
                config.verifier,
            )
            v.summary = summarize(v.status, v.evidence)
        results.append(resolve(q.query_track_id, q.candidates))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gt-a", type=Path, required=True)
    parser.add_argument("--gt-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--additional-training",
        type=Path,
        nargs=3,
        action="append",
        default=[],
        metavar=("RUN", "GT_A", "GT_B"),
        help="Pool only tuning pairs from another dataset; never its held-out queries",
    )
    parser.add_argument("--coefficients", type=Path, help="Freeze existing JSON coefficients")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.coefficients and args.additional_training:
        parser.error("Frozen coefficients cannot be combined with training inputs")
    run = RunResult.model_validate_json((args.run / "results.json").read_bytes())
    cfg = AppConfig.model_validate(run.config_snapshot)
    settings = yaml.safe_load(Path("configs/data_preparation.yaml").read_text())["calibration"]
    a, ids_a = map_truth(run.tracks_A, args.gt_a, cfg.evaluation)
    b, ids_b = map_truth(run.tracks_B, args.gt_b, cfg.evaluation)
    labels = a | b
    cameras = {t.track_id: t.camera for t in run.tracks_A + run.tracks_B}
    gallery = {"A": ids_a, "B": ids_b}
    tuning = [
        q
        for q in run.queries
        if q.query_track_id in labels and tuning_identity(labels[q.query_track_id])
    ]
    held_out = [
        q
        for q in run.queries
        if q.query_track_id in labels and not tuning_identity(labels[q.query_track_id])
    ]
    rows, outcomes = [], []
    names = list(Calibrator.load(Path("configs/calibrator.json")).weights)
    for q in tuning:
        for c in q.candidates:
            if (
                c.verification
                and c.candidate_track_id in labels
                and tuning_pair(labels[q.query_track_id], labels[c.candidate_track_id])
            ):
                rows.append([c.verification.features[k] for k in names])
                outcomes.append(int(labels[q.query_track_id] == labels[c.candidate_track_id]))
    training_sources = [{"run": str(args.run), "pairs": len(rows), "positives": sum(outcomes)}]
    for extra_run, extra_a, extra_b in args.additional_training:
        extra = RunResult.model_validate_json((extra_run / "results.json").read_bytes())
        extra_cfg = AppConfig.model_validate(extra.config_snapshot)
        extra_labels = (
            map_truth(extra.tracks_A, extra_a, extra_cfg.evaluation)[0]
            | map_truth(extra.tracks_B, extra_b, extra_cfg.evaluation)[0]
        )
        before, positives_before = len(rows), sum(outcomes)
        for q in extra.queries:
            if q.query_track_id not in extra_labels:
                continue
            for c in q.candidates:
                if (
                    c.verification
                    and c.candidate_track_id in extra_labels
                    and tuning_pair(
                        extra_labels[q.query_track_id], extra_labels[c.candidate_track_id]
                    )
                ):
                    rows.append([c.verification.features[k] for k in names])
                    outcomes.append(
                        int(extra_labels[q.query_track_id] == extra_labels[c.candidate_track_id])
                    )
        training_sources.append(
            {
                "run": str(extra_run),
                "pairs": len(rows) - before,
                "positives": sum(outcomes) - positives_before,
            }
        )
    if (not args.coefficients and len(set(outcomes)) != 2) or not held_out:
        raise ValueError("Both training classes and held-out identities are required")
    if args.coefficients:
        calibration = Calibrator.load(args.coefficients)
    else:
        model = LogisticRegression(
            C=settings["regularization"],
            max_iter=settings["max_iterations"],
            random_state=settings["seed"],
        ).fit(np.asarray(rows), outcomes)
        calibration = Calibrator(
            intercept=float(model.intercept_[0]),
            weights=dict(zip(names, model.coef_[0].tolist(), strict=True)),
        )
    best_score = (-1.0, -1.0)
    chosen = None
    combinations = itertools.product(
        settings["min_match_scores"],
        settings["ambiguity_margins"],
        settings["accept_probabilities"],
        settings["reject_probabilities"],
    )
    for minimum, ambiguity, accept, reject in combinations:
        if reject >= accept:
            continue
        config = cfg.model_copy(deep=True)
        config.matching.min_match_score = minimum
        config.matching.ambiguity_margin = ambiguity
        config.verifier.p_accept, config.verifier.p_reject = accept, reject
        predictions = reviewed_queries(tuning, calibration, config)
        metrics = evaluate(predictions, labels, gallery, cameras, cfg.evaluation.calibration_bins)
        score = (
            metrics["decisions"]["accepted"]["f1"] or 0,
            metrics["decisions"]["accepted"]["precision"] or 0,
        )
        if score > best_score:
            best_score, chosen = score, config
    if chosen is None:
        raise RuntimeError("No valid threshold combination")
    args.out.mkdir(parents=True)
    calibration.save(args.out / "calibrator.json")
    manifest = {
        "source_run": str(args.run),
        "training_sources": training_sources if not args.coefficients else [],
        "frozen_coefficients_source": str(args.coefficients) if args.coefficients else None,
        "method": "scikit-learn LogisticRegression; pretrained neural weights unchanged",
        "split": "SHA-256 identity parity; both training-pair identities must be tuning",
        "training_pairs": len(rows) if not args.coefficients else 0,
        "training_positives": sum(outcomes) if not args.coefficients else 0,
        "training_negatives": len(outcomes) - sum(outcomes) if not args.coefficients else 0,
        "tuning_identity_ids": sorted({labels[q.query_track_id] for q in tuning}),
        "held_out_identity_ids": sorted({labels[q.query_track_id] for q in held_out}),
        "feature_names": names,
        "settings": settings,
        "threshold_objective": "accepted F1, then precision; tuning queries only",
        "assignment_scope": "independently within each query split",
        "pretrained_models_finetuned": False,
    }
    (args.out / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    overrides = {
        "matching": {
            "min_match_score": chosen.matching.min_match_score,
            "ambiguity_margin": chosen.matching.ambiguity_margin,
        },
        "verifier": {"p_accept": chosen.verifier.p_accept, "p_reject": chosen.verifier.p_reject},
    }
    (args.out / "thresholds.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")
    for split, queries in (("tuning", tuning), ("held_out", held_out)):
        predictions = reviewed_queries(queries, calibration, chosen)
        result = evaluate(predictions, labels, gallery, cameras, cfg.evaluation.calibration_bins)
        result.update(
            query_ids=[q.query_track_id for q in queries],
            identity_ids=sorted({labels[q.query_track_id] for q in queries}),
            calibration="logistic regression, identity-separated SHA-256 split",
            excluded_unmapped_queries=sum(q.query_track_id not in labels for q in run.queries),
        )
        (args.out / f"{split}_metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (args.out / f"{split}_queries.json").write_text(
            json.dumps([q.model_dump() for q in predictions], indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "training_pairs": manifest["training_pairs"],
                "training_positives": manifest["training_positives"],
                "tuning_queries": len(tuning),
                "held_out_queries": len(held_out),
                "thresholds": overrides,
            }
        )
    )


if __name__ == "__main__":
    main()
