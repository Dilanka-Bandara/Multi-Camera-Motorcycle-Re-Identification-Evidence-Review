"""Evaluate query decisions, retrieval and calibration separately."""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from app.schemas import QueryResult


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def evaluate(
    queries: list[QueryResult],
    labels: dict[str, str],
    gallery: dict[str, set[str]],
    cameras: dict[str, str],
    bins: int,
) -> dict[str, Any]:
    positives = rank1 = rank5 = absent = rejected = 0
    counts = {"accepted": [0, 0], "accepted_or_uncertain": [0, 0]}
    truth, confidence, false_matches = [], [], []
    for q in queries:
        ident = labels.get(q.query_track_id)
        other = "B" if cameras[q.query_track_id] == "A" else "A"
        has_match = ident in gallery[other]
        positives += int(has_match)
        ranked = [
            ident is not None and labels.get(c.candidate_track_id) == ident for c in q.candidates
        ]
        rank1 += int(has_match and any(ranked[:1]))
        rank5 += int(has_match and any(ranked[:5]))
        absent += int(not has_match)
        rejected += int(not has_match and q.outcome != "matched")
        for c, correct in zip(q.candidates, ranked, strict=True):
            if c.verification:
                truth.append(int(correct))
                confidence.append(c.verification.confidence)
            if q.best_pair_id == c.pair_id:
                for mode, tally in counts.items():
                    if q.outcome == "matched" or (
                        mode == "accepted_or_uncertain" and q.outcome == "uncertain"
                    ):
                        tally[0 if correct else 1] += 1
                if not correct and q.outcome == "matched":
                    false_matches.append(c.model_dump())
    calibration = 0.0
    probs = np.asarray(confidence)
    for idx in range(bins):
        chosen = np.minimum((probs * bins).astype(int), bins - 1) == idx
        if chosen.any():
            calibration += chosen.mean() * abs(
                probs[chosen].mean() - np.asarray(truth)[chosen].mean()
            )
    result = {}
    for mode, (tp, fp) in counts.items():
        result[mode] = {
            "tp": tp,
            "fp": fp,
            "fn": positives - tp,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, positives),
            "f1": ratio(2 * tp, positives + tp + fp),
        }
    return dict(
        decisions=result,
        rank1=ratio(rank1, positives),
        rank5=ratio(rank5, positives),
        correct_rejection_rate=ratio(rejected, absent),
        false_matches=false_matches,
        roc_auc=float(roc_auc_score(truth, probs)) if len(set(truth)) == 2 else None,
        expected_calibration_error=float(calibration) if truth else None,
        queries_with_match=positives,
        queries_without_match=absent,
    )
