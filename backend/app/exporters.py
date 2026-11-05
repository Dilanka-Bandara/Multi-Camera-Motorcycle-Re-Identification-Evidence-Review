"""Persist a validated full run and interoperable CSV review tables."""

import csv
from pathlib import Path

from app.schemas import RunResult

COLUMNS = (
    "run_id query_track_id candidate_track_id rank assigned visual color temporal direction "
    "combined delta_t_s delta_t_adjusted_s ambiguous status confidence supporting_evidence "
    "contradicting_evidence missing_evidence summary"
).split()


def export(result: RunResult, root: Path) -> None:
    with (
        (root / "candidates.csv").open("x", newline="", encoding="utf-8") as cf,
        (root / "queries.csv").open("x", newline="", encoding="utf-8") as qf,
    ):
        candidates = csv.DictWriter(cf, fieldnames=COLUMNS)
        candidates.writeheader()
        queries = csv.writer(qf)
        queries.writerow(["query_track_id", "outcome", "best_candidate_track_id", "confidence"])
        for q in result.queries:
            best = next((c for c in q.candidates if c.pair_id == q.best_pair_id), None)
            queries.writerow(
                [
                    q.query_track_id,
                    q.outcome,
                    best.candidate_track_id if best else None,
                    best.verification.confidence if best and best.verification else None,
                ]
            )
            for c in q.candidates:
                v = c.verification
                row = c.model_dump(
                    include={
                        "query_track_id",
                        "candidate_track_id",
                        "rank",
                        "assigned",
                        "delta_t_s",
                        "delta_t_adjusted_s",
                        "ambiguous",
                    }
                )
                row.update(c.scores.model_dump(), run_id=result.run_id)
                if v:
                    row.update(status=v.status, confidence=v.confidence, summary=v.summary)
                    for column, polarity in (
                        ("supporting_evidence", "supports"),
                        ("contradicting_evidence", "contradicts"),
                        ("missing_evidence", "missing"),
                    ):
                        row[column] = ";".join(
                            f"{e.check}:{e.message}" for e in v.evidence if e.polarity == polarity
                        )
                candidates.writerow(row)
    (root / "results.json").write_text(result.model_dump_json(), encoding="utf-8")
