"""Hand-computed evaluation and export edge cases."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import load_config
from app.evaluation.metrics import evaluate
from app.exporters import export
from app.schemas import (
    Candidate,
    CandidateScores,
    ClockOffset,
    EvidenceItem,
    QueryResult,
    RunResult,
    Verification,
)
from app.verify.calibrator import Calibrator
from scripts.fit_calibrator import reviewed_queries


def query(name: str, target: str, status: str, confidence: float) -> QueryResult:
    candidate = Candidate(
        pair_id=name + "__" + target,
        query_track_id=name,
        candidate_track_id=target,
        rank=1,
        assigned=True,
        ambiguous=False,
        competing_pair_ids=[],
        delta_t_s=10,
        delta_t_adjusted_s=6,
        scores=CandidateScores(visual=1, color=1, temporal=1, direction=1, combined=1),
        verification=Verification(
            status=status,
            confidence=confidence,
            features={},
            evidence=[],
            summary='A comma, a quote " and a newline\nare valid.',
        ),
    )
    return QueryResult(
        query_track_id=name,
        outcome="matched" if status == "accepted" else "uncertain",
        best_pair_id=candidate.pair_id,
        candidates=[candidate],
    )


def test_metrics_hand_calculation() -> None:
    rows = [
        query("a1", "b1", "accepted", 0.8),
        query("a2", "b1", "uncertain", 0.6),
        QueryResult(
            query_track_id="a3", outcome="no_reliable_match", best_pair_id=None, candidates=[]
        ),
    ]
    result = evaluate(
        rows,
        {"a1": "one", "a2": "two", "a3": "three", "b1": "one", "b2": "two"},
        {"A": {"one", "two", "three"}, "B": {"one", "two"}},
        {"a1": "A", "a2": "A", "a3": "A"},
        10,
    )
    assert result["decisions"]["accepted"] == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1,
        "recall": 0.5,
        "f1": 2 / 3,
    }
    assert result["decisions"]["accepted_or_uncertain"]["precision"] == 0.5
    assert result["rank1"] == result["rank5"] == 0.5
    assert result["correct_rejection_rate"] == 1
    assert result["roc_auc"] == 1
    assert result["expected_calibration_error"] == pytest.approx(0.4)
    assert evaluate([], {}, {"A": set(), "B": set()}, {}, 10)["rank1"] is None


def test_unmapped_tracks_are_not_labelled_calibration_negatives() -> None:
    rows = [query("unknown", "b1", "accepted", 0.9), query("a1", "unknown_b", "uncertain", 0.8)]
    result = evaluate(
        rows,
        {"a1": "one", "b1": "one"},
        {"A": {"one"}, "B": {"one"}},
        {"unknown": "A", "a1": "A"},
        10,
    )
    assert result["excluded_unmapped_queries"] == 1
    assert result["excluded_unmapped_calibration_pairs"] == 1
    assert result["queries_without_match"] == 0
    assert result["expected_calibration_error"] is None
    assert result["decisions"]["accepted"]["fp"] == 0
    assert result["rank1"] == 0


def test_calibration_replay_removes_out_of_split_conflict() -> None:
    row = query("a1", "b1", "uncertain", 0.8)
    verification = row.candidates[0].verification
    verification.features = {"margin": 0.2, "track_quality_min": 1}
    verification.evidence = [
        EvidenceItem(
            check="assignment_conflict",
            polarity="contradicts",
            severity="high",
            value=None,
            threshold=None,
            message="another query has a stronger claim",
        )
    ]
    replay = reviewed_queries(
        [row], Calibrator(intercept=3, weights={"margin": 0, "track_quality_min": 0}), load_config()
    )
    assert replay[0].outcome == "matched"
    assert replay[0].candidates[0].verification.evidence == []
    assert verification.evidence[0].check == "assignment_conflict"


def test_csv_quotes_and_empty_query_roundtrip(tmp_path: Path) -> None:
    rows = [
        query("A", "B", "accepted", 0.9),
        QueryResult(
            query_track_id="empty", outcome="no_reliable_match", best_pair_id=None, candidates=[]
        ),
    ]
    run = RunResult(
        run_id="test",
        created_at=datetime.now(UTC),
        config_snapshot={},
        videos={},
        clock_offset=ClockOffset(mode="none", estimated_s=0, n_pairs_used=0, reliable=True),
        tracks_A=[],
        tracks_B=[],
        queries=rows,
        metrics=None,
        timings={},
    )
    export(run, tmp_path)
    assert RunResult.model_validate_json((tmp_path / "results.json").read_bytes()) == run
    with (tmp_path / "candidates.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert exported[0]["summary"] == rows[0].candidates[0].verification.summary
    with (tmp_path / "queries.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream))[1]["best_candidate_track_id"] == ""
    with pytest.raises(FileExistsError):
        export(run, tmp_path)


def test_calibration_roundtrip_and_extreme_scores(tmp_path: Path) -> None:
    calibration = Calibrator(intercept=0, weights={"test": 1})
    path = tmp_path / "coefficients.json"
    calibration.save(path)
    assert Calibrator.load(path) == calibration
    assert calibration.probability({"test": 0}) == 0.5
    assert calibration.probability({"test": -1e6}) == 0
    assert calibration.probability({"test": 1e6}) == 1
    with pytest.raises(ValueError):
        calibration.probability({"other": 1})
    with pytest.raises(FileExistsError):
        calibration.save(path)


@pytest.mark.parametrize("name", ["../calibrator.json", "nested/calibrator.json", "weights.pkl"])
def test_calibrator_setting_must_be_a_local_json_name(name: str) -> None:
    with pytest.raises(ValueError):
        load_config(overrides={"verifier": {"calibrator": name}})


def test_fixture_calibrator_matches_the_feature_contract() -> None:
    directory = Path("configs")
    fixture = Calibrator.load(directory / load_config("dev_fixture").verifier.calibrator)
    default = Calibrator.load(directory / load_config().verifier.calibrator)
    assert fixture.weights.keys() == default.weights.keys() and fixture != default
