import json
import math
import os
from pathlib import Path

import pytest

from app.config import load_config
from app.matching.assignment import assign
from app.matching.candidates import rank_direction
from app.matching.temporal import estimate_clock, transit_score
from app.pipeline.representation import load_representation
from app.schemas import Candidate, CandidateScores, Track


def test_clock_and_window() -> None:
    cfg = load_config("dev_fixture")
    opt = cfg.topology.clock_offset
    assert not estimate_clock([14], opt).reliable
    assert not estimate_clock(
        [15] * 4, opt.model_copy(update={"expected_transit_s": None})
    ).reliable
    clock = estimate_clock([14.5, 15.5, 16.5], opt)
    assert clock.reliable and clock.estimated_s == 4
    assert estimate_clock([6.5, 7.5, 8.5], opt, -1).estimated_s == 4
    for edge in (cfg.topology.min_transit_s, cfg.topology.max_transit_s):
        assert transit_score(edge, clock, cfg.topology, cfg.matching) == (1, False)
    outside = cfg.topology.max_transit_s + cfg.matching.temporal_soft_tau_s
    assert transit_score(outside, clock, cfg.topology, cfg.matching)[0] == pytest.approx(
        math.exp(-1)
    )
    assert transit_score(-3, clock, cfg.topology, cfg.matching) == (0, True)
    assert transit_score(outside, estimate_clock([], opt), cfg.topology, cfg.matching) == (1, False)


def test_hungarian_dummy_and_conflict() -> None:
    cfg = load_config().matching

    def row(name: str, score: float, delta: float) -> list[Candidate]:
        return [
            Candidate(
                pair_id=name + "-b",
                query_track_id=name,
                candidate_track_id="b",
                rank=1,
                assigned=False,
                ambiguous=False,
                competing_pair_ids=[],
                delta_t_s=delta,
                delta_t_adjusted_s=delta,
                scores=CandidateScores(visual=1, direction=1, color=1, temporal=0, combined=score),
            )
        ]

    rows = [row("first", 0.96, 10), row("weak", 0.4, 10), row("impossible", 1, -10), []]
    assign(rows, cfg)
    assert [r[0].assigned for r in rows if r] == [True, False, False]
    assert rows[0][0].ambiguous and "weak-b" in rows[0][0].competing_pair_ids


def test_dev_fixture_matching_acceptance() -> None:
    run = Path(os.environ.get("MOTO_PHASE4_INPUT", "data/user/runs/phase3-validation"))
    if not (run / "representations").is_dir():
        pytest.skip("Generate a full dev fixture run to evaluate matching")
    a, b = (
        [Track.model_validate(t) for t in json.loads((run / f"tracks_{c}.json").read_text())]
        for c in ("A", "B")
    )
    features = {
        t.track_id: load_representation(run / "representations" / f"{t.track_id}.npz")
        for t in a + b
    }
    clock, rows = rank_direction(a, b, features, load_config("dev_fixture"))
    assert clock.reliable and clock.estimated_s == pytest.approx(4, abs=1.5)
    # Only the evaluator sees the ground-truth detector's original identity IDs.
    candidates = {t.raw_track_ids[0]: rows[t.track_id] for t in a}
    labels = {t.track_id: t.raw_track_ids[0] for t in b}
    assert sum(labels[candidates[i][0].candidate_track_id] == i for i in range(9)) / 9 >= 0.8
    assert sum(not any(c.assigned for c in candidates[i]) for i in (9, 10, 11)) >= 2

    def flagged(left: int, right: int) -> bool:
        best = candidates[left][0]
        competitor = next(c for c in candidates[left] if labels[c.candidate_track_id] == right)
        return best.ambiguous and competitor.pair_id in best.competing_pair_ids

    # The guide requires both lookalike pairs to be flagged; one direction suffices.
    for first, second in ((0, 1), (2, 3)):
        assert flagged(first, second) or flagged(second, first)
