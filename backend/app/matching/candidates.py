"""Build ranked cross-camera proposals from tracks and appearance features."""

import numpy as np

from app.config import AppConfig
from app.matching.assignment import assign
from app.matching.similarity import color_similarity, set_similarity
from app.matching.temporal import direction_score, estimate_clock, transit_score
from app.schemas import Candidate, CandidateScores, ClockOffset, Track, TrackRepresentation


def rank_direction(
    queries: list[Track],
    targets: list[Track],
    features: dict[str, TrackRepresentation],
    cfg: AppConfig,
    reverse: bool = False,
    clock: ClockOffset | None = None,
) -> tuple[ClockOffset, dict[str, list[Candidate]]]:
    """Rank and assign a single camera direction, preserving empty query rows."""
    opt, topology = cfg.matching, cfg.topology
    sign = -1 if reverse else 1
    visual = np.empty((len(queries), len(targets)))
    for i, q in enumerate(queries):
        for j, t in enumerate(targets):
            visual[i, j] = set_similarity(
                features[q.track_id].emb_full, features[t.track_id].emb_full, opt.set_sim_top_m
            )
    deltas: list[float] = []
    if visual.size:
        best_queries = np.argmax(visual, axis=0)
        for i, j in enumerate(np.argmax(visual, axis=1)):
            if best_queries[j] == i and visual[i, j] >= topology.clock_offset.confident_visual_sim:
                deltas.append(targets[j].t_start_s - queries[i].t_end_s)
    clock = clock or estimate_clock(deltas, topology.clock_offset, sign)
    weights = opt.weights
    rows: dict[str, list[Candidate]] = {}
    for i, q in enumerate(queries):
        row: list[Candidate] = []
        for j, t in enumerate(targets):
            raw = t.t_start_s - q.t_end_s
            adjusted = raw - sign * clock.estimated_s
            timing, impossible = transit_score(adjusted, clock, topology, opt)
            appearance = float(visual[i, j])
            if impossible and appearance < opt.keep_impossible_visual:
                continue
            color = color_similarity(
                features[q.track_id].color_hist, features[t.track_id].color_hist
            )
            a, b = (t, q) if reverse else (q, t)
            direction = (
                direction_score(a.motion_vector, topology.expected_exit_dir_A)
                + direction_score(b.motion_vector, topology.expected_entry_dir_B)
            ) / 2
            combined = (
                weights.visual * appearance
                + weights.color * color
                + weights.temporal * timing
                + weights.direction * direction
            )
            row.append(
                Candidate(
                    pair_id=f"{q.track_id}__{t.track_id}",
                    query_track_id=q.track_id,
                    candidate_track_id=t.track_id,
                    rank=1,
                    delta_t_s=raw,
                    delta_t_adjusted_s=adjusted,
                    scores=CandidateScores(
                        visual=appearance,
                        color=color,
                        temporal=timing,
                        direction=direction,
                        combined=min(1.0, combined),
                    ),
                    assigned=False,
                    ambiguous=False,
                    competing_pair_ids=[],
                )
            )
        row.sort(key=lambda c: (-c.scores.combined, c.candidate_track_id))
        rows[q.track_id] = row[: opt.top_n_candidates]
        for rank, candidate in enumerate(rows[q.track_id], start=1):
            candidate.rank = rank
    assign(list(rows.values()), opt)
    return clock, rows
