"""One-to-one proposals with explicit unmatched alternatives and ambiguity evidence."""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.config import MatchingConfig
from app.schemas import Candidate


def assign(rows: list[list[Candidate]], cfg: MatchingConfig) -> None:
    if not rows:
        return
    columns = sorted({c.candidate_track_id for row in rows for c in row})
    index = {key: i for i, key in enumerate(columns)}
    costs = np.full((len(rows), len(columns) + len(rows)), np.inf)
    leaders: dict[str, list[Candidate]] = defaultdict(list)
    for i, row in enumerate(rows):
        costs[i, len(columns) + i] = 1 - cfg.min_match_score
        if row:
            leaders[row[0].candidate_track_id].append(row[0])
        close = [
            c for c in row if row[0].scores.combined - c.scores.combined <= cfg.ambiguity_margin
        ]
        for candidate in row:
            candidate.assigned = False
            candidate.competing_pair_ids = [
                c.pair_id for c in close if candidate in close and c is not candidate
            ]
            candidate.ambiguous = bool(candidate.competing_pair_ids)
            if candidate.delta_t_adjusted_s >= cfg.impossible_delta_s:
                costs[i, index[candidate.candidate_track_id]] = 1 - candidate.scores.combined
    for group in leaders.values():
        for c in group:
            peers = {other.pair_id for other in group if other is not c}
            c.competing_pair_ids = sorted(set(c.competing_pair_ids) | peers)
            c.ambiguous = bool(c.competing_pair_ids)
    for i, j in zip(*linear_sum_assignment(costs), strict=True):
        if j < len(columns):
            next(c for c in rows[i] if c.candidate_track_id == columns[j]).assigned = True
