"""Independent evidence policy, applied after probability calibration."""

from typing import Literal

from app.config import VerifierConfig
from app.schemas import Candidate, EvidenceItem, QueryResult

Decision = Literal["accepted", "uncertain", "rejected"]


def decide(
    probability: float,
    evidence: list[EvidenceItem],
    ambiguous: bool,
    quality: float,
    config: VerifierConfig,
) -> Decision:
    contradictions = [e for e in evidence if e.polarity == "contradicts"]
    if any(e.severity == "critical" for e in contradictions) or probability < config.p_reject:
        return "rejected"
    if (
        probability >= config.p_accept
        and quality >= config.min_track_quality
        and not ambiguous
        and not any(e.severity == "high" or e.check == "keypoint_inliers" for e in contradictions)
    ):
        return "accepted"
    return "uncertain"


def summarize(status: Decision, evidence: list[EvidenceItem]) -> str:
    priorities = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ordered = sorted(evidence, key=lambda e: priorities[e.severity])
    supporting = [e.message for e in ordered if e.polarity == "supports"][:2]
    opposing = [e.message for e in ordered if e.polarity == "contradicts"][:2]
    reasons = ", ".join(supporting + opposing) or "available evidence is limited"
    return f"{status.capitalize()} because {reasons}."


def resolve(query_id: str, row: list[Candidate]) -> QueryResult:
    assigned = next((c for c in row if c.assigned), None)
    if assigned and assigned.verification and assigned.verification.status == "accepted":
        return QueryResult(
            query_track_id=query_id,
            candidates=row,
            best_pair_id=assigned.pair_id,
            outcome="matched",
        )
    reviewed = [c for c in row if c.verification]
    if assigned and reviewed and reviewed[0].verification.status == "uncertain":
        return QueryResult(
            query_track_id=query_id,
            candidates=row,
            best_pair_id=reviewed[0].pair_id,
            outcome="uncertain",
        )
    return QueryResult(
        query_track_id=query_id, candidates=row, best_pair_id=None, outcome="no_reliable_match"
    )
