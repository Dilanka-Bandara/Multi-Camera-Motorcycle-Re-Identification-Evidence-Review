"""Deterministic identity-level partitioning shared by preparation and calibration."""

import hashlib


def tuning_identity(identity: str) -> bool:
    """Assign every appearance of the same identity to the same partition."""
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 2 == 0


def tuning_pair(query_identity: str, candidate_identity: str) -> bool:
    """Keep held-out identities out of both roles in calibration training."""
    return tuning_identity(query_identity) and tuning_identity(candidate_identity)
