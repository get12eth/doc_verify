from collections.abc import Iterable

from .schemas import Decision


def decide(score: float, match_at: float, review_at: float) -> Decision:
    """Three-way decision: confident match, human review band, confident non-match."""
    if score >= match_at:
        return "match"
    if score >= review_at:
        return "review"
    return "no_match"


def combine(decisions: Iterable[Decision]) -> Decision:
    """Conservative aggregation: any hard non-match wins; a match needs every check to match."""
    ds = list(decisions)
    if all(d == "unavailable" for d in ds):
        return "unavailable"
    if "no_match" in ds:
        return "no_match"
    if "unavailable" in ds:
        return "review"
    return "match" if all(d == "match" for d in ds) else "review"
