"""
Ranking logic for the prediction service.

Right now this is a *stub* that always returns a static, safe list.
This allows you to meet the strict response contract and latency SLO
while you build real models. Later, replace `recommend_for_user`
to call your loaded model(s) and remove the static list.

Design notes:
- Keep networking and I/O out of the request path -> low latency.
- Cache per-user results if you precompute or have deterministic output.
- Always return <= 20 IDs, no trailing comma or newline.
"""

from functools import lru_cache
from typing import List

# Safe/popular fallback list used for cold start and until models are wired.
# Replace with something derived from your data once ready.
_COLD_START: List[str] = [
    "TheMatrix1999", "Inception2010", "Alien1979", "Heat1995", "Jaws1975",
    "PulpFiction1994", "Se7en1995", "Gladiator2000", "Casablanca1942", "Psycho1960",
    "SpiritedAway2001", "Amelie2001", "Up2009", "Coco2017", "Interstellar2014",
    "TheGodfather1972", "ToyStory1995", "Shrek2001", "TheDarkKnight2008", "InsideOut2015",
]

# Example: placeholders for model and ID maps you'll load at process start.
_MODEL = None          # e.g., your CF/Content/Hybrid model object
_USER_IDMAP = None     # e.g., mapping from external user_id -> internal index
_MOVIE_IDMAP = None    # e.g., reverse map from internal index -> external movie_id


@lru_cache(maxsize=100_000)
def _cached(uid: int, k: int) -> List[str]:
    """
    Optional cache layer keyed by (uid, k) to avoid recomputation for hot users.
    Adjust maxsize to your memory budget; consider invalidation on model updates.
    """
    # With a real model, compute recommendations here and return movie IDs (strings).
    # For now, return the first k from the cold-start list.
    return _COLD_START[:k]


def recommend_for_user(uid: int, k: int = 20) -> List[str]:
    """
    Core recommendation entrypoint used by the HTTP layer.

    Args:
        uid: External numeric user ID from the request path.
        k:   Upper bound on number of movie IDs to return (max 20 per spec).

    Returns:
        A list of up to k movie IDs (strings), already ordered high -> low.
    """
    # Defensive: never exceed 20 to satisfy the course contract.
    k = min(int(k), 20)

    try:
        # If you have a model, this is where you'd:
        # 1) map `uid` to an internal index (using _USER_IDMAP),
        # 2) get scores/top-K from _MODEL,
        # 3) map internal item IDs back to external movie IDs via _MOVIE_IDMAP,
        # 4) deduplicate and enforce business rules (e.g., max per franchise).
        return _cached(int(uid), k)
    except Exception:
        # Fail-safe: always return something valid rather than erroring.
        return _COLD_START[:k]


def healthcheck():
    """
    Health status for /healthz.

    Returns:
        (ok: bool, msg: str) indicating whether critical resources are loaded.
        Upgrade this to verify model/ID maps, e.g.:
            ok = _MODEL is not None and _USER_IDMAP is not None and _MOVIE_IDMAP is not None
    """
    ok = True if _COLD_START else False
    msg = "bootstrap" if ok else "no-fallback-list"
    return ok, msg
