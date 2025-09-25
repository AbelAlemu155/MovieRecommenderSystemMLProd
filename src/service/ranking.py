from functools import lru_cache
from typing import List

_COLD_START: List[str] = [
    "TheMatrix1999","Inception2010","Alien1979","Heat1995","Jaws1975",
    "PulpFiction1994","Se7en1995","Gladiator2000","Casablanca1942","Psycho1960",
    "SpiritedAway2001","Amelie2001","Up2009","Coco2017","Interstellar2014",
    "TheGodfather1972","ToyStory1995","Shrek2001","TheDarkKnight2008","InsideOut2015",
]

@lru_cache(maxsize=100_000)
def _cached(uid: int, k: int) -> List[str]:
    return _COLD_START[:k]

def recommend_for_user(uid: int, k: int = 20) -> List[str]:
    k = min(int(k), 20)
    try:
        return _cached(int(uid), k)
    except Exception:
        return _COLD_START[:k]

def healthcheck():
    return True, "bootstrap"
