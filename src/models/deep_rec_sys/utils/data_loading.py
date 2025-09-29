import numpy as np
from typing import Dict, Any, Optional

def load_user_features_npz(path: str) -> Dict[str, Any]:
    """
    Expects NPZ keys: X (float32, [N_users, D_u]), ids ([N_users]), feature_names (optional)
    Returns: { "X", "ids", "feature_names" }
    """
    data = np.load(path, allow_pickle=True)
    for k in ("X", "ids"):
        if k not in data.files:
            raise KeyError(f"{path} missing required key '{k}'")
    X = data["X"].astype(np.float32, copy=False)
    ids = data["ids"]
    feat_names: Optional[np.ndarray] = data["feature_names"] if "feature_names" in data.files else None
    return {"X": X, "ids": ids, "feature_names": feat_names}

def load_movie_features_npz(path: str) -> Dict[str, Any]:
    """
    Expects NPZ keys: X (float32, [N_items, D_i]), ids ([N_items]), feature_names (optional)
    Returns: { "X", "ids", "feature_names" }
    """
    data = np.load(path, allow_pickle=True)
    for k in ("X", "ids"):
        if k not in data.files:
            raise KeyError(f"{path} missing required key '{k}'")
    X = data["X"].astype(np.float32, copy=False)
    ids = data["ids"]
    feat_names: Optional[np.ndarray] = data["feature_names"] if "feature_names" in data.files else None
    return {"X": X, "ids": ids, "feature_names": feat_names}

def load_engagements_npz(path: str) -> Dict[str, Any]:
    """
    Expects NPZ keys: user_ids, movie_ids, engagement (float32, [N_rows])
    Returns: { "user_ids", "movie_ids", "engagement" }
    """
    data = np.load(path, allow_pickle=True)
    for k in ("user_ids", "movie_ids", "engagement"):
        if k not in data.files:
            raise KeyError(f"{path} missing required key '{k}'")
    user_ids = data["user_ids"]
    movie_ids = data["movie_ids"]
    engagement = data["engagement"].astype(np.float32, copy=False)
    if len(user_ids) != len(movie_ids) or len(user_ids) != len(engagement):
        raise ValueError("engagements.npz arrays must have same length")
    return {"user_ids": user_ids, "movie_ids": movie_ids, "engagement": engagement}

def quick_summary(u: Dict[str, Any], m: Dict[str, Any], e: Dict[str, Any]) -> str:
    return (
        f"Users: X {u['X'].shape}, ids {len(u['ids'])}\n"
        f"Movies: X {m['X'].shape}, ids {len(m['ids'])}\n"
        f"Engagement rows: {len(e['engagement'])}"
    )

def peek_ids(users, movies, eng, n=10):
    u_ids = users["ids"]
    m_ids = movies["ids"]
    eu = eng["user_ids"]
    em = eng["movie_ids"]

    print("types:", type(u_ids[0]), type(m_ids[0]), type(eu[0]), type(em[0]))
    print("dtype:", u_ids.dtype, m_ids.dtype, eu.dtype, em.dtype)

    u_set = set(map(str, u_ids.tolist()))
    m_set = set(map(str, m_ids.tolist()))
    eu_set = set(map(str, eu.tolist()))
    em_set = set(map(str, em.tolist()))

    print("users ∩ eng.user_ids:", len(u_set & eu_set))
    print("movies ∩ eng.movie_ids:", len(m_set & em_set))

    if len(u_set & eu_set) == 0:
        only_eu = list(eu_set - u_set)[:n]
        only_u  = list(u_set - eu_set)[:n]
        print("e.user_ids not in users.ids:", only_eu)
        print("users.ids not in e.user_ids:", only_u)

    if len(m_set & em_set) == 0:
        only_em = list(em_set - m_set)[:n]
        only_m  = list(m_set - em_set)[:n]
        print("e.movie_ids not in movies.ids:", only_em)
        print("movies.ids not in e.movie_ids:", only_m)