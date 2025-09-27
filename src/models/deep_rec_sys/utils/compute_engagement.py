import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import os


# --------------------------------------------------------
# 2) Engagement from watch_history + ratings  -> engagement.npz
# --------------------------------------------------------
def compute_engagement_from_watch_and_ratings(
    watch_csv: str,
    ratings_csv: str,
    out_npz: str,
    *,
    today: Optional[datetime] = None,
    max_minutes: int = 180,   # cap watch duration at 3h
    tau_days: float = 30.0,   # recency decay
    w_duration: float = 0.4,  # weights sum to 1.0
    w_recency: float = 0.3,
    w_rating: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce one engagement score per (user_id, movie_id) by combining:
      - duration_score ∈ [0,1]
      - recency_score  ∈ [0,1]
      - rating_score   ∈ [0,1]

    Final: engagement = 0.4*duration + 0.3*recency + 0.3*rating  (simple, interpretable).
    Saves NPZ with arrays: user_ids, movie_ids, engagement (float32).
    If multiple rows per (u,i) in watch_history/ratings, aggregates to a single row.
    """
    if today is None:
        today = datetime.now()

    # --- Read watch history ---
    dw = pd.read_csv(watch_csv, parse_dates=["start_time", "last_watch"])
    # compute per-row duration & recency
    dw["last_watch"] = pd.to_datetime(dw["last_watch"], format='mixed')
    dw["start_time"] = pd.to_datetime(dw["start_time"], format='mixed')
    duration_minutes = (dw["last_watch"] - dw["start_time"]).dt.total_seconds().clip(lower=0) / 60.0
    duration_score = np.clip(duration_minutes, 0, max_minutes) / float(max_minutes)

    days_since = (today - dw["last_watch"]).dt.days.clip(lower=0)
    recency_score = np.exp(-days_since / tau_days)

    dw["duration_score"] = duration_score
    dw["recency_score"] = recency_score

    # Collapse multiple watch rows for same (u,i): take the MAX signal observed (simple & optimistic)
    w_agg = dw.groupby(["user_id", "movie_id"], as_index=False)[["duration_score", "recency_score"]].max()

    # --- Read ratings (optional; many pairs may have no rating) ---
    dr = pd.read_csv(ratings_csv)
    # normalize rating ∈ {1..5} -> [0,1]
    if "rating" not in dr.columns:
        raise ValueError("ratings_csv must have a 'rating' column.")
    dr["rating_score"] = (dr["rating"].astype(float) - 1.0) / 4.0
    r_agg = dr.groupby(["user_id", "movie_id"], as_index=False)["rating_score"].mean()

    # --- Join both on (user_id, movie_id); missing rating -> 0 (no signal) ---
    df = pd.merge(w_agg, r_agg, on=["user_id", "movie_id"], how="left")
    df["rating_score"] = df["rating_score"].fillna(0.0)

    # --- Final engagement in [0,1] ---
    # NOTE: we’re not re-normalizing; weights already sum to 1.
    df["engagement"] = (
        w_duration * df["duration_score"].astype(float)
        + w_recency * df["recency_score"].astype(float)
        + w_rating  * df["rating_score"].astype(float)
    ).clip(lower=0.0, upper=1.0)

    # Save to NPZ
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        out_npz,
        user_ids=df["user_id"].to_numpy(),
        movie_ids=df["movie_id"].to_numpy(),
        engagement=df["engagement"].astype(np.float32).to_numpy(),
    )

    return (
        df["user_id"].to_numpy(),
        df["movie_id"].to_numpy(),
        df["engagement"].astype(np.float32).to_numpy(),
    )

if __name__ == "__main__":
    out_engagements_npz = "src/models/deep_rec_sys/features/engagements.npz"
    os.makedirs(os.path.dirname(out_engagements_npz), exist_ok=True)

    user_ids, movie_ids, engagement = compute_engagement_from_watch_and_ratings(
        watch_csv="src/db/watch_history.csv",
        ratings_csv="src/db/ratings.csv",
        out_npz=out_engagements_npz,
        today=datetime.now(),
        max_minutes=180,
        tau_days=30.0,
        w_duration=0.4,
        w_recency=0.3,
        w_rating=0.3,
    )
    print(f"Computed engagement for {len(engagement)} (user,movie) pairs.")