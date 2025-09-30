import os, time, math, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from statistics import median
from model import UBCFRecommender


def load_ratings(db_path: str) -> pd.DataFrame:
    eng = create_engine("sqlite:///" + db_path)
    df = pd.read_sql_query("SELECT user_id, movie_id, rating, rated_at FROM ratings", eng)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id","movie_id","rating","rated_at"]).copy()
    df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["rated_at"])
    return df

def time_split_three(df: pd.DataFrame, p_train=0.70, p_val=0.20):
    assert 0 < p_train < 1 and 0 < p_val < 1 and p_train + p_val < 1
    q1 = df["rated_at"].quantile(p_train)
    q2 = df["rated_at"].quantile(p_train + p_val)
    train = df[df["rated_at"] < q1].copy()
    val   = df[(df["rated_at"] >= q1) & (df["rated_at"] < q2)].copy()
    test  = df[df["rated_at"] >= q2].copy()
    return train, val, test, q1, q2

def precision_at_k_for_model(recommend_fn, eval_df: pd.DataFrame, K: int) -> float:
    relevant = eval_df[eval_df["rating"] >= 4]
    users = relevant["user_id"].unique().tolist()
    if not users:
        return 0.0
    precisions = []
    for u in users:
        rel_items = set(map(str, relevant.loc[relevant["user_id"]==u, "movie_id"].unique().tolist()))
        try:
            recs = recommend_fn(user_id=u, top_k=K)
        except Exception:
            continue
        got = set(map(str, getattr(recs, "tolist", lambda: recs)()))
        precisions.append(len(got & rel_items) / K)
    return float(np.mean(precisions)) if precisions else 0.0

def latency_ms(recommend_fn, user_ids, K: int, runs_per_user: int = 1):
    times = []
    if not user_ids:
        return {"p50_ms": None, "p95_ms": None, "n": 0}
    try:
        _ = recommend_fn(user_id=user_ids[0], top_k=K)
    except Exception:
        pass
    for u in user_ids:
        for _ in range(runs_per_user):
            t0 = time.perf_counter_ns()
            _ = recommend_fn(user_id=u, top_k=K)
            times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    if not times:
        return {"p50_ms": None, "p95_ms": None, "n": 0}
    def pct(p):
        i = max(0, min(len(times)-1, int(math.ceil(p*len(times)) - 1)))
        return times[i]
    return {"p50_ms": pct(0.50), "p95_ms": pct(0.95), "n": len(times)}

def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0

def model_size_bytes_ubcf(ubcf: UBCFRecommender) -> int:
    return file_size_bytes(Path(ubcf.model_path))

def ubcf_training_cost_ops(train_df: pd.DataFrame) -> int:
    """
    Training FLOPs proxy for UBCF cosine-sim over mean-centered utility matrix.
    Dense back-of-the-envelope dominant term: ~ 2 * m^2 * n (dot-products).
    """
    m = train_df["user_id"].nunique()
    n = train_df["movie_id"].nunique()
    return 2 * m * m * n

def tune_ubcf(train_df, val_df, K, k_grid):
    """
    Tune UBCF over neighbor count. We re-train once (train_df) since UBCF training
    builds the matrices and similarity; 'k' is used at inference time.
    """
    print(f"[3] Training UBCF once on TRAIN for tuning...", flush=True)
    base_model = UBCFRecommender()
    t0 = time.perf_counter()
    base_model.train(train_df)
    print(f"    UBCF trained in {time.perf_counter()-t0:.2f}s", flush=True)

    best = None
    for i, k_neighbors in enumerate(k_grid, start=1):
        print(f"[3] Tuning {i}/{len(k_grid)} → k_neighbors={k_neighbors} ...", flush=True)
        rec_fn = lambda user_id, top_k: base_model.recommend(user_id=user_id, k=k_neighbors, n_recommendations=top_k)
        p = precision_at_k_for_model(rec_fn, val_df, K)
        print(f"    └─ Precision@{K} (val) = {p:.4f}", flush=True)
        best = max(best or (-1, None, None), (p, k_neighbors, base_model))
    return {"precision_val": best[0], "k_neighbors": best[1]}, best[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="src/db/home_alone.db")
    ap.add_argument("--krec", type=int, default=20)
    ap.add_argument("--p_train", type=float, default=0.70)
    ap.add_argument("--p_val", type=float, default=0.15)
    ap.add_argument("--ubcf_k_grid", default="5,10,20,40,80")
    ap.add_argument("--latency_users", type=int, default=200)
    args = ap.parse_args()

    print(f"[1] Loading ratings from {args.db} ...", flush=True)
    t0 = time.perf_counter()
    df = load_ratings(args.db)
    print(f"    Loaded {len(df):,} rows in {time.perf_counter()-t0:.2f}s", flush=True)

    print(f"[2] Time-based split (train={args.p_train:.2f}, val={args.p_val:.2f}, test={1-args.p_train-args.p_val:.2f}) ...", flush=True)
    train, val, test, q1, q2 = time_split_three(df, args.p_train, args.p_val)
    print(f"    Cutoffs: train_end={q1}, val_end={q2}", flush=True)
    print(f"    Splits:  train={len(train):,}, val={len(val):,}, test={len(test):,}", flush=True)

    # Grid
    ubcf_k_grid = [int(x) for x in str(args.ubcf_k_grid).split(",")]

    print(f"[3] Hyperparameter tuning (UBCF) on validation (K={args.krec}) ...", flush=True)
    tuned, trained_model = tune_ubcf(train, val, args.krec, ubcf_k_grid)
    print(f"    Best on val → k_neighbors={tuned['k_neighbors']}, precision_val={tuned['precision_val']:.4f}", flush=True)

    print("[4] Retraining UBCF on train∪val (building matrices) ...", flush=True)
    train_val = pd.concat([train, val], ignore_index=True)
    ubcf_final = UBCFRecommender()
    t0 = time.perf_counter()
    ubcf_final.train(train_val)
    print(f"    Retrain time: {time.perf_counter()-t0:.2f}s", flush=True)

    print(f"[5] Precision@{args.krec} on TEST ...", flush=True)
    rec_fn_final = lambda user_id, top_k: ubcf_final.recommend(user_id=user_id, k=tuned["k_neighbors"], n_recommendations=top_k)
    t0 = time.perf_counter()
    p_at_k_ubcf = precision_at_k_for_model(rec_fn_final, test, args.krec)
    print(f"    Precision@{args.krec} (test) = {p_at_k_ubcf:.4f} (computed in {time.perf_counter()-t0:.2f}s)", flush=True)

    print("[6] Latency on TEST users present in train∪val ...", flush=True)
    seen_users = pd.Index(train_val["user_id"].unique())
    test_users = pd.Index(test["user_id"].unique())
    latency_users = seen_users.intersection(test_users)
    rng = np.random.default_rng(0)
    sample_users = rng.choice(latency_users.values, size=min(args.latency_users, len(latency_users)), replace=False).tolist() if len(latency_users) else []
    lat = latency_ms(rec_fn_final, sample_users, args.krec, runs_per_user=1)
    print(f"    Latency p50={lat['p50_ms']} ms, p95={lat['p95_ms']} ms over N={lat['n']} calls", flush=True)

    print("[7] Model size & training cost ...", flush=True)
    sz = model_size_bytes_ubcf(ubcf_final)
    ops = ubcf_training_cost_ops(train_val)
    print(f"    Model size: {round(sz/(1024*1024), 2)} MB; Training cost: {round(ops/1e9, 2)} Gops", flush=True)

    def fmt_mb(b): return round(b / (1024*1024), 2)
    def fmt_gops(x): return round(x / 1e9, 2)

    report = {
        "cutoffs": {"train_end": str(q1), "val_end": str(q2)},
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "K": args.krec,
        "tuned": {"UBCF": {
            "precision_val": float(tuned["precision_val"]),
            "k_neighbors": int(tuned["k_neighbors"])
        }},
        "precision_at_K_test": {"UBCF": float(p_at_k_ubcf)},
        "latency_ms_test_known": {"UBCF": {
            "p50_ms": None if lat["p50_ms"] is None else float(lat["p50_ms"]),
            "p95_ms": None if lat["p95_ms"] is None else float(lat["p95_ms"]),
            "n": int(lat["n"])
        }},
        "model_size_MB": {"UBCF": fmt_mb(sz)},
        "training_cost_Gops": {"UBCF": fmt_gops(ops)},
    }

    print("\n===== FINAL REPORT =====")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
