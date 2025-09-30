import os, time, math, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from statistics import median
from mf import MatrixFactorizationRecommender
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_ratings(db_path: str) -> pd.DataFrame:
    logging.info(f"Loading ratings from database: {db_path}")
    eng = create_engine("sqlite:///" + db_path)
    df = pd.read_sql_query("SELECT user_id, movie_id, rating, rated_at FROM ratings", eng)
    initial_len = len(df)
    logging.info(f"Read {initial_len} rows from database.")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id","movie_id","rating","rated_at"]).copy()
    logging.info(f"After dropping NaNs in core columns: {len(df)} rows remain.")

    df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["rated_at"])
    logging.info(f"After dropping NaNs in 'rated_at': {len(df)} rows remain.")
    
    return df

def time_split_three(df: pd.DataFrame, p_train=0.70, p_val=0.20):
    assert 0 < p_train < 1 and 0 < p_val < 1 and p_train + p_val < 1
    logging.info(f"Splitting data temporally with p_train={p_train}, p_val={p_val}")

    q1 = df["rated_at"].quantile(p_train)
    q2 = df["rated_at"].quantile(p_train + p_val)
    
    train = df[df["rated_at"] < q1].copy()
    val   = df[(df["rated_at"] >= q1) & (df["rated_at"] < q2)].copy()
    test  = df[df["rated_at"] >= q2].copy()
    
    logging.info(f"Split points: Train end {q1}, Validation end {q2}")
    logging.info(f"Split sizes: Train={len(train)}, Validation={len(val)}, Test={len(test)}")

    return train, val, test, q1, q2

def precision_at_k_for_model(model, eval_df: pd.DataFrame, K: int) -> float:
    relevant = eval_df[eval_df["rating"] >= 4]
    users = relevant["user_id"].unique().tolist()
    if not users:
        return 0.0
    precisions = []
    print(len(users))
    for u in users:
        rel_items = set(map(str, relevant.loc[relevant["user_id"]==u, "movie_id"].unique().tolist()))
        try:
            recs = model.recommend(user_id=u, top_k=K)
        except Exception:
            continue
        got = set(map(str, getattr(recs, "tolist", lambda: recs)()))
        precisions.append(len(got & rel_items) / K)
    return float(np.mean(precisions)) if precisions else 0.0

def latency_ms(model, user_ids, K: int, runs_per_user: int = 1):
    times = []
    if not user_ids:
        return {"p50_ms": None, "p95_ms": None, "n": 0}
    
    try:
        _ = model.recommend(user_id=user_ids[0], top_k=K)
    except Exception:
        pass

    for u in user_ids:
        for _ in range(runs_per_user):
            t0 = time.perf_counter_ns()
            _ = model.recommend(user_id=u, top_k=K)
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

def model_size_bytes_mf(mf: MatrixFactorizationRecommender) -> int:
    s = 0
    s += file_size_bytes(mf.factors_path)
    s += file_size_bytes(mf.maps_path)
    s += file_size_bytes(mf.popularity_path)
    return s

def mf_training_cost_ops(train_df: pd.DataFrame, k: int, iters: int) -> int:
    m = train_df["user_id"].nunique()
    n = train_df["movie_id"].nunique()
    R = len(train_df)
    ops = iters * (4*k*R + 2*(m+n)*(k**2) + (m+n)*int(round((k**3)/3)))
    return ops

def tune_mf(train_df, val_df, K, grid_k, grid_iters):
    logging.info(f"Starting MF hyperparameter tuning on validation set (K={K})")
    logging.info(f"k grid: {grid_k}, iters grid: {grid_iters}")
    best = None
    total_runs = len(grid_k) * len(grid_iters)
    current_run = 0

    for k in grid_k:
        for iters in grid_iters:
            current_run += 1
            logging.info(f"Tuning run {current_run}/{total_runs}: k={k}, iters={iters}")
            
            # Start timer for training
            t0_train = time.perf_counter()
            model = MatrixFactorizationRecommender(k=k, iters=iters)
            model.train(train_df)
            train_time = time.perf_counter() - t0_train
            logging.info(f"Training complete in {train_time}s.")
            
            # Evaluate
            t0_eval = time.perf_counter()
            p = precision_at_k_for_model(model, val_df, K)
            eval_time = time.perf_counter() - t0_eval
            logging.info(f"Evaluation complete in {eval_time}s. Precision@{K}: {p}")

            if best is None or p > best[0]:
                best = (p, k, iters)
                logging.info(f"--> NEW BEST: Precision@{K}={best[0]} with k={best[1]}, iters={best[2]}")

    logging.info(f"Tuning finished. Best params: k={best[1]}, iters={best[2]} with Precision@{K}={best[0]}")
    return {"precision_val": best[0], "k": best[1], "iters": best[2]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="src/db/home_alone.db")
    ap.add_argument("--krec", type=int, default=20)
    ap.add_argument("--p_train", type=float, default=0.70)
    ap.add_argument("--p_val", type=float, default=0.15)
    ap.add_argument("--mf_k_grid", default="32,64,128")
    ap.add_argument("--mf_iters_grid", default="5,10")
    ap.add_argument("--latency_users", type=int, default=200)
    args = ap.parse_args()

    logging.info("Starting evaluation script.")
    
    df = load_ratings(args.db)
    train, val, test, q1, q2 = time_split_three(df, args.p_train, args.p_val)

    # Grids
    mf_k_grid = [int(x) for x in str(args.mf_k_grid).split(",")]
    mf_iters_grid = [int(x) for x in str(args.mf_iters_grid).split(",")]

    mf_tuned  = tune_mf(train, val, args.krec, mf_k_grid, mf_iters_grid)

    logging.info("Retraining MF model on TRAIN + VAL with best hyperparameters.")
    train_val = pd.concat([train, val], ignore_index=True)
    logging.info(f"Train+Val size: {len(train_val)} ratings.")

    mf_final = MatrixFactorizationRecommender(k=mf_tuned["k"], iters=mf_tuned["iters"])
    t0_final_train = time.perf_counter()
    mf_final.train(train_val)
    final_train_time = time.perf_counter() - t0_final_train
    logging.info(f"Final MF training complete in {final_train_time}s.")

    logging.info("Loading maps and means from disk after training.")
    mf_final.load()

    logging.info(f"Evaluating Precision@{args.krec} on TEST set (N={len(test)}).")
    p_at_k_mf = precision_at_k_for_model(mf_final, test, args.krec)
    logging.info(f"MF Precision@{args.krec} on test: {p_at_k_mf}")

    logging.info("Calculating recommendation latency.")
    seen_users = pd.Index(train_val["user_id"].unique())
    test_users = pd.Index(test["user_id"].unique())
    latency_users = seen_users.intersection(test_users)
    rng = np.random.default_rng(0)
    sample_users = rng.choice(latency_users.values, size=min(args.latency_users, len(latency_users)), replace=False).tolist() if len(latency_users) else []
    
    logging.info(f"Sampling {len(sample_users)} users for latency measurement (requested: {args.latency_users}).")
    lat_mf    = latency_ms(mf_final, sample_users, args.krec, runs_per_user=1)
    logging.info(f"MF Latency (p50/p95): {lat_mf['p50_ms']}ms / {lat_mf['p95_ms']}ms (N={lat_mf['n']} runs).")

    sz_mf    = model_size_bytes_mf(mf_final)
    logging.info(f"MF Model size: {sz_mf} bytes.")

    ops_mf    = mf_training_cost_ops(train_val, mf_tuned["k"], mf_tuned["iters"])
    logging.info(f"MF Training Cost (ops): {ops_mf}.")
    
    logging.info("Cleaned up final MF model files.")
    def fmt_mb(b): return round(b / (1024*1024), 2)
    def fmt_gops(x): return round(x / 1e9, 2)

    report = {
        "cutoffs": {"train_end": str(q1), "val_end": str(q2)},
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "K": args.krec,
        "tuned": {"MF": mf_tuned},
        "precision_at_K_test": {"MF": p_at_k_mf},
        "latency_ms_test_known": {"MF": lat_mf},
        "model_size_MB": {"MF": fmt_mb(sz_mf)},
        "training_cost_Gops": {"MF": fmt_gops(ops_mf)},
    }
    
    logging.info("--- Final Report ---")
    print(json.dumps(report, indent=2))
    logging.info("Script finished successfully.")

if __name__ == "__main__":
    main()