from __future__ import annotations
import argparse, importlib, sqlite3, json
from typing import Callable, Dict, Iterable, Set, Any, List, Tuple
import pandas as pd
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, average_precision, hit_rate_at_k

def _ratings(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    try: return pd.read_sql_query("SELECT user_id, movie_id, rating, rated_at FROM ratings", con)
    finally: con.close()
# Reads the ratings table with the columns

def time_split(df: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce", utc=True)
    tr = df[df["rated_at"] < pd.Timestamp(train_end, tz="UTC")]
    va = df[(df["rated_at"] >= pd.Timestamp(train_end, tz="UTC")) & (df["rated_at"] < pd.Timestamp(val_end, tz="UTC"))]
    te = df[df["rated_at"] >= pd.Timestamp(val_end, tz="UTC")]
    return tr, va, te
# Splits the ratings DataFrame into train, validation, and test sets based on the timestamp

def _truths(df: pd.DataFrame, thr: float = 4.0) -> Dict[int, Set[str]]:
    rel = df[df["rating"] >= thr]
    return {int(u): set(g["movie_id"].astype(str)) for u, g in rel.groupby("user_id")}
# Generates a dictionary mapping user IDs to sets of relevant movie IDs based on a rating threshold

def _evaluate(preds: Dict[int, List[str]], truths: Dict[int, Set[str]], k: int) -> Dict[str, Any]:
    users = [u for u in truths if truths[u]]
    if not users: return {"precision":0,"recall":0,"ndcg":0,"hit_rate":0,"map":0,"users":0}
    P = [precision_at_k(preds.get(u, []), truths[u], k) for u in users]
    R = [recall_at_k(preds.get(u, []), truths[u], k) for u in users]
    N = [ndcg_at_k(preds.get(u, []), truths[u], k) for u in users]
    H = [hit_rate_at_k(preds.get(u, []), truths[u], k) for u in users]
    M = [average_precision(preds.get(u, []), truths[u], k) for u in users]
    n = len(users)
    return {"precision":sum(P)/n,"recall":sum(R)/n,"ndcg":sum(N)/n,"hit_rate":sum(H)/n,"map":sum(M)/n,"users":n}
# Computes evaluation metrics by comparing predicted recommendations against ground truth for each user
# Inputs: fn (recommender function), test (ratings DataFrame), k (top-k cutoff), thr (relevance threshold); Output: dict of averaged metrics.

def evaluate(fn: Callable[[int, int], Iterable[str]], test: pd.DataFrame, k: int = 20, thr: float = 4.0) -> Dict[str, Any]:
    truths = _truths(test, thr)
    preds = {}
    for u in truths: 
        try: preds[u] = list(map(str, fn(u, k)))[:k]
        except: preds[u] = []
    return _evaluate(preds, truths, k)
# (Model agnostic )Evaluates a recommender function on the test set and returns the computed metrics

def _dummy(u: int, k: int): return [str(i) for i in range(1, k + 1)]
# A dummy recommender function that returns the first k movie IDs as strings

def _resolve(o: Any) -> Callable[[int, int], Iterable[str]]:
    if callable(o) and getattr(o, "__name__", "") == "recommend": return o
    if hasattr(o, "recommend"): return getattr(o, "recommend")
    if isinstance(o, type) and hasattr(o, "recommend"): return getattr(o(), "recommend")
    raise ValueError("Invalid recommender")
# Resolves an object to a recommender function

def _load(dotted: str) -> Callable[[int, int], Iterable[str]]:
    if not dotted: return _dummy
    if ":" in dotted:
        m, s = dotted.split(":", 1); mod = importlib.import_module(m)
        for p in s.split("."): mod = getattr(mod, p)
        return _resolve(mod)
    d = dotted.rfind(".")
    if d == -1:
        mod = importlib.import_module(dotted)
        return _resolve(getattr(mod, "recommend"))
    m, a = dotted[:d], dotted[d+1:]; mod = importlib.import_module(m)
    return _resolve(getattr(mod, a))
# Loads the recommender function from a dotted module path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="src/db/home_alone.db")
    p.add_argument("--train_end", required=True)
    p.add_argument("--val_end", default=None)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--rec", default="")
    p.add_argument("--threshold", type=float, default=4.0)
    a = p.parse_args()
    val_end = a.val_end or pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d")
    _, _, te = time_split(_ratings(a.db), a.train_end, val_end)
    fn = _load(a.rec)
    r = evaluate(fn, te, k=a.k, thr=a.threshold)
    print(json.dumps({"config":{"db":a.db,"train_end":a.train_end,"val_end":val_end,"k":a.k,"rec":a.rec or "<dummy>","threshold":a.threshold},"results":r}, indent=2))
# Entry point for command-line execution: parses arguments, loads data, evaluates the recommender, and prints results

if __name__ == "__main__": main()

# python -m evaluation.offline_eval --db src/db/home_alone.db --train_end 2025-07-15 --val_end 2025-10-15 --k 20 --rec src.models.ubcf.model:UBCFRecommender.recommend --threshold 4.0
# python -m evaluation.offline_eval --db src/db/home_alone.db --train_end 2025-07-15 --val_end 2025-10-15 --k 20 --rec src.models.matrix_factorization.mf:MFRecommender.recommend --threshold 4.0
# python -m evaluation.offline_eval --db src/db/home_alone.db --train_end 2025-07-15 --val_end 2025-10-15 --k 20 --threshold 4.0 (dummy)

