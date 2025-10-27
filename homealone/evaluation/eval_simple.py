import os, io, time, contextlib
from typing import List, Tuple, Iterable
import numpy as np
import pandas as pd

# optional progress bar
try:
    from tqdm import tqdm
    def pbar(it, **kw): return tqdm(it, **kw)
except Exception:
    def pbar(it, **kw): return it

# -------- dual-path imports --------
try:
    from src.evaluation.get_data import get_data
    from src.models.collaborative_filtering.helpers import compute_implicit_rating
    from src.models.collaborative_filtering.item_based_filtering import ItemBasedFiltering
    from src.models.content_based.content_based_rec import ContentBasedRecommender
    from src.models.ubcf.model import UBCFRecommender
    from src.models.deep_rec_sys.deep_rec_model import DeepRecSysModel
    IN_SRC = False; OUT_DIR = "src/evaluation"
except ModuleNotFoundError:
    from evaluation.get_data import get_data
    from models.collaborative_filtering.helpers import compute_implicit_rating
    from models.collaborative_filtering.item_based_filtering import ItemBasedFiltering
    from models.content_based.content_based_rec import ContentBasedRecommender
    from models.ubcf.model import UBCFRecommender
    from models.deep_rec_sys.deep_rec_model import DeepRecSysModel
    IN_SRC = True; OUT_DIR = "evaluation"

# -------- config (env knobs) --------
TOP_K = int(os.getenv("EVAL_TOPK", "20"))
USER_LIMIT = int(os.getenv("EVAL_USER_LIMIT", "0"))  # 0 = all users
# train toggles (ON by default so you get nonzero metrics)
EVAL_TRAIN_CONTENT = int(os.getenv("EVAL_TRAIN_CONTENT", "1"))
EVAL_FORCE_TRAIN_UBCF = int(os.getenv("EVAL_FORCE_TRAIN_UBCF", "1"))
EVAL_TRAIN_DEEP = int(os.getenv("EVAL_TRAIN_DEEP", "1"))
DEEP_EPOCHS = int(os.getenv("EVAL_DEEP_EPOCHS", "2"))    # keep small for M1
DEEP_BATCH  = int(os.getenv("EVAL_DEEP_BATCH", "512"))
DEEP_K      = int(os.getenv("EVAL_DEEP_K", "40"))

QUERY = """
    SELECT 
        WH.user_id,
        WH.movie_id,
        WH.total_watch_time,
        WH.last_watch,
        (julianday(WH.last_watch) - julianday(WH.start_time)) * 24*60 AS gap_watch,
        R.rating, 
        m.runtime
    FROM watch_history AS WH
    JOIN movies as m ON m.movie_id = WH.movie_id
    LEFT JOIN ratings AS R
    ON WH.user_id = R.user_id AND WH.movie_id = R.movie_id;
"""

# -------- helpers --------
def _norm(ids: Iterable[str]) -> List[str]:
    return [str(x).strip().lower() for x in ids]

def _first_existing(paths: List[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _artifact_size_mb(paths: List[str]) -> float:
    p = _first_existing(paths)
    if not p: return 0.0
    try: return os.path.getsize(p) / (1024**2)
    except Exception: return 0.0

def _item_artifacts(): return [
    "models/collaborative_filtering/saved_models/item_based_models.pkl",
    "src/models/collaborative_filtering/saved_models/item_based_models.pkl",
]
def _ubcf_artifacts(): return [
    os.getenv("UBCF_MODEL_PATH", "").strip(),
    "artifacts/ubcf.pkl", "src/artifacts/ubcf.pkl",
]
def _deep_artifacts(): return [
    "src/models/deep_rec_sys/checkpoints/two_tower_bpr_best.pt",
    "models/deep_rec_sys/checkpoints/two_tower_bpr_best.pt",
]
def _content_artifacts(): return [
    "artifacts/content_model.joblib", "src/artifacts/content_model.joblib",
]

def _build_ground_truth(test_df: pd.DataFrame, k: int = TOP_K) -> Tuple[pd.Index, List[List[str]]]:
    df = compute_implicit_rating(df=test_df)[["user_id","movie_id","implicit_pref"]]
    mat = df.pivot_table(index="user_id", columns="movie_id", values="implicit_pref", aggfunc="mean")
    if USER_LIMIT > 0: mat = mat.iloc[:USER_LIMIT]
    rel = mat.apply(lambda r: r.nlargest(k).index.tolist(), axis=1)
    return mat.index, [_norm(x) for x in rel.tolist()]

def precision_at_k(pred, rel, k): 
    if not pred: return 0.0
    s=set(rel); return sum(1 for x in pred[:k] if x in s)/min(k,len(pred))
def recall_at_k(pred, rel, k): 
    if not rel: return 0.0
    s=set(rel); return sum(1 for x in pred[:k] if x in s)/len(rel)
def dcg_at_k(pred, rel, k):
    s=set(rel); return sum( (1.0/np.log2(i+1+1)) for i,x in enumerate(pred[:k]) if x in s )
def ndcg_at_k(pred, rel, k):
    ideal=min(len(rel),k)
    if ideal==0: return 0.0
    idcg=sum(1.0/np.log2(i+2) for i in range(ideal))
    return dcg_at_k(pred, rel, k)/idcg
def hit_at_k(pred, rel, k):
    s=set(rel); return int(any(x in s for x in pred[:k]))

def evaluate_lists(pred_lists, rel_lists, k):
    P,R,N,H=[],[],[],[]
    for pred,rel in zip(pred_lists, rel_lists):
        P.append(precision_at_k(pred,rel,k))
        R.append(recall_at_k(pred,rel,k))
        N.append(ndcg_at_k(pred,rel,k))
        H.append(hit_at_k(pred,rel,k))
    distinct_pred={m for preds in pred_lists for m in preds}
    distinct_rel={m for rel in rel_lists for m in rel}
    return {
        f"Precision@{k}": float(np.mean(P)) if P else 0.0,
        f"Recall@{k}": float(np.mean(R)) if R else 0.0,
        f"NDCG@{k}": float(np.mean(N)) if N else 0.0,
        f"HR@{k}": float(np.mean(H)) if H else 0.0,
        "Coverage": float(len(distinct_pred)/max(1,len(distinct_rel))),
    }

def latency_stats_ms(times_s):
    if not times_s: return dict(Latency_p50_ms=0,Latency_p95_ms=0,Latency_mean_ms=0,Throughput_rps=0)
    arr=np.array(times_s)*1000.0
    mean=float(arr.mean())
    return {
        "Latency_p50_ms": float(np.percentile(arr,50)),
        "Latency_p95_ms": float(np.percentile(arr,95)),
        "Latency_mean_ms": mean,
        "Throughput_rps": 1000.0/mean if mean>0 else 0.0,
    }

# -------- evaluators (TRAIN ALL) --------
def _ensure_item_dir():
    rel = "models/collaborative_filtering/saved_models" if IN_SRC else "src/models/collaborative_filtering/saved_models"
    os.makedirs(rel, exist_ok=True)

def evaluate_item_based(users, rel_lists):
    _ensure_item_dir()
    train_df, val_df, _ = get_data(QUERY)
    model = ItemBasedFiltering(train_set=train_df, val_set=val_df)
    t0=time.time(); model.train(); train_time=time.time()-t0
    preds,times=[],[]
    for uid in pbar(users, desc="ItemBased"):
        t1=time.time()
        try: out=model.recommend(int(uid))
        except Exception: out=[]
        times.append(time.time()-t1); preds.append(_norm(out))
    m=evaluate_lists(preds, rel_lists, TOP_K); m.update(latency_stats_ms(times))
    m["ModelSize_MB"]=_artifact_size_mb(_item_artifacts()); 
    return "ItemBased", m, train_time

def evaluate_content(users, rel_lists):
    rec = ContentBasedRecommender()
    t0=time.time()
    if EVAL_TRAIN_CONTENT and hasattr(rec,"train"):
        try: rec.train()
        except Exception: pass
    train_time=time.time()-t0
    preds,times=[],[]
    for uid in pbar(users, desc="Content"):
        t1=time.time()
        try: out=rec.recommend(user_id=int(uid), k=TOP_K)
        except Exception: out=[]
        times.append(time.time()-t1); preds.append(_norm(out))
    m=evaluate_lists(preds, rel_lists, TOP_K); m.update(latency_stats_ms(times))
    m["ModelSize_MB"]=_artifact_size_mb(_content_artifacts())
    return "Content", m, train_time

def evaluate_ubcf(users, rel_lists):
    ubcf = UBCFRecommender()
    t0=time.time()
    if EVAL_FORCE_TRAIN_UBCF and hasattr(ubcf,"train"):
        try: ubcf.train()
        except Exception: pass
    train_time=time.time()-t0
    preds,times=[],[]
    for uid in pbar(users, desc="UBCF"):
        t1=time.time()
        try:
            recs=ubcf.recommend(user_id=int(uid), k=TOP_K, n_recommendations=TOP_K)
            out=list(recs.index)[:TOP_K] if hasattr(recs,"index") else list(recs)[:TOP_K]
        except Exception: out=[]
        times.append(time.time()-t1); preds.append(_norm(out))
    m=evaluate_lists(preds, rel_lists, TOP_K); m.update(latency_stats_ms(times))
    m["ModelSize_MB"]=_artifact_size_mb(_ubcf_artifacts())
    return "UBCF", m, train_time

def evaluate_deep(users, rel_lists):
    model = DeepRecSysModel(base_path="src/models/deep_rec_sys")
    # expected CSVs (your DeepRecSys loads/creates NPZs from these)
    model.watch_csv  = "src/db/watch_history.csv"
    model.ratings_csv= "src/db/ratings.csv"
    model.movies_csv = "src/db/movies.csv"
    model.load_data()
    t0=time.time()
    if EVAL_TRAIN_DEEP and hasattr(model,"train"):
        try: model.train(epochs=DEEP_EPOCHS, batch_size=DEEP_BATCH, K=DEEP_K)
        except Exception: pass
    train_time=time.time()-t0
    preds,times=[],[]
    for uid in pbar(users, desc="Deep"):
        t1=time.time()
        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                out=model.recommend(user_id=int(uid), top_k=TOP_K)
        except Exception: out=[]
        times.append(time.time()-t1); preds.append(_norm(out))
    m=evaluate_lists(preds, rel_lists, TOP_K); m.update(latency_stats_ms(times))
    m["ModelSize_MB"]=_artifact_size_mb(_deep_artifacts())
    return "Deep", m, train_time

# -------- orchestrator --------
def run_all():
    _, _, test_df = get_data(QUERY)
    users, rel_lists = _build_ground_truth(test_df, k=TOP_K)
    print(f"Evaluating {len(users)} users | K={TOP_K}")
    results=[]
    for fn in (evaluate_item_based, evaluate_content, evaluate_ubcf, evaluate_deep):
        name, metrics, train_time = fn(users, rel_lists)
        row={"Model":name,"Train Time (s)":round(train_time,3),
             "ModelSize_MB":round(metrics.pop("ModelSize_MB",0.0),3)}
        row.update({k: (round(v,4) if isinstance(v,float) else v) for k,v in metrics.items()})
        results.append(row); print(f"[{name}] {row}")
    df=pd.DataFrame(results).set_index("Model")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv=os.path.join(OUT_DIR,"results_simple.csv"); out_md=os.path.join(OUT_DIR,"results_simple.md")
    df.to_csv(out_csv)
    with open(out_md,"w") as f:
        f.write("| Model | Precision | Recall | NDCG | HR | Coverage | p50 (ms) | p95 (ms) | mean (ms) | RPS | Size (MB) | Train (s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for m,r in df.iterrows():
            f.write(f"| {m} | {r[f'Precision@{TOP_K}']:.4f} | {r[f'Recall@{TOP_K}']:.4f} | "
                    f"{r[f'NDCG@{TOP_K}']:.4f} | {r[f'HR@{TOP_K}']:.4f} | {r['Coverage']:.4f} | "
                    f"{r['Latency_p50_ms']:.1f} | {r['Latency_p95_ms']:.1f} | {r['Latency_mean_ms']:.1f} | "
                    f"{r['Throughput_rps']:.2f} | {r['ModelSize_MB']:.3f} | {r['Train Time (s)']:.3f} |\n")
    print("\nSaved:", out_csv, out_md)

if __name__ == "__main__":
    run_all()