# evaluation/metrics.py
from __future__ import annotations
import math
from typing import Sequence, Set, Dict, Iterable, List

def precision_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if k <= 0: return 0.0
    topk = pred[:k]
    return len(set(topk) & truth) / k

def recall_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if not truth: return 0.0
    topk = pred[:k]
    return len(set(topk) & truth) / len(truth)

def _dcg_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(pred[:k], start=1):
        if item in truth:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def ndcg_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if not truth: return 0.0
    ideal = _dcg_at_k(list(truth), truth, min(k, len(truth)))
    if ideal == 0: return 0.0
    return _dcg_at_k(pred, truth, k) / ideal

def average_precision(pred: Sequence[str], truth: Set[str], k: int | None = None) -> float:
    if not truth: return 0.0
    if k is None: k = len(pred)
    k = min(k, len(pred))
    hits, s = 0, 0.0
    for i in range(k):
        if pred[i] in truth:
            hits += 1
            s += hits / (i + 1)
    return s / len(truth)

def hit_rate_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    return 1.0 if set(pred[:k]) & truth else 0.0

def aggregate(users: Iterable[str],
              preds: Dict[str, List[str]],
              truths: Dict[str, Set[str]],
              k_prec=10, k_rec=20, k_ndcg=10, k_hr=10, k_map=None) -> Dict[str, float]:
    if k_map is None: k_map = k_prec
    P = []; R = []; N = []; H = []; M = []
    for u in users:
        p = preds.get(u, [])
        t = truths.get(u, set())
        P.append(precision_at_k(p, t, k_prec))
        R.append(recall_at_k(p, t, k_rec))
        N.append(ndcg_at_k(p, t, k_ndcg))
        H.append(hit_rate_at_k(p, t, k_hr))
        M.append(average_precision(p, t, k_map))
    return {
        "Precision@10": sum(P)/len(P) if P else 0.0,
        "Recall@20":    sum(R)/len(R) if R else 0.0,
        "NDCG@10":      sum(N)/len(N) if N else 0.0,
        "HR@10":        sum(H)/len(H) if H else 0.0,
        "MAP":          sum(M)/len(M) if M else 0.0,
    }
