import argparse
import sqlite3
from typing import List, Tuple, Dict
import numpy as np
import os

from .faiss_hnsw import FaissHNSWIndex

def fetch_meta(db_path: str, table: str, ids: List[str]) -> Dict[str, Tuple[str, str]]:
    """
    Returns dict: movie_id -> (title, year)
    Assumes columns: movie_id (TEXT), title (TEXT), release_date (TEXT 'YYYY-MM-DD' or 'YYYY')
    """
    if not ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        qmarks = ",".join(["?"] * len(ids))
        sql = f"""
            SELECT movie_id, title,
                   COALESCE(substr(release_date, 1, 4), '') as year
            FROM {table}
            WHERE movie_id IN ({qmarks})
        """
        rows = conn.execute(sql, ids).fetchall()
        return {r[0]: (r[1] or "", r[2] or "") for r in rows}
    finally:
        conn.close()

def print_recs(seed_id: str, recs: List[Tuple[str, float]], db_path: str, table: str) -> None:
    meta = fetch_meta(db_path, table, [mid for mid, _ in recs])
    print("Movie Recommendations:")
    print("--------------------------------------------------")
    for i, (mid, sim) in enumerate(recs, start=1):
        title, year = meta.get(mid, (mid, ""))  # fallback to id if not found
        line = f"{i:2d}. {title} ({year})"
        padding = max(0, 54 - len(line))
        print(f" {line}{' ' * padding} - Similarity: {sim:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/faiss_hnsw.index")
    ap.add_argument("--meta",  default="artifacts/faiss_hnsw.meta.json")
    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--table", required=True, help="Table with movie metadata")
    ap.add_argument("--seed", required=True, help="Seed movie_id to query")
    ap.add_argument("--topk", type=int, default=5)

    # Build-on-the-fly options (npy + ids csv)
    ap.add_argument("--build_from", default="", help="Optional: npy dense matrix to build FAISS if index missing")
    ap.add_argument("--ids_csv", default="", help="Optional: CSV with id_map (columns: row,movie_id)")

    # Preferred path: existing mixed-model artifacts (joblib)
    ap.add_argument("--x_joblib", default="", help="Path to cb_mixed_X.joblib (sparse or dense)")
    ap.add_argument("--idmap_joblib", default="", help="Path to cb_mixed_idmap.joblib (pandas DataFrame)")
    args = ap.parse_args()

    # 1) Load or build FAISS index
    idx = None
    try:
        tmp = FaissHNSWIndex(dim=0)              # placeholder, real dim comes from load()
        tmp.load(args.index, args.meta)
        idx = tmp
    except Exception:
        # Try to build from JOBLIB artifacts first (you already have these)
        if args.x_joblib and args.idmap_joblib:
            from joblib import load as _load
            X = _load(args.x_joblib)
            idmap = _load(args.idmap_joblib)
            # align checks
            n_rows = X.shape[0]
            if len(idmap) != n_rows:
                raise ValueError("Row count mismatch between X joblib and idmap joblib.")
            ids = idmap["movie_id"].astype(str).tolist()

            # normalize to float32 and dense row-major where needed
            try:
                import scipy.sparse as sp  # optional
                if sp.issparse(X):
                    X = X.astype("float32").tocsr()
                else:
                    X = X.astype("float32", copy=False)
            except Exception:
                # if scipy not present, try best-effort conversion
                X = X.astype("float32", copy=False)

            # For FAISS we need a dense ndarray; if sparse, convert
            try:
                X_dense = X.toarray().astype("float32")  # if sparse CSR
            except AttributeError:
                X_dense = X.astype("float32", copy=False)

            idx = FaissHNSWIndex(dim=X_dense.shape[1], metric="ip", hnsw_m=32, ef_search=80, ef_construction=200)
            idx.fit(X_dense, ids)
            # Persist for future runs
            os.makedirs(os.path.dirname(args.index), exist_ok=True)
            idx.save(args.index, args.meta)

        # Otherwise try npy + csv
        elif args.build_from and args.ids_csv:
            import pandas as pd
            X = np.load(args.build_from).astype("float32", copy=False)
            ids = pd.read_csv(args.ids_csv)["movie_id"].astype(str).tolist()
            if X.shape[0] != len(ids):
                raise ValueError("X rows and ids_csv length mismatch.")
            idx = FaissHNSWIndex(dim=X.shape[1], metric="ip", hnsw_m=32, ef_search=80, ef_construction=200)
            idx.fit(X, ids)
            os.makedirs(os.path.dirname(args.index), exist_ok=True)
            idx.save(args.index, args.meta)
        else:
            raise RuntimeError(
                "FAISS index not found. Provide either "
                "--x_joblib and --idmap_joblib (recommended) or "
                "--build_from and --ids_csv to build it."
            )

    # 2) Resolve the REAL seed vector
    seed_vec = None
    row = None

    if args.x_joblib and args.idmap_joblib:
        from joblib import load as _load
        X = _load(args.x_joblib)
        idmap = _load(args.idmap_joblib)
        try:
            row = int(idmap.loc[idmap["movie_id"] == str(args.seed), "row"].values[0])
        except Exception:
            raise KeyError(f"Seed movie_id '{args.seed}' not found in idmap joblib")
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                seed_vec = X[row].toarray().astype("float32").ravel()
            else:
                seed_vec = np.asarray(X[row]).astype("float32").ravel()
        except Exception:
            seed_vec = np.asarray(X[row]).astype("float32").ravel()

    elif args.build_from and args.ids_csv:
        import pandas as pd
        X = np.load(args.build_from).astype("float32", copy=False)
        ids = pd.read_csv(args.ids_csv)["movie_id"].astype(str).tolist()
        try:
            row = ids.index(str(args.seed))
        except ValueError:
            raise KeyError(f"Seed movie_id '{args.seed}' not found in ids_csv")
        seed_vec = X[row]

    if seed_vec is None:
        raise RuntimeError(
            "No query vector available. Provide either "
            "--x_joblib and --idmap_joblib (preferred) or "
            "--build_from and --ids_csv."
        )

    # 3) Query FAISS (exclude the seed)
    recs = idx.query(vector=seed_vec, top_k=args.topk, exclude_ids={str(args.seed)})

    # 4) Pretty-print
    print_recs(args.seed, recs, args.db, args.table)

if __name__ == "__main__":
    main()


"""

Run this in terminal:

python -m models.content_based.faiss_cli \
  --db src/db/home_alone.db \
  --table movies \
  --seed "pulp+fiction+1994" \          <---- Switch the seed movie_id here
  --topk 5 \
  --x_joblib src/models/content_based/artifacts/cb_mixed_X.joblib \
  --idmap_joblib src/models/content_based/artifacts/cb_mixed_idmap.joblib \
  --index src/models/content_based/artifacts/faiss_hnsw.index \
  --meta  src/models/content_based/artifacts/faiss_hnsw.meta.json


"""