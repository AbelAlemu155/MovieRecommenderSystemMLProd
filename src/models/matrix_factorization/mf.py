import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from models.base_model import Recommender

class MatrixFactorizationRecommender(Recommender):
    def __init__(self, model_dir: str | None = None, k: int = 64, reg: float = 0.1, iters: int = 10):
        load_dotenv()
        self.k = k
        self.reg = reg
        self.iters = iters

        PROJECT_ROOT = Path(__file__).resolve().parents[3]
        default_dir = os.environ.get("MF_MODEL_DIR", "src/models/matrix_factorization")
        self.model_dir = PROJECT_ROOT / (model_dir or default_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.factors_path = self.model_dir / "mf_factors.npz"
        self.maps_path = self.model_dir / "mf_id_maps.pkl"

        self.U = None
        self.V = None
        self.user_id2idx = {}
        self.item_id2idx = {}
        self.user_idx2id = []
        self.item_idx2id = []

    # Data
    def _get_ratings(self) -> pd.DataFrame:
        path_to_db = os.getenv("DB_PATH", "src/db/home_alone.db")
        engine = create_engine("sqlite:///" + path_to_db)
        return pd.read_sql_query("SELECT user_id, movie_id, rating FROM ratings", engine)

    def _get_user_seen(self, uid: int) -> set[int]:
        try:
            df = self._get_ratings()
            return set(df.loc[df.user_id == uid, "movie_id"].unique().tolist())
        except Exception:
            return set()

    def _popularity_top_n(self, n: int, exclude_movie_ids=None) -> pd.Index:
        df = self._get_ratings()
        if df.empty:
            return pd.Index([])

        grp = df.groupby('movie_id').agg(v=('rating', 'count'), R=('rating', 'mean')).reset_index()
        C = df['rating'].mean()
        m = max(int(grp['v'].quantile(0.60)), 5)
        grp['score'] = (grp['v']/(grp['v']+m))*grp['R'] + (m/(grp['v']+m))*C

        if exclude_movie_ids:
            grp = grp[~grp['movie_id'].isin(exclude_movie_ids)]

        grp = grp.sort_values('score', ascending=False)
        return pd.Index(grp['movie_id'].head(n).tolist())

    # Train, save and load
    def train(self, df: pd.DataFrame | None = None):
        df = df if df is not None else self._get_ratings()
        if df.empty:
            raise ValueError("No ratings found to train MatrixFactorizationRecommender.")

        user_ids = df.user_id.unique()
        item_ids = df.movie_id.unique()

        self.user_id2idx = {u: i for i, u in enumerate(user_ids)}
        self.item_id2idx = {m: i for i, m in enumerate(item_ids)}
        self.user_idx2id = user_ids.tolist()
        self.item_idx2id = item_ids.tolist()

        n_users = len(user_ids)
        n_items = len(item_ids)
        k = self.k

        rng = np.random.default_rng(2025)
        U = 0.01 * rng.standard_normal((n_users, k))
        V = 0.01 * rng.standard_normal((n_items, k))

        # For each user: list of (item_idx, rating)
        user_items = [[] for _ in range(n_users)]
        # For each item: list of (user_idx, rating)
        item_users = [[] for _ in range(n_items)]

        user_means = df.groupby("user_id")["rating"].mean()
        df["rating_centered"] = df.apply(
            lambda r: r["rating"] - user_means.loc[r["user_id"]], axis=1
        )

        for _, r in df.iterrows():
            ui = self.user_id2idx[r.user_id]
            ii = self.item_id2idx[r.movie_id]
            user_items[ui].append((ii, r.rating_centered))
            item_users[ii].append((ui, r.rating_centered))

        regI = self.reg * np.eye(k)

        # ALS iterations
        for it in range(self.iters):
            # Solve for U (fix V)
            VtV = V.T @ V + regI
            for u in range(n_users):
                pairs = user_items[u]
                if not pairs:
                    continue
                idxs = np.fromiter((i for i, _ in pairs), dtype=int)
                r = np.fromiter((rv for _, rv in pairs), dtype=float)
                V_u = V[idxs]  # (nu, k)
                A = VtV - (V_u.T @ V_u) + (V_u.T @ V_u)  # keep numerics clean
                b = V_u.T @ r
                U[u] = np.linalg.solve(A, b)

            # Solve for V (fix U)
            UtU = U.T @ U + regI
            for i in range(n_items):
                pairs = item_users[i]
                if not pairs:
                    continue
                idxs = np.fromiter((u for u, _ in pairs), dtype=int)
                r = np.fromiter((rv for _, rv in pairs), dtype=float)
                U_i = U[idxs]  # (ni, k)
                A = UtU - (U_i.T @ U_i) + (U_i.T @ U_i)
                b = U_i.T @ r
                V[i] = np.linalg.solve(A, b)

        # store
        self.U, self.V = U, V
        np.savez_compressed(self.factors_path, U=U, V=V, k=k)
        with open(self.maps_path, "wb") as f:
            pickle.dump(
                {
                    "user_id2idx": self.user_id2idx,
                    "item_id2idx": self.item_id2idx,
                    "user_idx2id": self.user_idx2id,
                    "item_idx2id": self.item_idx2id,
                    "user_means": user_means.to_dict(),  # needed to re-add mean on scoring if desired
                },
                f,
            )

    def load(self):
        data = np.load(self.factors_path)
        self.U = data["U"]
        self.V = data["V"]
        with open(self.maps_path, "rb") as f:
            maps = pickle.load(f)
            self.user_id2idx = maps["user_id2idx"]
            self.item_id2idx = maps["item_id2idx"]
            self.user_idx2id = maps["user_idx2id"]
            self.item_idx2id = maps["item_idx2id"]
            self.user_means = maps.get("user_means", {})

    # Recommend
    def recommend(self, user_id: int, top_k: int = 20) -> pd.Index:
        if self.U is None or self.V is None:
            self.load()

        # unknown user
        if user_id not in self.user_id2idx:
            seen = self._get_user_seen(user_id)
            prev: list[int] = []

            pool_size = max(top_k * 3, top_k)
            pop_pool = self._popularity_top_n(
                pool_size,
                exclude_movie_ids=set(seen).union(set(prev))
            ).tolist()

            if len(pop_pool) > top_k:
                rng = np.random.default_rng()
                sampled = rng.choice(pop_pool, size=top_k, replace=False).tolist()
                prev = prev + sampled
            else:
                prev = prev + pop_pool

            return pd.Index(prev)

        uidx = self.user_id2idx[user_id]
        uvec = self.U[uidx]
        scores = self.V @ uvec

        mu = self.user_means.get(user_id, 0.0)
        scores = scores + mu

        seen = self._get_user_seen(user_id)
        if seen:
            seen_idx = [self.item_id2idx[i] for i in seen if i in self.item_id2idx]
            scores[np.array(seen_idx, dtype=int)] = -np.inf

        n_items = scores.shape[0]
        k = min(top_k, n_items)
        if k <= 0:
            return pd.Index([])
        
        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]

        top_movie_ids = [self.item_idx2id[i] for i in topk_idx]
        return pd.Index(top_movie_ids)


if __name__ == "__main__":
    mf = MatrixFactorizationRecommender()
    print("Training MF.")
    mf.train()
    print("Training done.")

    print("Loading model.")
    mf.load()
    print("Loading done.")

    print("Recommending")
    res = mf.recommend(user_id=56, top_k=20)
    print(f"Recommendations: {res}")