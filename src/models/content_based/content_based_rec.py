"""
Core features: Content-based recommender using movie metadata only.
- SentenceTransformer embeddings for `overview + tagline` (true semantic signal).
- Light genres TF-IDF and a tiny numeric block (runtime, popularity, vote stats).
- Public API: train(df), save(path), load(path), recommend(movie_id, k),

CLI printing behavior:
- When you request recommendations (by user_id), stdout prints a single line:
  a comma-separated list of recommended movie_id values. No extra text.

Examples:
  Train and save:
    # add imports at top of file:
    from models.content_based.content_based_rec import ContentBasedRecommender, Settings, _load_sqlite

    # in api call:
    db_path = "src/db/home_alone.db" 
    table = "movies"
    save_path = "src/models/content_based/artifacts/model.joblib"
    settings = Settings()
    df = _load_sqlite(db_path, table=table)
    rec = ContentBasedRecommender(settings=settings).train(df)
    rec.save(save_path)

  Recommend from a user_id:
    recommended = ContentBasedRecommender().recommend(user_id=int(userid), k=20)
"""
from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
import ast
import math
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from dataclasses import asdict


load_dotenv()  # take environment variables from .env file

# Lazy import for sentence-transformers (only when training)
_SENTENCE_TRANSFORMERS_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


#  ----------------------   CUSTOM EXCEPTIONS   ----------------------
class NoUserFound(LookupError):
    """Raised when a user has no watch history."""

    def __init__(self, user_id, message=None):
        if message is None:
            message = f"No watch history found for user_id '{user_id}'."
        super().__init__(message)
        self.user_id = user_id

class MovieNotFound(LookupError):
    """Raised when a movie_id is not found in the data."""

    def __init__(self, movie_id, message=None):
        if message is None:
            message = f"movie_id '{movie_id}' not found in the data."
        super().__init__(message)
        self.movie_id = movie_id


# ---------------------------- SQL helpers ----------------------------

def build_sql_select(table_name: str) -> str:
    return f"""
SELECT
    movie_id, tmdb_id, imdb_id, adult, genres, overview,
    original_language, popularity, production_companies,
    release_date, revenue, budget, runtime, spoken_languages,
    belongs_to_collection, vote_average, vote_count, video,
    title, production_countries, status, tagline
FROM {table_name}
"""


def latest_movie_for_user(user_id: str | int) -> Optional[str]:
    """
    Look up the most recently watched movie for a given user_id.
    Prefers last_watch DESC, then start_time DESC.
    Returns the movie_id string or None if not found.
    """

    db_path = os.getenv("DB_PATH")
    watch_table = "watch_history"

    q = f"""
    SELECT movie_id
    FROM {watch_table}
    WHERE user_id = ?
    ORDER BY
        COALESCE(last_watch, '') DESC,
        COALESCE(start_time, '') DESC
    LIMIT 1
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(q, (str(user_id),))
        row = cur.fetchone()
        return str(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


# ---------------------------- helpers ----------------------------

def _as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [t.strip() for t in x.split(',') if t.strip()]
    return []


def _maybe_json_list(x):
    """Try to parse JSON/list-like strings (e.g., "['Action','Drama']"). Fallback to comma-split."""
    if not isinstance(x, str):
        return _as_list(x)
    x = x.strip()
    if not x:
        return []
    if (x.startswith("[") and x.endswith("]")) or (x.startswith("(") and x.endswith(")")):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v]
        except Exception:
            pass
    return _as_list(x)


def _safe_str(x: Optional[str]) -> str:
    return x if isinstance(x, str) and x.strip() else ""


def build_text_for_embedding(row: pd.Series) -> str:
    """Combine overview and tagline for embedding input."""
    overview = _safe_str(row.get("overview"))
    tagline = _safe_str(row.get("tagline"))
    if overview and tagline:
        return f"{overview}\n{tagline}"
    return overview or tagline or _safe_str(row.get("title"))


def build_genre_string(row: pd.Series) -> str:
    genres_val = row.get("genres")
    genres_list = genres_val if isinstance(
        genres_val, list) else _maybe_json_list(genres_val)
    return " ".join(genres_list)


def _numeric_block(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ["runtime", "popularity",
                        "vote_average", "vote_count"] if c in df.columns]
    if not cols:
        return np.zeros((len(df), 0))
    X = df[cols].copy()
    if "vote_count" in X.columns:
        X["vote_count"] = np.log1p(X["vote_count"])  # long tail
    if "popularity" in X.columns:
        X["popularity"] = np.log1p(X["popularity"])  # long tail
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X.values)
    return Xs


# ---------------------------- model ----------------------------

@dataclass
class Settings:
    embed_model_name: str = "all-MiniLM-L6-v2"  # 384-dim, fast & light
    embed_weight: float = 0.6
    genre_weight: float = 0.3
    num_weight: float = 0.1
    genre_tfidf_max_features: int = 64


class ContentBasedRecommender:
    def __init__(self, settings: Settings | None = None, n_neighbors: int = 50):
        self.settings = settings or Settings()
        self.n_neighbors = n_neighbors
        # Components created/fit during training
        self.embed_model_name: str = self.settings.embed_model_name
        self.genre_vectorizer: Optional[TfidfVectorizer] = None
        self.nn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        # Cached matrices
        self._E = None       # embeddings (dense -> csr)
        self._X_genre = None  # csr
        self._X_num = None   # csr
        self.X = None        # union csr
        # Data
        self.df: Optional[pd.DataFrame] = None
        self.row_of = None   # movie_id -> row index

    # ---- training ----
    def train(self, df: pd.DataFrame):
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. pip install sentence-transformers")

        self.df = df.copy()
        self.df["_embed_text"] = self.df.apply(
            build_text_for_embedding, axis=1)
        self.df["_genres_str"] = self.df.apply(build_genre_string, axis=1)

        # 1) Semantic embeddings for overview+tagline
        model = SentenceTransformer(self.settings.embed_model_name)
        E = model.encode(
            self.df["_embed_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._E = sparse.csr_matrix(E)

        # 2) Genres as tiny TF-IDF
        self.genre_vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[^\s]+",
            max_features=self.settings.genre_tfidf_max_features,
            ngram_range=(1, 1),
            min_df=1,
        )
        self._X_genre = self.genre_vectorizer.fit_transform(
            self.df["_genres_str"].astype(str))

        # 3) Numerics
        num = _numeric_block(self.df)
        self._X_num = sparse.csr_matrix(
            num) if num.shape[1] > 0 and self.settings.num_weight > 0 else sparse.csr_matrix((len(self.df), 0))

        # 4) Weighted union
        blocks = []
        if self._E is not None and self._E.shape[1] > 0 and self.settings.embed_weight > 0:
            blocks.append(self._E * self.settings.embed_weight)
        if self._X_genre is not None and self._X_genre.shape[1] > 0 and self.settings.genre_weight > 0:
            blocks.append(self._X_genre * self.settings.genre_weight)
        if self._X_num is not None and self._X_num.shape[1] > 0 and self.settings.num_weight > 0:
            blocks.append(self._X_num * self.settings.num_weight)
        if not blocks:
            raise ValueError(
                "No feature blocks available. Check settings/inputs.")
        self.X = sparse.hstack(blocks).tocsr()

        # index
        self.df["movie_id"] = self.df["movie_id"].astype(str)
        self.row_of = {mid: i for i, mid in enumerate(
            self.df["movie_id"].tolist())}

        # fit neighbors on union
        self.nn.fit(self.X)
        return self

    # ---- recommend ----
    def _recommend_from_row(self, row: int, k: int) -> List[Tuple[str, float]]:
        distances, indices = self.nn.kneighbors(
            self.X[row], n_neighbors=min(k + 1, len(self.df)))
        idxs = indices[0].tolist()
        dists = distances[0].tolist()

        # remove self
        if row in idxs:
            j = idxs.index(row)
            idxs.pop(j)
            dists.pop(j)

        idxs = idxs[:k]
        dists = dists[:k]
        sims = [max(0.0, min(0.999, float(1.0 - d if not math.isnan(d) else 0.0)))
                for d in dists]
        ids = [str(self.df.iloc[i]["movie_id"]) for i in idxs]
        return list(zip(ids, sims))

    def get_similar_movies(self, movie_id: str, k: int = 20) -> List[Tuple[str, float]]:
        """Recommend K similar movies given a movie_id."""
        row = self.row_of.get(str(movie_id))
        if row is None:
            raise MovieNotFound(movie_id)
        return self._recommend_from_row(row, k)

    def recommend(self, user_id: str | int, k: int = 20) -> List[str]:
        """
        Look up the user's latest watched movie from watch_history, then recommend.
        """

        # load the model path from environment variable or default
        model_path = os.getenv("CB_MODEL_DIR", "src/models/content_based/artifacts/model.joblib")
        m = ContentBasedRecommender.get_loaded(model_path)

        # get the latest movie for the user
        latest_mid = latest_movie_for_user(user_id=user_id)
        if not latest_mid:
            raise NoUserFound(user_id)
        returned_pairs = m.get_similar_movies(latest_mid, k=k)
        returned_movies = [id for id, _ in returned_pairs]
        return returned_movies

    # ---- persistence ----
    def save(self, path: str):
        """Save model + artifacts to a single joblib file."""
        try:
            import joblib
        except Exception as e:
            raise RuntimeError(
                "joblib is required for save/load. pip install joblib") from e
        payload = {
            "settings_dict": asdict(self.settings),
            "n_neighbors": self.n_neighbors,
            "embed_model_name": self.embed_model_name,
            "genre_vectorizer": self.genre_vectorizer,
            "nn": self.nn,
            "X": self.X,
            "df": self.df[["movie_id", "title", "release_date", "_embed_text", "_genres_str"]] if self.df is not None else None,
            "row_of": self.row_of,
        }
        joblib.dump(payload, path)

    _cached_model = None  # singleton for get_loaded()
    @classmethod
    def get_loaded(cls, path: str):
        if cls._cached_model is None:
            cls._cached_model = cls.load(path)
        return cls._cached_model

    @classmethod
    def load(cls, path: str) -> "ContentBasedRecommender":
        """Load a previously saved recommender."""
        try:
            import joblib
        except Exception as e:
            raise RuntimeError(
                "joblib is required for save/load. pip install joblib") from e
        payload = joblib.load(path)
        settings = Settings(**payload["settings_dict"])
        obj = cls(settings=settings, n_neighbors=payload.get("n_neighbors", 50))
        obj.embed_model_name = payload.get(
            "embed_model_name", obj.settings.embed_model_name)
        obj.genre_vectorizer = payload.get("genre_vectorizer")
        obj.nn = payload["nn"]
        obj.X = payload["X"]
        obj.df = payload["df"]
        obj.row_of = payload["row_of"] or ({mid: i for i, mid in enumerate(
            obj.df["movie_id"].tolist())} if obj.df is not None else None)
        obj._E = None
        obj._X_genre = None
        obj._X_num = None
        return obj


# ---------------------------- data loading ----------------------------

def _load_sqlite(db_path: str, table: str = "movies") -> pd.DataFrame:
    sql = build_sql_select(table)
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()

    required = ["movie_id", "title", "overview", "tagline", "genres"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"SQLite table must include column '{col}'")

    df["movie_id"] = df["movie_id"].astype(str)
    df["genres"] = df["genres"].apply(_maybe_json_list)

    for col in ["runtime", "popularity", "vote_average", "vote_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------- CLI ----------------------------

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--db", default=(os.getenv("DB_PATH", "src/db/home_alone.db")),
#                    help="Path to SQLite database that has both 'movies' and 'watch_history' (e.g., home_alone.db)")
#     p.add_argument("--table", default="movies",
#                    help="Movies table name (default: movies)")
#     p.add_argument("--watch_table", default="watch_history",
#                    help="Watch history table name (default: watch_history)")
#     p.add_argument("--user_id", default=None, help="If provided, we lookup the latest movie for this user and recommend from it")
#     p.add_argument("--save", default=None,
#                    help="Optional path to save a trained model (e.g., model.joblib)")
#     p.add_argument("--model", default=None,
#                    help="Optional path to a saved model to load (skips training)")
#     p.add_argument("--embed_model", default=None,
#                    help="Override embedding model name (default: all-MiniLM-L6-v2)")
#     args = p.parse_args()

#     # Load or train
#     if args.model:
#         rec = ContentBasedRecommender.load(args.model)
#     else:
#         if not args.db:
#             raise SystemExit("--db is required when not using --model")
#         settings = Settings()
#         if args.embed_model:
#             settings.embed_model_name = args.embed_model
#         df = _load_sqlite(args.db, table=args.table)
#         rec = ContentBasedRecommender(settings=settings).train(df)
#         if args.save:
#             rec.save(args.save)

#     # Determine seed
#     pairs: List[Tuple[str, float]]
#     if args.user_id is not None:
#         if not args.db:
#             raise SystemExit("--db is required when using --user_id")
#         pairs = rec.recommend(user_id=args.user_id)
#     else:
#         raise SystemExit("Provide either --user_id for recommendation")


# if __name__ == "__main__":
#     main()
