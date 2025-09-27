"""
base_model.py

Internal Documentation:

This module provides utilities and a base class for building content-based movie recommender systems.
It includes:
- Functions for loading and normalizing movie data from a SQLite database.
- Helpers for parsing and cleaning movie metadata fields.
- Methods for constructing weighted text documents from movie features for vectorization.
- Persistence utilities for saving/loading model artifacts.
- The BaseContentRecommender class, which wraps a vectorizer and NearestNeighbors for similarity queries.
Subclasses should implement specific vectorization logic for fit_transform and transform methods.

Intended for use as a foundation for content-based recommendation pipelines.
"""
import os
import ast
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd
from joblib import dump, load
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from models.content_based.config import DB_PATH, TABLE_NAME, REQUIRED_COLS, ARTIFACT_DIR

# ---------- DB utilities ----------

# Query to SELECT required columns (subset of REQUIRED_COLS)
SQL_SELECT = f"""
SELECT
    movie_id, tmdb_id, imdb_id, adult, genres, overview,
    original_language, popularity, production_companies,
    release_date, revenue, budget, runtime, spoken_languages,
    belongs_to_collection, vote_average, vote_count, video,
    title, production_countries, status, tagline
FROM {TABLE_NAME}
"""

# ---------- Data loading and normalization ----------


def _safe_parse_listish(x):
    """
    Accepts a python list, a TMDB-like stringified list of dicts, or string of names.
    Returns a list[str] of 'name' or tokens.
    """
    if isinstance(x, list):
        if x and isinstance(x[0], dict):
            return [d.get('name', '') for d in x if isinstance(d, dict)]
        return [str(v) for v in x]
    if isinstance(x, str):
        # Try to parse TMDB-like JSONish string
        try:
            v = ast.literal_eval(x)
            return _safe_parse_listish(v)
        except Exception:
            # Fallback: comma-separated names
            return [t.strip() for t in x.split(",") if t.strip()]
    return []

# Parses a collection input to safely extract the 'name' field if present.


def _safe_parse_collection(x):
    if isinstance(x, dict):
        return x.get('name', '')
    if isinstance(x, str):
        try:
            d = ast.literal_eval(x)
            if isinstance(d, dict):
                return d.get('name', '')
        except Exception:
            return x
    return ""

# Extract year from a date string (ISO format)


def _to_year(s):
    try:
        return datetime.fromisoformat(str(s)).year
    except Exception:
        return None

# Load and lightly normalize movie data from the SQLite DB.


def read_movies(limit: Optional[int] = None) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {DB_PATH}. Did you run the generator?")
    con = sqlite3.connect(str(DB_PATH))  # <-- cast to str
    q = SQL_SELECT + (f" LIMIT {int(limit)}" if limit else "")
    df = pd.read_sql(q, con)
    con.close()

    # Light normalization (ETL already cleaned, this handles representation issues)
    # lists/dicts → list[str]
    for col in ['genres', 'production_companies', 'spoken_languages', 'production_countries']:
        df[col] = df[col].apply(_safe_parse_listish)

    df['collection_name'] = df['belongs_to_collection'].apply(
        _safe_parse_collection)
    df['year'] = df['release_date'].apply(_to_year)

    # fill text nulls
    for c in ['title', 'tagline', 'overview', 'original_language', 'status', 'collection_name']:
        df[c] = df[c].fillna('')

    # numerics
    for c in ['popularity', 'revenue', 'budget', 'runtime', 'vote_average', 'vote_count']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # booleans
    for c in ['adult', 'video']:
        df[c] = df[c].astype(bool)

    # make sure movie_id is always a string
    df['movie_id'] = df['movie_id'].astype(str)

    return df

# ---------- Text document builder ----------


def build_text_doc(row: pd.Series) -> str:
    """Compose a weighted text document for TF-IDF."""
    genres = " ".join(row['genres'])
    prods = " ".join(row['production_companies'])
    langs = " ".join(row['spoken_languages'])
    countries = " ".join(row['production_countries'])
    parts = [
        (row['title'], 1),
        (row['tagline'], 1),
        (row['overview'], 2),
        (genres, 2),
        (prods, 1),
        (langs, 1),
        (countries, 1),
        (row['collection_name'], 1),
        (row['original_language'], 1),
        (row['status'], 1),
    ]
    doc = " ".join((" ".join([txt]*w)).strip() for txt, w in parts if txt)
    return doc


def add_docs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['doc'] = df.apply(build_text_doc, axis=1)
    return df

# ---------- Persistence helpers ----------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_artifact(obj, name: str):
    ensure_dir(ARTIFACT_DIR)
    dump(obj, os.path.join(ARTIFACT_DIR, name))


def load_artifact(name: str):
    return load(os.path.join(ARTIFACT_DIR, name))

# ---------- Base recommender ----------


class BaseContentRecommender:
    """
    Common wrapper around a vectorizer/pipeline and NearestNeighbors.
    Subclasses must implement:
      - fit_transform(df) -> sparse matrix
      - transform(df) -> sparse matrix
      - name (str): used as filename prefix
    """
    name: str = "base"

    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nn = NearestNeighbors(
            metric=metric, algorithm="brute", n_neighbors=n_neighbors)
        self.X = None
        self.id_map = None  # DataFrame: row <-> movie_id
        self.df_meta = None  # store a slim copy for quick lookups

    # -- subclass hooks --
    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        raise NotImplementedError

    # -- workflow --
    def fit(self, df: pd.DataFrame, id_col: str = "movie_id"):
        self.df_meta = df[[id_col, "title",
                           "year", "original_language"]].copy()
        self.df_meta[id_col] = self.df_meta[id_col].astype(
            str)   # <-- add this
        self.id_map = self.df_meta.reset_index().rename(
            columns={"index": "row"})
        self.X = self.fit_transform(df)
        self.nn.fit(self.X)
        return self

    def save(self):
        prefix = f"{self.name}"
        save_artifact(self.nn,       f"{prefix}_nn.joblib")
        save_artifact(self.X,        f"{prefix}_X.joblib")
        save_artifact(self.id_map,   f"{prefix}_idmap.joblib")
        # subclasses should also persist their pipelines/vectorizers

    def load(self):
        prefix = f"{self.name}"
        self.nn     = load_artifact(f"{prefix}_nn.joblib")
        self.X      = load_artifact(f"{prefix}_X.joblib")
        self.id_map = load_artifact(f"{prefix}_idmap.joblib")
        # ensure id_map uses string ids
        if 'movie_id' in self.id_map.columns:
            self.id_map['movie_id'] = self.id_map['movie_id'].astype(str)


    # -- queries --
    def _row_of(self, movie_id: str) -> int:
        rec = self.id_map.loc[self.id_map['movie_id'] == movie_id]
        if rec.empty:
            raise KeyError(f"movie_id {movie_id} not found")
        return int(rec['row'].values[0])

    def similar_by_id(self, movie_id: str, k: int = 10, exclude_self: bool = True):
        # locate the row for this string id
        rec = self.id_map.loc[self.id_map['movie_id'] == str(movie_id)]
        if rec.empty:
            raise KeyError(f"movie_id '{movie_id}' not found")
        row = int(rec['row'].values[0])

        # nearest neighbors
        distances, indices = self.nn.kneighbors(self.X[row], n_neighbors=k+1)
        idxs = indices[0].tolist()

        # optionally remove the query itself
        if exclude_self and row in idxs:
            idxs.remove(row)

        # map row indices -> string ids
        inv = self.id_map.set_index('row')['movie_id']
        ids = [inv[i] for i in idxs[:k]]
        sims = 1 - distances[0][:len(ids)]
        # Extra guard: drop any residual self matches by id value
        pairs = [(mid, float(sim)) for mid, sim in zip(ids, sims) if str(mid) != str(movie_id)]
        return pairs[:k]
