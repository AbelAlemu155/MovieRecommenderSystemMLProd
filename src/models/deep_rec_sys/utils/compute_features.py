import math
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction import FeatureHasher


# --------------------------
# 1) USER FEATURES -> NPZ
# --------------------------
def build_user_features_npz(
    user_csv: str,
    out_npz_path: str,
    *,
    user_id_col: str = "id",
    age_col: str = "age",
    occupation_col: str = "occupation",
    gender_col: str = "gender",
    n_hash_occupation: int = 32,
    n_hash_gender: int = 4,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Read user_data.csv (id, age, occupation, gender) and produce a fixed-length numeric vector per user.
    Saved to a single NPZ: {X: (N,D), ids: (N,), feature_names: (D,)}.

    Features:
      - age_raw
      - age_bucket (decade)
      - log1p_age
      - occupation_hash_* (FeatureHasher over occupation string)
      - gender_hash_* (FeatureHasher over gender string)
    """
    rng = np.random.default_rng(random_state)
    df = pd.read_csv(user_csv)

    # Basic checks
    for col in [user_id_col, age_col, occupation_col, gender_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from {user_csv}")

    # Prepare hashers
    h_occ = FeatureHasher(n_features=n_hash_occupation, input_type="string", alternate_sign=False)
    h_gen = FeatureHasher(n_features=n_hash_gender, input_type="string", alternate_sign=False)

    def _to_float(x) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    # Build features row-by-row (keeps memory modest if needed)
    user_ids = df[user_id_col].astype(object).to_numpy()
    X_blocks = []

    # Numeric block (age transforms)
    age_raw = df[age_col].apply(_to_float).to_numpy(dtype=float)
    age_bucket = (np.floor(age_raw / 10.0) * 10.0).astype(float)
    log1p_age = np.log1p(np.maximum(age_raw, 0.0))
    numeric_block = np.stack([age_raw, age_bucket, log1p_age], axis=1)

    # Hashed occupation (string)
    occ_input = [[str(x)] for x in df[occupation_col].fillna("").astype(str).tolist()]
    occ_block = h_occ.transform(occ_input).toarray()

    # Hashed gender (string)
    gen_input = [[str(x)] for x in df[gender_col].fillna("").astype(str).tolist()]
    gen_block = h_gen.transform(gen_input).toarray()

    # Concatenate
    X = np.concatenate([numeric_block, occ_block, gen_block], axis=1).astype(np.float32)

    feature_names = (
        ["age_raw", "age_bucket", "log1p_age"]
        + [f"occ_hash_{i}" for i in range(n_hash_occupation)]
        + [f"gender_hash_{i}" for i in range(n_hash_gender)]
    )

    # Save
    np.savez_compressed(
        out_npz_path,
        X=X,
        ids=user_ids,
        feature_names=np.array(feature_names, dtype=object),
    )

    return {"X": X, "ids": user_ids, "feature_names": feature_names, "out": out_npz_path}


def build_user_features_from_interactions(
    watch_csv: str,
    ratings_csv: str,
    out_npz: str,
):
    watch = pd.read_csv(watch_csv)
    ratings = pd.read_csv(ratings_csv)

    user_ids = sorted(set(watch["user_id"]) | set(ratings["user_id"]))
    id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

    X = np.eye(len(user_ids), dtype=np.float32)  # one-hot
    np.savez_compressed(out_npz, X=X, ids=np.array(user_ids, dtype=object))

    return {"X": X, "ids": user_ids, "out": out_npz}


# --------------------------
# 2) MOVIE FEATURES -> NPZ
# --------------------------
def build_movie_features_npz(
    movie_csv: str,
    out_npz_path: str,
    *,
    movie_id_col: str = "movie_id",
    # schema columns
    cols: Dict[str, str] = None,
    # hash sizes
    n_hash_lang: int = 16,
    n_hash_genres: int = 32,
    n_hash_countries: int = 16,
    n_hash_sp_langs: int = 16,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Read movie_data.csv and produce a fixed-length numeric vector per movie.
    Saved to a single NPZ: {X: (M,D), ids: (M,), feature_names: (D,)}.

    Numeric features (light transforms):
      - is_adult, has_video
      - release_year, release_decade
      - log1p_runtime, log1p_budget, log1p_revenue, budget_to_revenue_ratio
      - popularity, vote_average, log1p_vote_count
      - n_genres, n_spoken_languages, n_production_countries, in_collection
      - overview_len, title_len, tagline_len

    Hashed blocks:
      - original_language -> n_hash_lang
      - genres -> n_hash_genres   (list)
      - production_countries -> n_hash_countries (list)
      - spoken_languages -> n_hash_sp_langs (list)
    """
    if cols is None:
        cols = {
            "tmdb_id": "tmdb_id",
            "imdb_id": "imdb_id",
            "adult": "adult",
            "genres": "genres",
            "overview": "overview",
            "original_language": "original_language",
            "popularity": "popularity",
            "release_date": "release_date",
            "revenue": "revenue",
            "budget": "budget",
            "runtime": "runtime",
            "spoken_languages": "spoken_languages",
            "belongs_to_collection": "belongs_to_collection",
            "vote_average": "vote_average",
            "vote_count": "vote_count",
            "video": "video",
            "title": "title",
            "production_countries": "production_countries",
            "status": "status",
            "tagline": "tagline",
        }

    rng = np.random.default_rng(random_state)
    df = pd.read_csv(movie_csv)

    if movie_id_col not in df.columns:
        raise ValueError(f"Column '{movie_id_col}' missing from {movie_csv}")

    # helpers
    def _to_bool(x) -> float:
        s = str(x).strip().lower()
        return 1.0 if s in {"1", "true", "t", "yes", "y"} else 0.0

    def _to_float(x) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def _len_or_zero(x) -> int:
        return len(x) if isinstance(x, str) else 0

    def _parse_year(x) -> int:
        if pd.isna(x):
            return 0
        s = str(x).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
            try:
                return datetime.strptime(s, fmt).year
            except Exception:
                pass
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
        return 0

    def _parse_list(x) -> List[str]:
        if pd.isna(x):
            return []
        s = str(x).strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if arr and isinstance(arr[0], dict) and "name" in arr[0]:
                    return [str(d.get("name", "")).strip() for d in arr if isinstance(d, dict)]
                return [str(e).strip() for e in arr]
            except Exception:
                pass
        # fallback: pipe-delimited
        return [t.strip() for t in s.split("|") if t.strip()]

    # Prepare hashers
    h_lang = FeatureHasher(n_features=n_hash_lang, input_type="string", alternate_sign=False)
    h_genres = FeatureHasher(n_features=n_hash_genres, input_type="string", alternate_sign=False)
    h_countries = FeatureHasher(n_features=n_hash_countries, input_type="string", alternate_sign=False)
    h_sp_langs = FeatureHasher(n_features=n_hash_sp_langs, input_type="string", alternate_sign=False)

    movie_ids = df[movie_id_col].astype(object).to_numpy()

    # Numeric core
    is_adult = df[cols["adult"]].apply(_to_bool).to_numpy(dtype=float) if cols["adult"] in df else np.zeros(len(df))
    has_video = df[cols["video"]].apply(_to_bool).to_numpy(dtype=float) if cols["video"] in df else np.zeros(len(df))

    y = df[cols["release_date"]].apply(_parse_year) if cols["release_date"] in df else pd.Series([0]*len(df))
    y = y.fillna(0).astype(int).to_numpy()
    decade = ((y // 10) * 10).astype(int)

    runtime = df[cols["runtime"]].apply(_to_float).to_numpy(dtype=float) if cols["runtime"] in df else np.zeros(len(df))
    budget = df[cols["budget"]].apply(_to_float).to_numpy(dtype=float) if cols["budget"] in df else np.zeros(len(df))
    revenue = df[cols["revenue"]].apply(_to_float).to_numpy(dtype=float) if cols["revenue"] in df else np.zeros(len(df))
    popularity = df[cols["popularity"]].apply(_to_float).to_numpy(dtype=float) if cols["popularity"] in df else np.zeros(len(df))
    vote_avg = df[cols["vote_average"]].apply(_to_float).to_numpy(dtype=float) if cols["vote_average"] in df else np.zeros(len(df))
    vote_cnt = df[cols["vote_count"]].apply(_to_float).to_numpy(dtype=float) if cols["vote_count"] in df else np.zeros(len(df))

    log1p_runtime = np.log1p(np.maximum(runtime, 0.0))
    log1p_budget = np.log1p(np.maximum(budget, 0.0))
    log1p_revenue = np.log1p(np.maximum(revenue, 0.0))
    log1p_vote_cnt = np.log1p(np.maximum(vote_cnt, 0.0))
    br_ratio = np.where(budget > 0, revenue / (budget + 1e-6), 0.0)

    # Lists
    genres_list = df[cols["genres"]].apply(_parse_list) if cols["genres"] in df else pd.Series([[]]*len(df))
    sp_langs_list = df[cols["spoken_languages"]].apply(_parse_list) if cols["spoken_languages"] in df else pd.Series([[]]*len(df))
    prod_countries_list = df[cols["production_countries"]].apply(_parse_list) if cols["production_countries"] in df else pd.Series([[]]*len(df))

    n_genres = genres_list.apply(len).to_numpy(dtype=float)
    n_sp_langs = sp_langs_list.apply(len).to_numpy(dtype=float)
    n_countries = prod_countries_list.apply(len).to_numpy(dtype=float)

    in_collection = (
        df[cols["belongs_to_collection"]]
        .fillna("")
        .astype(str)
        .apply(lambda s: 0.0 if s.strip() in {"", "[]", "null", "None", "0"} else 1.0)
        .to_numpy(dtype=float)
        if cols["belongs_to_collection"] in df
        else np.zeros(len(df))
    )

    # Text lengths
    overview_len = df[cols["overview"]].fillna("").astype(str).apply(len).to_numpy(dtype=float) if cols["overview"] in df else np.zeros(len(df))
    title_len = df[cols["title"]].fillna("").astype(str).apply(len).to_numpy(dtype=float) if cols["title"] in df else np.zeros(len(df))
    tagline_len = df[cols["tagline"]].fillna("").astype(str).apply(len).to_numpy(dtype=float) if cols["tagline"] in df else np.zeros(len(df))

    # Hashed blocks
    lang_input = [[str(x).strip()] for x in df[cols["original_language"]].fillna("").astype(str).tolist()] if cols["original_language"] in df else [[] for _ in range(len(df))]
    lang_block = h_lang.transform(lang_input).toarray()

    genres_input = genres_list.tolist()
    genres_block = h_genres.transform(genres_input).toarray()

    countries_input = prod_countries_list.tolist()
    countries_block = h_countries.transform(countries_input).toarray()

    sp_langs_input = sp_langs_list.tolist()
    sp_langs_block = h_sp_langs.transform(sp_langs_input).toarray()

    # Assemble numeric block
    numeric_cols = np.stack([
        is_adult, has_video, y.astype(float), decade.astype(float),
        log1p_runtime, log1p_budget, log1p_revenue, br_ratio,
        popularity, vote_avg, log1p_vote_cnt,
        n_genres, n_sp_langs, n_countries, in_collection,
        overview_len, title_len, tagline_len,
    ], axis=1).astype(np.float32)

    X = np.concatenate(
        [numeric_cols, lang_block, genres_block, countries_block, sp_langs_block],
        axis=1
    ).astype(np.float32)

    feature_names = (
        ["is_adult","has_video","release_year","release_decade",
         "log1p_runtime","log1p_budget","log1p_revenue","budget_to_revenue_ratio",
         "popularity","vote_average","log1p_vote_count",
         "n_genres","n_spoken_languages","n_production_countries","in_collection",
         "overview_len","title_len","tagline_len"] +
        [f"orig_lang_hash_{i}" for i in range(n_hash_lang)] +
        [f"genres_hash_{i}" for i in range(n_hash_genres)] +
        [f"prod_countries_hash_{i}" for i in range(n_hash_countries)] +
        [f"spoken_langs_hash_{i}" for i in range(n_hash_sp_langs)]
    )

    # Save
    np.savez_compressed(
        out_npz_path,
        X=X,
        ids=movie_ids,
        feature_names=np.array(feature_names, dtype=object),
    )

    return {"X": X, "ids": movie_ids, "feature_names": feature_names, "out": out_npz_path}



if __name__ == "__main__":

    movie_csv_path = "src/db/movies.csv"

    out_user_npz = "src/models/deep_rec_sys/features/user_features.npz"
    out_movie_npz = "src/models/deep_rec_sys/features/movie_features.npz"

    os.makedirs(os.path.dirname(out_user_npz), exist_ok=True)
    os.makedirs(os.path.dirname(out_movie_npz), exist_ok=True)

    user_result = build_user_features_from_interactions(
        watch_csv="src/db/watch_history.csv",
        ratings_csv="src/db/ratings.csv",
        out_npz=out_user_npz,
    )
    print(f"User features saved to {user_result['out']} with shape {user_result['X'].shape}")

    movie_result = build_movie_features_npz(movie_csv_path, out_movie_npz)
    print(f"Movie features saved to {movie_result['out']} with shape {movie_result['X'].shape}")