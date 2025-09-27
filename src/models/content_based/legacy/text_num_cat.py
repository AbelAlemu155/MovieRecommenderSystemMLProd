"""
content-based/text_num_cat.py

Implements a content-based movie recommender using text, numeric, and categorical features.
- NumericBlock: Transforms and log-scales selected numeric columns.
- UnionAssembler: Combines TF-IDF text features, scaled numeric features, and one-hot encoded categorical features into a single sparse matrix.
- ContentModelMixed: Main recommender class using the combined features for similarity-based recommendations.
- build_and_save: Utility to train and persist the model.
- load_model: Utility to load a saved model.

Configuration and helper functions are imported from local modules.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump, load
from models.content_based.base_model import BaseContentRecommender, read_movies, add_docs, save_artifact, load_artifact
from models.content_based.config import (
    TFIDF_MAX_FEATURES, STOP_WORDS,
    TEXT_WEIGHT, NUM_WEIGHT, CAT_WEIGHT
)

NUM_COLS = ["popularity","runtime","vote_average","vote_count","budget","revenue"]
CAT_COLS = ["original_language","status"]

# ---- small helpers ----
class NumericBlock(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, df):
        arr = df[NUM_COLS].copy()
        arr["budget"]     = np.log1p(arr["budget"])
        arr["revenue"]    = np.log1p(arr["revenue"])
        arr["vote_count"] = np.log1p(arr["vote_count"])
        return arr.values

class UnionAssembler(BaseEstimator, TransformerMixin):
    def __init__(self, vec, scaler, ohe):
        self.vec = vec
        self.scaler = scaler
        self.ohe = ohe

    def fit(self, df, y=None):
        Xt = self.vec.fit_transform(df['doc'])
        Xn = self.scaler.fit_transform(NumericBlock().transform(df))
        Xn = sparse.csr_matrix(Xn)
        Xc = self.ohe.fit_transform(df[CAT_COLS])
        self._shape_text = Xt.shape[1]
        self._shape_num  = Xn.shape[1]
        self._shape_cat  = Xc.shape[1]
        self._Xt0, self._Xn0, self._Xc0 = Xt, Xn, Xc
        return self

    def transform(self, df):
        Xt = self.vec.transform(df['doc'])
        Xn = self.scaler.transform(NumericBlock().transform(df))
        Xn = sparse.csr_matrix(Xn)
        Xc = self.ohe.transform(df[CAT_COLS])
        return sparse.hstack([
            Xt * TEXT_WEIGHT,
            Xn * NUM_WEIGHT,
            Xc * CAT_WEIGHT
        ], format="csr")

    def fit_transform(self, df, y=None):
        self.fit(df, y=y)
        return sparse.hstack([
            self._Xt0 * TEXT_WEIGHT,
            self._Xn0 * NUM_WEIGHT,
            self._Xc0 * CAT_WEIGHT
        ], format="csr")

class ContentModelMixed(BaseContentRecommender):
    name = "cb_mixed"

    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        super().__init__(n_neighbors=n_neighbors, metric=metric)
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1,2),
            min_df=3,
            max_df=0.8,
            stop_words=STOP_WORDS
        )
        self.scaler = StandardScaler(with_mean=False)
        self.ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
        self.union = UnionAssembler(self.vectorizer, self.scaler, self.ohe)

    def fit_transform(self, df):
        return self.union.fit_transform(df)

    def transform(self, df):
        return self.union.transform(df)

    def save(self):
        super().save()
        save_artifact(self.vectorizer, f"{self.name}_tfidf.joblib")
        save_artifact(self.scaler,    f"{self.name}_scaler.joblib")
        save_artifact(self.ohe,       f"{self.name}_ohe.joblib")

    def load(self):
        super().load()
        self.vectorizer = load_artifact(f"{self.name}_tfidf.joblib")
        self.scaler    = load_artifact(f"{self.name}_scaler.joblib")
        self.ohe       = load_artifact(f"{self.name}_ohe.joblib")
        self.union = UnionAssembler(self.vectorizer, self.scaler, self.ohe)

# -------- convenience runners --------

def build_and_save(limit=None):
    df = add_docs(read_movies(limit=limit))
    model = ContentModelMixed(n_neighbors=50, metric="cosine").fit(df)
    model.save()
    return model

def load_model():
    m = ContentModelMixed()
    m.load()
    return m

def format_movie_recommendations(recommendations):
    """
    Format movie recommendations list into readable output
    
    Args:
        recommendations: List of tuples (movie_title, similarity_score)
    
    Returns:
        Formatted string output
    """
    print("Movie Recommendations:")
    print("-" * 50)
    
    for i, (movie_title, similarity) in enumerate(recommendations, 1):
        # Clean up the movie title
        clean_title = movie_title.replace('+', ' ').title()
        
        # Extract year if present
        import re
        year_match = re.search(r'(\d{4})', clean_title)
        if year_match:
            year = year_match.group(1)
            title_without_year = clean_title.replace(year, '').strip()
            formatted_title = f"{title_without_year} ({year})"
        else:
            formatted_title = clean_title
        
        # Format similarity score
        similarity_score = float(similarity) if hasattr(similarity, 'item') else similarity
        
        print(f"{i:2d}. {formatted_title:<50} - Similarity: {similarity_score:.3f}")


# Internal Test run
if __name__ == "__main__":
    # build_and_save(limit=1000)
    # print("Model built and artifacts saved.")
    m = load_model()
    print("Model and artifacts loaded.")
    format_movie_recommendations(m.similar_by_id("titanic+1997")[:5])  # should find "die hard 1988"