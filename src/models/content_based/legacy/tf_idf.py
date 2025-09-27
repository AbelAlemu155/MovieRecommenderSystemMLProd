"""
TF-IDF Content-Based Recommender Model.

This module implements a content-based recommender system using TF-IDF vectorization
for movie documents. It provides functionality to fit, transform, save, and load
the TF-IDF model and its artifacts.

Classes:
    ContentModelText: Content-based recommender using TF-IDF features.

Functions:
    build_and_save: Builds the model from movie data and saves artifacts.
    load_model: Loads a previously saved model and its vectorizer.

Internal:
    - Uses sklearn's TfidfVectorizer for feature extraction.
    - Artifacts are saved/loaded using joblib for persistence.
    - Configuration parameters are imported from config.
"""

from typing import Optional
from scipy import sparse
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from models.content_based.base_model import BaseContentRecommender, read_movies, add_docs, save_artifact, load_artifact
from models.content_based.config import TFIDF_MAX_FEATURES, STOP_WORDS

class ContentModelText(BaseContentRecommender):
    name = "cb_text"

    def __init__(self, n_neighbors: int = 50, metric: str = "cosine"):
        super().__init__(n_neighbors=n_neighbors, metric=metric)
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1,2),
            min_df=3,
            max_df=0.8,
            stop_words=STOP_WORDS
        )

    def fit_transform(self, df):
        X = self.vectorizer.fit_transform(df['doc'])
        return X

    def transform(self, df):
        return self.vectorizer.transform(df['doc'])

    def save(self):
        super().save()
        save_artifact(self.vectorizer, f"{self.name}_tfidf.joblib")

    def load(self):
        super().load()
        self.vectorizer = load_artifact(f"{self.name}_tfidf.joblib")

# -------- convenience runners --------

def build_and_save(limit: Optional[int] = None):
    df = add_docs(read_movies(limit=limit))
    model = ContentModelText(n_neighbors=50, metric="cosine").fit(df)
    model.save()
    return model

def load_model():
    m = ContentModelText()
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
    build_and_save(limit=1000)
    print("Model built and artifacts saved.")
    m = load_model()
    print("Model and artifacts loaded.")
    format_movie_recommendations(m.similar_by_id("titanic+1997")[:5])  # should find "harry potter and the chamber of secrets 2002"