"""
Configuration file for the content-based recommendation model.

- Sets paths for SQLite database and artifacts directory.
- Loads modeling parameters and weights from constants.
- Defines required columns expected in the movies database.
- Provides settings for TF-IDF vectorizer and stop words.
"""
import os
from . import constants as const
from dotenv import load_dotenv

load_dotenv()

# --- Data source (SQLite) ---
DB_PATH = os.getenv("DB_PATH")
TABLE_NAME = os.getenv("MOVIES_TABLE_NAME", "movies")

# --- Artifacts (saved under this folder) ---
ARTIFACT_DIR = os.getenv(
    "CB_ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "artifacts")
)

# --- Modeling knobs ---
TFIDF_MAX_FEATURES = const.TFIDF_MAX_FEATURES
TEXT_WEIGHT = const.CB_TEXT_WEIGHT
NUM_WEIGHT  = const.CB_NUM_WEIGHT
CAT_WEIGHT  = const.CB_CAT_WEIGHT

# For reproducible vectorizer behavior (tokenization, stop words)
STOP_WORDS = "english"

# Columns expected in the DB (aligned with your ETL)
REQUIRED_COLS = [
    'movie_id', 'tmdb_id', 'imdb_id', 'adult', 'genres', 'overview',
    'original_language', 'popularity', 'production_companies',
    'release_date', 'revenue', 'budget', 'runtime', 'spoken_languages',
    'belongs_to_collection', 'vote_average', 'vote_count', 'video',
    'title', 'production_countries', 'status', 'tagline'
]
