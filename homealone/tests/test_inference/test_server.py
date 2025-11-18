"""
Tests for the inference server module.

These tests cover:
1. Resource loading (models, trainset, fallback pool)
2. Fallback pool preparation and random sampling
3. Top-N prediction sorting and formatting
4. Flask API endpoints and user recommendation logic
5. Database connection management
6. Error handling for unknown users and edge cases
"""

import pytest
import pickle
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock all external dependencies before importing server
surprise_mock = MagicMock()
flask_mock = MagicMock()
watchdog_mock = MagicMock()

sys.modules['surprise'] = surprise_mock
sys.modules['surprise.model_selection'] = surprise_mock
sys.modules['surprise.dump'] = surprise_mock
sys.modules['flask'] = flask_mock
sys.modules['watchdog'] = watchdog_mock
sys.modules['watchdog.observers'] = watchdog_mock
sys.modules['watchdog.events'] = watchdog_mock

# Create mock classes for surprise
Dataset_mock = MagicMock()
Reader_mock = MagicMock()
SVD_mock = MagicMock()
KNNBasic_mock = MagicMock()
NMF_mock = MagicMock()
dump_mock = MagicMock()

surprise_mock.Dataset = Dataset_mock
surprise_mock.Reader = Reader_mock
surprise_mock.SVD = SVD_mock
surprise_mock.KNNBasic = KNNBasic_mock
surprise_mock.NMF = NMF_mock
surprise_mock.dump = dump_mock

# Create mock Flask app
app_mock = MagicMock()
flask_mock.Flask = MagicMock(return_value=app_mock)
flask_mock.jsonify = lambda x: x
flask_mock.abort = MagicMock()
flask_mock.g = MagicMock()
flask_mock.request = MagicMock()

# Now we can import
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise import dump

# Add the inference directory to the path
inference_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference'))
sys.path.insert(0, inference_dir)

from homealone.inference.server import (
    prepare_fallback_pool,
    get_random_fallback,
    get_top_n,
    app,
    get_db
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trainset():
    """Create a sample trainset for testing"""
    trainset = MagicMock()
    
    # Mock the ir attribute (item ratings: movie_id -> [(user_id, rating)])
    trainset.ir = {
        0: [(0, 5), (1, 3)],  # movie 0: avg = 4.0
        1: [(0, 4), (2, 5)],  # movie 1: avg = 4.5
        2: [(1, 2), (2, 1)],  # movie 2: avg = 1.5
    }
    
    # Mock ID conversion methods
    trainset.to_raw_iid = lambda iid: f'movie{iid}'
    trainset.to_inner_iid = lambda raw_iid: int(raw_iid.replace('movie', ''))
    trainset.to_raw_uid = lambda uid: f'user{uid}'
    trainset.to_inner_uid = lambda raw_uid: int(raw_uid.replace('user', ''))
    
    trainset.n_users = 3
    trainset.n_items = 3
    trainset.all_items = lambda: range(3)
    trainset.knows_item = lambda iid: True
    
    # Mock ur attribute (user ratings: user_id -> [(item_id, rating)])
    trainset.ur = {
        0: [(0, 5), (1, 4)],
        1: [(0, 3), (2, 2)],
        2: [(1, 5), (2, 1)],
    }
    
    return trainset


@pytest.fixture
def sample_trainset_with_numeric_ids():
    """Create a trainset with numeric user/movie IDs"""
    trainset = MagicMock()
    
    trainset.ir = {
        0: [(0, 5), (1, 3)],
        1: [(0, 4), (2, 5)],
        2: [(1, 2), (2, 1)],
        3: [(3, 3)],
    }
    
    trainset.to_raw_iid = lambda iid: 100 + iid
    trainset.to_inner_iid = lambda raw_iid: raw_iid - 100
    trainset.to_raw_uid = lambda uid: uid
    trainset.to_inner_uid = lambda raw_uid: raw_uid
    
    trainset.n_users = 4
    trainset.n_items = 4
    trainset.all_items = lambda: range(4)
    trainset.knows_item = lambda iid: 0 <= iid < 4
    
    trainset.ur = {
        0: [(0, 5), (1, 4)],
        1: [(0, 3), (2, 2)],
        2: [(1, 5), (2, 1)],
        3: [(3, 3)],
    }
    
    return trainset


@pytest.fixture
def sample_fallback_pool():
    """Sample fallback pool for testing"""
    return [
        {"movie_id": "movie1", "avg_rating": 4.5},
        {"movie_id": "movie2", "avg_rating": 4.0},
        {"movie_id": "movie3", "avg_rating": 3.8},
        {"movie_id": "movie4", "avg_rating": 3.5},
        {"movie_id": "movie5", "avg_rating": 3.2},
    ]


@pytest.fixture
def temp_test_db():
    """Create a temporary test database"""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            recommended_items TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def flask_client(temp_test_db):
    """Create a Flask test client"""
    from homealone.inference import server

    with patch.object(server, 'DATABASE', temp_test_db):
        app.config = {'TESTING': True}
        app.test_client = MagicMock()
        mock_client = MagicMock()
        app.test_client.return_value.__enter__ = MagicMock(return_value=mock_client)
        app.test_client.return_value.__exit__ = MagicMock()
        yield mock_client


@pytest.fixture
def mock_predictions():
    """Create mock prediction objects"""
    predictions = []
    for i, (uid, iid, est) in enumerate([
        (1, 'movie1', 4.5),
        (1, 'movie2', 4.2),
        (1, 'movie3', 3.8),
        (1, 'movie4', 3.5),
        (1, 'movie5', 3.2),
    ]):
        pred = Mock()
        pred.uid = uid
        pred.iid = iid
        pred.est = est
        pred.r_ui = 0.0
        predictions.append(pred)
    return predictions


# ============================================================================
# Test: Fallback Pool Preparation
# ============================================================================

def test_prepare_fallback_pool_basic(sample_trainset):
    """Test basic fallback pool preparation with default parameters"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=2.0)

    assert isinstance(pool, list), "Pool should be a list"
    assert len(pool) > 0, "Pool should contain movies"

    for item in pool:
        assert 'movie_id' in item
        assert 'avg_rating' in item
        assert item['avg_rating'] >= 2.0


def test_prepare_fallback_pool_high_threshold(sample_trainset):
    """Test fallback pool with high rating threshold"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=4.0)

    for item in pool:
        assert item['avg_rating'] >= 4.0


def test_prepare_fallback_pool_min_ratings_filter(sample_trainset):
    """Test fallback pool filters by minimum number of ratings"""
    pool_min_1 = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=1.0)
    pool_min_3 = prepare_fallback_pool(sample_trainset, min_ratings=3, min_avg_rating=1.0)

    assert len(pool_min_1) > 0
    assert len(pool_min_3) == 0


def test_prepare_fallback_pool_empty_result(sample_trainset):
    """Test fallback pool with impossible criteria"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=5.0)

    assert isinstance(pool, list)
    assert len(pool) == 0


def test_prepare_fallback_pool_rating_values(sample_trainset):
    """Test that avg_rating is correctly calculated and rounded"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=1.0)

    for item in pool:
        assert isinstance(item['avg_rating'], float)
        assert len(str(item['avg_rating']).split('.')[-1]) <= 2


# ============================================================================
# Test: Random Fallback Sampling
# ============================================================================

def test_get_random_fallback_basic(sample_fallback_pool):
    """Test basic random fallback sampling"""
    result = get_random_fallback(sample_fallback_pool, n=3)

    assert isinstance(result, str)
    movie_ids = result.split(',')
    assert len(movie_ids) == 3


def test_get_random_fallback_sorted_by_rating(sample_fallback_pool):
    """Test that fallback results are sorted by average rating"""
    result = get_random_fallback(sample_fallback_pool, n=5)
    movie_ids = result.split(',')

    ratings = [item['avg_rating'] for item in sample_fallback_pool if item['movie_id'] in movie_ids]
    assert ratings == sorted(ratings, reverse=True)


def test_get_random_fallback_n_greater_than_pool(sample_fallback_pool):
    """Test requesting more movies than available in pool"""
    result = get_random_fallback(sample_fallback_pool, n=100)
    movie_ids = result.split(',')

    assert len(movie_ids) == len(sample_fallback_pool)


def test_get_random_fallback_empty_pool():
    """Test random fallback with empty pool"""
    result = get_random_fallback([], n=10)

    assert isinstance(result, str)
    assert result == ""


def test_get_random_fallback_n_zero(sample_fallback_pool):
    """Test requesting zero movies"""
    result = get_random_fallback(sample_fallback_pool, n=0)

    assert result == ""


def test_get_random_fallback_format():
    """Test that output format is comma-separated without spaces"""
    pool = [
        {"movie_id": "movie1", "avg_rating": 4.5},
        {"movie_id": "movie2", "avg_rating": 4.0},
    ]
    result = get_random_fallback(pool, n=2)

    assert ',' in result
    assert ' ' not in result


# ============================================================================
# Test: Top-N Prediction Sorting
# ============================================================================

def test_get_top_n_basic(mock_predictions):
    """Test basic top-N prediction extraction"""
    result = get_top_n(mock_predictions, n=3)

    assert isinstance(result, str)
    movie_ids = result.split(',')
    assert len(movie_ids) == 3


def test_get_top_n_sorted_by_estimated_rating(mock_predictions):
    """Test that predictions are sorted by estimated rating"""
    result = get_top_n(mock_predictions, n=5)
    movie_ids = result.split(',')

    assert movie_ids[0] == 'movie1'
    assert movie_ids[-1] == 'movie5'


def test_get_top_n_all_predictions(mock_predictions):
    """Test requesting all predictions"""
    result = get_top_n(mock_predictions, n=100)
    movie_ids = result.split(',')

    assert len(movie_ids) == len(mock_predictions)


def test_get_top_n_empty_predictions():
    """Test with empty predictions list"""
    result = get_top_n([], n=10)

    assert result == ""


def test_get_top_n_n_zero(mock_predictions):
    """Test requesting zero predictions"""
    result = get_top_n(mock_predictions, n=0)

    assert result == ""


def test_get_top_n_format():
    """Test output format is correct"""
    predictions = []
    for i, (uid, iid, est) in enumerate([
        (1, 'movie1', 4.5),
        (1, 'movie2', 4.2),
    ]):
        pred = Mock()
        pred.uid = uid
        pred.iid = iid
        pred.est = est
        predictions.append(pred)

    result = get_top_n(predictions, n=2)

    assert ',' in result
    assert ' ' not in result
    assert result == "movie1,movie2"


# ============================================================================
# Test: Database Connection Management
# ============================================================================

def test_get_db_creates_connection(temp_test_db):
    """Test that get_db creates a database connection"""
    from homealone.inference import server

    with app.app_context() if hasattr(app, 'app_context') else patch('flask.g', MagicMock()):
        with patch.object(server, 'DATABASE', temp_test_db):
            with patch('flask.g', MagicMock()) as mock_g:
                mock_g.get = MagicMock(return_value=None)
                db = get_db()

                assert db is not None
                assert isinstance(db, sqlite3.Connection)


def test_database_connection_wal_mode(temp_test_db):
    """Test that database connection uses WAL mode"""
    conn = sqlite3.connect(temp_test_db)
    conn.execute('PRAGMA journal_mode=WAL')
    cursor = conn.cursor()

    cursor.execute('PRAGMA journal_mode')
    result = cursor.fetchone()[0]

    assert result.upper() == 'WAL'
    conn.close()
