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
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise import dump
import sys

# Add the inference directory to the path
inference_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../inference'))
sys.path.insert(0, inference_dir)

from homealone.inference.server import (
    load_resources,
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
    data_dict = {
        'user_id': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3'],
        'movie_id': ['movie1', 'movie2', 'movie1', 'movie3', 'movie2', 'movie3'],
        'rating': [5, 4, 3, 2, 5, 1]
    }
    import pandas as pd
    df = pd.DataFrame(data_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    return trainset


@pytest.fixture
def sample_trainset_with_numeric_ids():
    """Create a trainset with numeric user/movie IDs"""
    data_dict = {
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'movie_id': [101, 102, 101, 103, 102, 103, 101, 104],
        'rating': [5, 4, 3, 2, 5, 1, 4, 3]
    }
    import pandas as pd
    df = pd.DataFrame(data_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()
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
def temp_model_files(sample_trainset_with_numeric_ids):
    """Create temporary model files for testing load_resources"""
    temp_dir = tempfile.mkdtemp()

    # Create trainset file
    trainset_path = os.path.join(temp_dir, "trainset.dump")
    with open(trainset_path, 'wb') as f:
        pickle.dump(sample_trainset_with_numeric_ids, f)

    # Create SVD model
    svd_model = SVD(n_factors=2, random_state=42)
    svd_model.fit(sample_trainset_with_numeric_ids)
    svd_path = os.path.join(temp_dir, "svd_model.dump")
    dump.dump(svd_path, algo=svd_model)

    # Create KNN model
    knn_model = KNNBasic(k=2, sim_options={'name': 'cosine', 'user_based': True})
    knn_model.fit(sample_trainset_with_numeric_ids)
    knn_path = os.path.join(temp_dir, "knn_model.dump")
    dump.dump(knn_path, algo=knn_model)

    # Create NMF model
    nmf_model = NMF(n_factors=2, random_state=42)
    nmf_model.fit(sample_trainset_with_numeric_ids)
    nmf_path = os.path.join(temp_dir, "nmf_model.dump")
    dump.dump(nmf_path, algo=nmf_model)

    yield {
        'dir': temp_dir,
        'trainset': trainset_path,
        'svd': svd_path,
        'knn': knn_path,
        'nmf': nmf_path
    }

    # Cleanup
    for file in [trainset_path, svd_path, knn_path, nmf_path]:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(temp_dir)


@pytest.fixture
def temp_test_db():
    """Create a temporary test database"""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables that telemetry needs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL UNIQUE,
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
    # Import the server module to patch its DATABASE variable
    from homealone.inference import server

    # Patch the DATABASE path to use our temp database
    with patch.object(server, 'DATABASE', temp_test_db):
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client


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
        pred.r_ui = 0.0  # actual rating (not used for recommendations)
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

    # Check structure of pool items
    for item in pool:
        assert 'movie_id' in item, "Each item should have movie_id"
        assert 'avg_rating' in item, "Each item should have avg_rating"
        assert item['avg_rating'] >= 2.0, f"All ratings should be >= min_avg_rating (2.0)"


def test_prepare_fallback_pool_high_threshold(sample_trainset):
    """Test fallback pool with high rating threshold"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=4.0)

    # Only movies with avg_rating >= 4.0 should be included
    for item in pool:
        assert item['avg_rating'] >= 4.0, "All ratings should be >= 4.0"


def test_prepare_fallback_pool_min_ratings_filter(sample_trainset):
    """Test fallback pool filters by minimum number of ratings"""
    # With our sample data, each movie has 2 ratings
    pool_min_1 = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=1.0)
    pool_min_3 = prepare_fallback_pool(sample_trainset, min_ratings=3, min_avg_rating=1.0)

    assert len(pool_min_1) > 0, "Should have movies with >= 1 rating"
    assert len(pool_min_3) == 0, "Should have no movies with >= 3 ratings"


def test_prepare_fallback_pool_empty_result():
    """Test fallback pool with impossible criteria"""
    data_dict = {
        'user_id': [1, 2],
        'movie_id': [101, 102],
        'rating': [1, 2]
    }
    import pandas as pd
    df = pd.DataFrame(data_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Require very high average rating
    pool = prepare_fallback_pool(trainset, min_ratings=1, min_avg_rating=4.5)

    assert isinstance(pool, list), "Should return empty list, not None"
    assert len(pool) == 0, "Should be empty with impossible criteria"


def test_prepare_fallback_pool_rating_values(sample_trainset):
    """Test that avg_rating is correctly calculated and rounded"""
    pool = prepare_fallback_pool(sample_trainset, min_ratings=1, min_avg_rating=1.0)

    for item in pool:
        # avg_rating should be rounded to 2 decimal places
        assert isinstance(item['avg_rating'], float), "avg_rating should be float"
        # Check that it's properly rounded (max 2 decimal places)
        assert len(str(item['avg_rating']).split('.')[-1]) <= 2, "Should be rounded to 2 decimals"


# ============================================================================
# Test: Random Fallback Sampling
# ============================================================================

def test_get_random_fallback_basic(sample_fallback_pool):
    """Test basic random fallback sampling"""
    result = get_random_fallback(sample_fallback_pool, n=3)

    assert isinstance(result, str), "Should return string of comma-separated IDs"

    movie_ids = result.split(',')
    assert len(movie_ids) == 3, "Should return exactly 3 movies"


def test_get_random_fallback_sorted_by_rating(sample_fallback_pool):
    """Test that fallback results are sorted by average rating (descending)"""
    result = get_random_fallback(sample_fallback_pool, n=5)
    movie_ids = result.split(',')

    # Extract ratings for returned movies
    ratings = [item['avg_rating'] for item in sample_fallback_pool if item['movie_id'] in movie_ids]

    # Check that they're in descending order
    assert ratings == sorted(ratings, reverse=True), "Should be sorted by rating descending"


def test_get_random_fallback_n_greater_than_pool(sample_fallback_pool):
    """Test requesting more movies than available in pool"""
    result = get_random_fallback(sample_fallback_pool, n=100)
    movie_ids = result.split(',')

    # Should return all movies in pool
    assert len(movie_ids) == len(sample_fallback_pool), "Should return all available movies"


def test_get_random_fallback_empty_pool():
    """Test random fallback with empty pool"""
    result = get_random_fallback([], n=10)

    assert isinstance(result, str), "Should return string"
    assert result == "", "Should return empty string for empty pool"


def test_get_random_fallback_n_zero(sample_fallback_pool):
    """Test requesting zero movies"""
    result = get_random_fallback(sample_fallback_pool, n=0)

    assert result == "", "Should return empty string when n=0"


def test_get_random_fallback_format():
    """Test that output format is comma-separated without spaces"""
    pool = [
        {"movie_id": "movie1", "avg_rating": 4.5},
        {"movie_id": "movie2", "avg_rating": 4.0},
    ]
    result = get_random_fallback(pool, n=2)

    assert ',' in result, "Should contain comma separator"
    assert ' ' not in result, "Should not contain spaces"
    assert result.startswith('movie'), "Should start with movie ID"
    assert result.endswith('movie1') or result.endswith('movie2'), "Should end with movie ID"


# ============================================================================
# Test: Top-N Prediction Sorting
# ============================================================================

def test_get_top_n_basic(mock_predictions):
    """Test basic top-N prediction extraction"""
    result = get_top_n(mock_predictions, n=3)

    assert isinstance(result, str), "Should return comma-separated string"

    movie_ids = result.split(',')
    assert len(movie_ids) == 3, "Should return exactly 3 predictions"


def test_get_top_n_sorted_by_estimated_rating(mock_predictions):
    """Test that predictions are sorted by estimated rating (descending)"""
    result = get_top_n(mock_predictions, n=5)
    movie_ids = result.split(',')

    # First item should be the one with highest estimated rating
    assert movie_ids[0] == 'movie1', "First should be movie1 (est=4.5)"
    assert movie_ids[-1] == 'movie5', "Last should be movie5 (est=3.2)"


def test_get_top_n_all_predictions(mock_predictions):
    """Test requesting all predictions"""
    result = get_top_n(mock_predictions, n=100)
    movie_ids = result.split(',')

    assert len(movie_ids) == len(mock_predictions), "Should return all predictions"


def test_get_top_n_empty_predictions():
    """Test with empty predictions list"""
    result = get_top_n([], n=10)

    assert result == "", "Should return empty string for empty predictions"


def test_get_top_n_n_zero(mock_predictions):
    """Test requesting zero predictions"""
    result = get_top_n(mock_predictions, n=0)

    assert result == "", "Should return empty string when n=0"


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

    assert ',' in result, "Should contain comma separator"
    assert ' ' not in result, "Should not contain spaces"
    assert result == "movie1,movie2", "Should be in correct format"


def test_get_top_n_modifies_list():
    """Test that get_top_n modifies the predictions list (sorts in-place)"""
    predictions = []
    for i, (uid, iid, est) in enumerate([
        (1, 'movie3', 3.0),
        (1, 'movie1', 5.0),
        (1, 'movie2', 4.0),
    ]):
        pred = Mock()
        pred.uid = uid
        pred.iid = iid
        pred.est = est
        predictions.append(pred)

    original_order = [p.iid for p in predictions]
    result = get_top_n(predictions, n=3)
    new_order = [p.iid for p in predictions]

    # The list should be sorted now
    assert original_order != new_order, "List should be modified"
    assert new_order == ['movie1', 'movie2', 'movie3'], "Should be sorted by est descending"


# ============================================================================
# Test: Flask API Endpoints
# ============================================================================

def test_root_endpoint(flask_client):
    """Test the root endpoint returns success"""
    response = flask_client.get('/')

    assert response.status_code == 200
    assert b'Home Alone Recommendation API is running!' in response.data


def test_recommend_endpoint_invalid_user_id_format(flask_client):
    """Test recommendation endpoint with non-numeric user_id"""
    with patch('server.trainset', Mock()):
        response = flask_client.get('/recommend/invalid_user')

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'must be an integer' in data['error']


def test_recommend_endpoint_numeric_user_id_as_string(flask_client, sample_trainset_with_numeric_ids):
    """Test recommendation endpoint accepts numeric string user_id"""
    mock_predictions = []
    for i, (uid, iid, est) in enumerate([
        (1, 101, 4.5),
        (1, 102, 4.2),
    ]):
        pred = Mock()
        pred.uid = uid
        pred.iid = iid
        pred.est = est
        mock_predictions.append(pred)

    with patch('homealone.inference.server.trainset', sample_trainset_with_numeric_ids):
        with patch('homealone.inference.server.model_svd') as mock_model:
            with patch('homealone.inference.server.fallback_pool', []):
                mock_model.test.return_value = mock_predictions

                response = flask_client.get('/recommend/1')

                assert response.status_code == 200


def test_recommend_endpoint_unknown_user_uses_fallback(flask_client, sample_trainset_with_numeric_ids, sample_fallback_pool):
    """Test that unknown user gets fallback recommendations"""
    with patch('homealone.inference.server.trainset', sample_trainset_with_numeric_ids):
        with patch('homealone.inference.server.fallback_pool', sample_fallback_pool):
            with patch('homealone.inference.server.get_random_fallback', return_value="movie1,movie2,movie3"):
                response = flask_client.get('/recommend/999')

                assert response.status_code == 200
                data = response.get_json()
                assert data == "movie1,movie2,movie3"


def test_recommend_endpoint_user_rated_all_items(flask_client, sample_trainset_with_numeric_ids, sample_fallback_pool):
    """Test recommendation when user has rated all items"""
    # Mock a scenario where user has rated everything
    modified_trainset = sample_trainset_with_numeric_ids

    with patch('homealone.inference.server.trainset', modified_trainset):
        with patch('homealone.inference.server.fallback_pool', sample_fallback_pool):
            with patch('homealone.inference.server.get_random_fallback', return_value="movie1,movie2"):
                # Try to get recommendations for a user
                # Mock the unrated_items to be empty by patching
                with patch.object(modified_trainset, 'ur') as mock_ur:
                    with patch.object(modified_trainset, 'all_items', return_value=set([0, 1])):
                        # User has rated items 0 and 1
                        mock_ur.__getitem__.return_value = [(0, 4.5), (1, 4.0)]

                        response = flask_client.get('/recommend/1')

                        # Should get fallback since no unrated items
                        assert response.status_code == 200


def test_recommend_endpoint_successful_recommendation(flask_client, sample_trainset_with_numeric_ids):
    """Test successful recommendation generation"""
    mock_predictions = []
    for i, (uid, iid, est) in enumerate([
        (1, 101, 4.5),
        (1, 102, 4.2),
        (1, 103, 4.0),
    ]):
        pred = Mock()
        pred.uid = uid
        pred.iid = iid
        pred.est = est
        mock_predictions.append(pred)

    with patch('homealone.inference.server.trainset', sample_trainset_with_numeric_ids):
        with patch('homealone.inference.server.model_svd') as mock_model:
            with patch('homealone.inference.server.fallback_pool', []):
                mock_model.test.return_value = mock_predictions

                response = flask_client.get('/recommend/1')

                assert response.status_code == 200
                data = response.get_json()
                assert isinstance(data, str)
                assert '101' in data or '102' in data or '103' in data


# ============================================================================
# Test: Database Connection Management
# ============================================================================

def test_get_db_creates_connection(temp_test_db):
    """Test that get_db creates a database connection"""
    from homealone.inference import server

    with app.app_context():
        with patch.object(server, 'DATABASE', temp_test_db):
            db = get_db()

            assert db is not None
            assert isinstance(db, sqlite3.Connection)


def test_get_db_returns_same_connection(temp_test_db):
    """Test that get_db returns same connection within request context"""
    from homealone.inference import server

    with app.app_context():
        with patch.object(server, 'DATABASE', temp_test_db):
            db1 = get_db()
            db2 = get_db()

            assert db1 is db2, "Should return same connection within context"


def test_database_connection_wal_mode(temp_test_db):
    """Test that database connection uses WAL mode for concurrency"""
    from homealone.inference import server

    with app.app_context():
        with patch.object(server, 'DATABASE', temp_test_db):
            db = get_db()
            cursor = db.cursor()

            # Check WAL mode is enabled
            cursor.execute('PRAGMA journal_mode')
            result = cursor.fetchone()[0]

            assert result.upper() == 'WAL', "Should use WAL journal mode"


# ============================================================================
# Test: Resource Loading
# ============================================================================

def test_load_resources_file_not_found():
    """Test that load_resources raises error when files don't exist"""
    with patch('server.TRAINSET_PATH', '/nonexistent/trainset.dump'):
        with patch('server.SVD_MODEL_PATH', '/nonexistent/svd.dump'):
            with patch('server.KNN_MODEL_PATH', '/nonexistent/knn.dump'):
                with patch('server.NMF_MODEL_PATH', '/nonexistent/nmf.dump'):
                    with pytest.raises(FileNotFoundError):
                        load_resources()


def test_load_resources_loads_all_models(temp_model_files):
    """Test that load_resources successfully loads all models"""
    # Import the server module to patch its variables
    from homealone.inference import server

    with patch.object(server, 'TRAINSET_PATH', temp_model_files['trainset']):
        with patch.object(server, 'SVD_MODEL_PATH', temp_model_files['svd']):
            with patch.object(server, 'KNN_MODEL_PATH', temp_model_files['knn']):
                with patch.object(server, 'NMF_MODEL_PATH', temp_model_files['nmf']):
                    with patch.object(server, 'MODEL_DIR', temp_model_files['dir']):
                        # Mock prepare_fallback_pool to avoid complex setup
                        with patch.object(server, 'prepare_fallback_pool', return_value=[]):
                            load_resources()

                            assert server.trainset is not None
                            assert server.model_svd is not None
                            assert server.model_knn is not None
                            assert server.model_nmf is not None


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================

def test_recommend_with_empty_fallback_pool(flask_client, sample_trainset_with_numeric_ids):
    """Test recommendation when fallback pool is empty"""
    with patch('server.trainset', sample_trainset_with_numeric_ids):
        with patch('server.fallback_pool', []):
            with patch('server.get_random_fallback', return_value=""):
                response = flask_client.get('/recommend/999')

                assert response.status_code == 200


def test_trainset_id_mapping_consistency(sample_trainset_with_numeric_ids):
    """Test that trainset maintains consistent ID mappings"""
    trainset = sample_trainset_with_numeric_ids

    # Test round-trip conversion
    raw_uid = 1
    inner_uid = trainset.to_inner_uid(raw_uid)
    back_to_raw = trainset.to_raw_uid(inner_uid)

    assert back_to_raw == raw_uid, "Round-trip conversion should preserve ID"


def test_trainset_all_items_method(sample_trainset_with_numeric_ids):
    """Test that trainset.all_items() returns valid item set"""
    trainset = sample_trainset_with_numeric_ids

    all_items = trainset.all_items()

    assert isinstance(all_items, (list, range)), "Should return iterable"
    assert len(list(all_items)) == trainset.n_items, "Should return all items"


def test_fallback_pool_with_single_movie():
    """Test fallback pool with only one movie"""
    data_dict = {
        'user_id': [1, 2, 3],
        'movie_id': [101, 101, 101],
        'rating': [5, 4, 5]
    }
    import pandas as pd
    df = pd.DataFrame(data_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    pool = prepare_fallback_pool(trainset, min_ratings=2, min_avg_rating=4.0)

    assert len(pool) == 1, "Should have exactly one movie"
    assert pool[0]['movie_id'] == 101
    assert pool[0]['avg_rating'] >= 4.0


# ============================================================================
# Test: Integration Tests
# ============================================================================

def test_complete_recommendation_flow(flask_client, sample_trainset_with_numeric_ids, sample_fallback_pool):
    """Test complete flow from API request to response"""
    mock_predictions = []
    for i in range(20):
        pred = Mock()
        pred.uid = 1
        pred.iid = 100 + i
        pred.est = 5.0 - (i * 0.1)
        mock_predictions.append(pred)

    with patch('homealone.inference.server.trainset', sample_trainset_with_numeric_ids):
        with patch('homealone.inference.server.model_svd') as mock_model:
            with patch('homealone.inference.server.fallback_pool', sample_fallback_pool):
                mock_model.test.return_value = mock_predictions

                response = flask_client.get('/recommend/1')

                assert response.status_code == 200
                data = response.get_json()

                # Should return comma-separated movie IDs
                assert isinstance(data, str)
                movie_ids = data.split(',')
                assert len(movie_ids) <= 20, "Should return at most 20 recommendations"
