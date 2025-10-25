import pytest
import pandas as pd
from homealone.evaluation.online_evaluation.snapshot_data import preprocess_snapshot
from homealone.evaluation.online_evaluation import online_evaluation


# ----------------------
# Fixtures for 20-movie recommendations
# ----------------------
@pytest.fixture
def recommendations_20():
    # Each user has 20 movies recommended
    movie_ids_user1 = ','.join([str(i) for i in range(1, 21)])
    movie_ids_user2 = ','.join([str(i) for i in range(21, 41)])
    return pd.DataFrame({
        'user_id': [1, 2],
        'movie_ids': [movie_ids_user1, movie_ids_user2]
    })


@pytest.fixture
def watch_ratings_20():
    # Watch + ratings for 20 movies per user
    return pd.DataFrame({
        'user_id': [1]*20 + [2]*20,
        'movie_id': [str(i) for i in range(1, 21)] + [str(i) for i in range(21, 41)],
        'watch_time': [80]*40,
        'movie_duration': [100]*40,
        'rating': [5]*40
    })


# ----------------------
# Test preprocessing and explode
# ----------------------
def test_preprocess_snapshot_explode_20_movies(recommendations_20, watch_ratings_20):
    df = preprocess_snapshot(recommendations_20, watch_ratings_20)

    # Each user should have 20 rows after explode
    rows_user1 = df[df['user_id'] == 1]
    rows_user2 = df[df['user_id'] == 2]

    assert len(rows_user1) == 20
    assert len(rows_user2) == 20

    # Check types
    assert df['movie_id'].apply(lambda x: isinstance(x, str)).all()
    assert df['user_id'].dtype == int

    # Check completion_rate calculation
    assert df['completion_rate'].between(0, 1).all()


# ----------------------
# Test success calculation logic
# ----------------------
def test_recommended_successful_logic(monkeypatch):
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['1,2']})
    ratings = pd.DataFrame({'user_id': [1,1], 'movie_id': ['1','2'], 'rating': [5,2]})
    watch = pd.DataFrame({'user_id': [1,1], 'movie_id': ['1','2'], 'watch_time': [80,50], 'movie_duration': [100,100]})

    # Monkeypatch get_snapshot to avoid DB access
    def fake_get_snapshot():
        return rec, watch.merge(ratings, on=['user_id','movie_id'], how='left')

    monkeypatch.setattr(online_evaluation, "get_snapshot", fake_get_snapshot)

    # Run the online measure
    success = online_evaluation.compute_online_measure()
    # Only movie 1 is successful (rating>=4), movie 2 fails → 0.5
    assert success == 0.5


# ----------------------
# Test type safety after explode
# ----------------------
def test_movie_id_type_after_explode(recommendations_20, watch_ratings_20):
    df = preprocess_snapshot(recommendations_20, watch_ratings_20)
    assert df['movie_id'].apply(lambda x: isinstance(x, str)).all()


# ----------------------
# Edge case: empty watch and rating data
# ----------------------
def test_empty_watch_rating_data(monkeypatch):
    rec = pd.DataFrame({'user_id': [1], 'movie_ids': ['1,2']})
    watch = pd.DataFrame(columns=['user_id','movie_id','watch_time','movie_duration'])
    ratings = pd.DataFrame(columns=['user_id','movie_id','rating'])

    def fake_get_snapshot():
        return rec, watch.merge(ratings, on=['user_id','movie_id'], how='left')

    monkeypatch.setattr(online_evaluation, "get_snapshot", fake_get_snapshot)

    success = online_evaluation.compute_online_measure()
    # All missing → no successful recommendations
    assert success == 0.0


# ----------------------
# Multiple users with mixed success
# ----------------------
def test_multiple_users_success(monkeypatch):
    rec = pd.DataFrame({'user_id': [1,2], 'movie_ids': ['1,2','3,4']})
    ratings = pd.DataFrame({
        'user_id':[1,1,2,2],
        'movie_id':['1','2','3','4'],
        'rating':[5,3,4,1]
    })
    watch = pd.DataFrame({
        'user_id':[1,1,2,2],
        'movie_id':['1','2','3','4'],
        'watch_time':[90,90,90,90],
        'movie_duration':[100,100,100,100]
    })

    def fake_get_snapshot():
        return rec, watch.merge(ratings, on=['user_id','movie_id'], how='left')

    monkeypatch.setattr(online_evaluation, "get_snapshot", fake_get_snapshot)

    success = online_evaluation.compute_online_measure()
    # Successful: movie 1 (user 1), movie 3 (user 2) → 2/4 = 0.5
    assert success == 0.5
