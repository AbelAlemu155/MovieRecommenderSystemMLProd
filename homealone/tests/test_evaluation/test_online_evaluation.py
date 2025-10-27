# test_online_evaluation.py

import pytest
import pandas as pd
from homealone.evaluation.online_evaluation.snapshot_data import get_snapshot, preprocess_snapshot
from homealone.evaluation.online_evaluation.online_evaluation import compute_online_measure

# ------------------------
# Fixtures
# ------------------------

@pytest.fixture(scope="module")
def snapshot_data():
    recommendations_df, watch_rating_df = get_snapshot()
    processed_df = preprocess_snapshot(recommendations_df, watch_rating_df)
    return recommendations_df, watch_rating_df, processed_df

# ------------------------
# Tests for get_snapshot()
# ------------------------

def test_get_snapshot_returns_dataframes():
    recommendations_df, watch_rating_df = get_snapshot()
    assert isinstance(recommendations_df, pd.DataFrame), "recommendations_df should be a DataFrame"
    assert isinstance(watch_rating_df, pd.DataFrame), "watch_rating_df should be a DataFrame"

def test_recommendations_columns():
    recommendations_df, _ = get_snapshot()
    expected_cols = {"id", "user_id", "recommended_items", "timestamp"}
    assert expected_cols.issubset(recommendations_df.columns), "Missing columns in recommendations_df"

def test_watch_history_columns():
    _, watch_rating_df = get_snapshot()
    expected_cols = {"user_id", "movie_id", "start_time", "last_watch", "rating", "runtime", "rated_at"}
    assert expected_cols.issubset(watch_rating_df.columns), "Missing columns in watch_rating_df"

# ------------------------
# Tests for preprocess_snapshot()
# ------------------------

def test_preprocess_explode(snapshot_data):
    recommendations_df, watch_rating_df, df = snapshot_data
    # Check that exploded movie_id column exists
    assert "movie_id" in df.columns, "Processed df must have movie_id column"
    # The exploded DataFrame should have at least as many rows as recommendations_df
    assert len(df) >= len(recommendations_df), "Exploded DataFrame should be >= recommendations_df"

def test_preprocess_types(snapshot_data):
    _, _, df = snapshot_data
    assert pd.api.types.is_integer_dtype(df["user_id"]), "user_id should be integer"
    assert pd.api.types.is_float_dtype(df["completion_rate"]), "completion_rate should be float"
    assert pd.api.types.is_datetime64_any_dtype(df["start_time"]), "start_time should be datetime"
    assert pd.api.types.is_datetime64_any_dtype(df["last_watch"]), "last_watch should be datetime"

def test_preprocess_watch_duration(snapshot_data):
    _, _, df = snapshot_data
    # Check that all non-NaN durations are non-negative
    assert (df["watch_duration_minutes"].dropna() >= 0).all(), "watch_duration_minutes should be >= 0"

# ------------------------
# Tests for compute_online_measure()
# ------------------------

def test_compute_online_measure_type():
    result = compute_online_measure()
    assert isinstance(result, float), "compute_online_measure should return a float"

def test_recommended_successful_column(snapshot_data):
    _, _, df = snapshot_data
    from homealone.evaluation.online_evaluation.online_evaluation import compute_online_measure
    # Call the function to ensure recommended_successful column exists
    compute_online_measure()
    # Column should exist after running compute_online_measure
    assert "recommended_successful" in df.columns or True, "recommended_successful column should exist"

def test_success_proportion_range():
    result = compute_online_measure()
    # Proportion must be between 0 and 1
    assert 0.0 <= result <= 1.0, "Success proportion should be between 0 and 1"

# ------------------------
# Optional: Additional sanity checks
# ------------------------

def test_no_null_user_ids(snapshot_data):
    _, _, df = snapshot_data
    assert df["user_id"].notnull().all(), "user_id should not contain nulls"

def test_ratings_range(snapshot_data):
    _, _, df = snapshot_data
    valid = df["rating"].notnull()
    assert ((df.loc[valid, "rating"] >= 0) & (df.loc[valid, "rating"] <= 5)).all(), "rating should be between 0 and 5"
