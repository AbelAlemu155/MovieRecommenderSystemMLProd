import pytest
import pandas as pd
import numpy as np
from homealone.evaluation.online_evaluation import online_evaluation as oe
import pandas as pd
from homealone.evaluation.online_evaluation.snapshot_data import preprocess_snapshot

@pytest.fixture
def mock_df():
    data = {
        "rating": [5, 4, 3, np.nan, np.nan, 2],
        "completion_rate": [0.9, 0.8, 0.6, 0.75, 0.5, 0.4],
        "timestamp": pd.to_datetime([
            "2024-01-01 10:00:00",
            "2024-01-01 10:05:00",
            "2024-01-01 11:00:00",
            "2024-01-02 08:00:00",
            "2024-01-02 09:00:00",
            "2024-01-02 09:10:00"
        ]),
        "rated_at": pd.to_datetime([
            "2024-01-01 10:10:00",
            "2024-01-01 10:06:00",
            "2024-01-01 11:30:00",
            "2024-01-02 09:00:00",
            "2024-01-02 09:30:00",
            "2024-01-02 09:20:00"
        ]),
        "last_watch": pd.to_datetime([
            "2024-01-01 10:30:00",
            "2024-01-01 10:10:00",
            "2024-01-01 11:20:00",
            "2024-01-02 09:10:00",
            "2024-01-02 09:40:00",
            "2024-01-02 09:25:00"
        ])
    }
    return pd.DataFrame(data)


def test_online_measure_all_conditions(monkeypatch, mock_df):
    """Patch compute_online_measure to use mock_df directly."""
    monkeypatch.setattr(oe, "compute_online_measure", lambda: oe._compute_online_measure_from_df(mock_df))
    result = oe._compute_online_measure_from_df(mock_df)
    assert pytest.approx(result, 0.01) == 0.5


def test_measure_with_nan_fields(monkeypatch):
    df = pd.DataFrame({
        "rating": [np.nan, 5],
        "completion_rate": [np.nan, 0.8],
        "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 10:05:00"]),
        "rated_at": pd.to_datetime(["2024-01-01 09:50:00", "2024-01-01 10:06:00"]),
        "last_watch": pd.to_datetime(["2024-01-01 09:40:00", "2024-01-01 10:20:00"])
    })

    result = oe._compute_online_measure_from_df(df)
    assert 0.0 <= result <= 1.0


def test_all_unsuccessful(monkeypatch):
    df = pd.DataFrame({
        "rating": [1, 2, 3],
        "completion_rate": [0.4, 0.3, 0.2],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "rated_at": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "last_watch": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    })
    result = oe._compute_online_measure_from_df(df)
    assert result == 0.0


def test_all_successful(monkeypatch):
    df = pd.DataFrame({
        "rating": [5, 4, 5],
        "completion_rate": [0.9, 0.8, 0.95],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "rated_at": pd.to_datetime(["2023-12-31", "2024-01-01", "2024-01-02"]),
        "last_watch": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    })
    result = oe._compute_online_measure_from_df(df)
    assert result == 1.0




def test_preprocess_snapshot_explode_and_merge():
    """
    Test that preprocess_snapshot correctly:
      - Splits the 'recommended_items' column into a list and explodes it
      - Merges with watch/rating data on user_id and movie_id
      - Produces 20 rows (one per recommended movie)
      - Computes valid watch_duration_minutes and completion_rate values
    """

    # --- Mock recommendations dataframe (like the one from SQLite) ---
    recommendations_df = pd.DataFrame({
        "id": [1],
        "user_id": [42],
        "recommended_items": [",".join([f"m{i}" for i in range(20)])],
        "timestamp": [pd.Timestamp("2025-10-27 12:00:00")],
    })

    # --- Mock watch history + ratings dataframe (joined view from SQL) ---
    watch_rating_df = pd.DataFrame({
        "user_id": [42] * 20,
        "movie_id": [f"m{i}" for i in range(20)],
        "start_time": pd.date_range("2025-10-27 11:00:00", periods=20, freq="min"),
        "last_watch": pd.date_range("2025-10-27 11:30:00", periods=20, freq="min"),
        "rating": [4.5] * 20,
        "runtime": [90.0] * 20,
        "rated_at": pd.date_range("2025-10-27 13:00:00", periods=20, freq="min"),
    })

    # --- Run preprocessing ---
    processed_df = preprocess_snapshot(recommendations_df, watch_rating_df)

    # --- Structural checks ---
    assert isinstance(processed_df, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(processed_df) == 20, "Exploded output should have 20 rows (one per movie)"
    assert "movie_id" in processed_df.columns, "movie_id column must be present"
    assert "watch_duration_minutes" in processed_df.columns, "watch_duration_minutes must be computed"
    assert "completion_rate" in processed_df.columns, "completion_rate must be computed"

    # --- Type checks ---
    assert pd.api.types.is_integer_dtype(processed_df["user_id"]), "user_id must be integer"
    for col in ["start_time", "last_watch", "timestamp"]:
        assert pd.api.types.is_datetime64_any_dtype(processed_df[col]), f"{col} must be datetime"
    assert pd.api.types.is_float_dtype(processed_df["completion_rate"]), "completion_rate should be float"

    # --- Value checks ---
    assert (processed_df["watch_duration_minutes"].dropna() >= 0).all(), \
        "watch_duration_minutes must be non-negative"
    assert (processed_df["completion_rate"].dropna() >= 0).all(), \
        "completion_rate must be non-negative"

    # --- Column renaming and explode behavior ---
    assert "recommended_items" not in processed_df.columns, "recommended_items column must be dropped"
    assert processed_df["movie_id"].nunique() == 20, "There should be 20 unique movies after explode"
    assert processed_df["user_id"].nunique() == 1, "All records must belong to the same user"

    # --- Logical consistency: merged data from both inputs ---
    sample = processed_df.iloc[0]
    assert sample["rating"] == 4.5, "Rating must be preserved after merge"
    assert sample["runtime"] == 90.0, "Runtime must be preserved after merge"
    assert pd.notnull(sample["watch_duration_minutes"]), "watch_duration_minutes must be computed correctly"
    assert pd.notnull(sample["completion_rate"]), "completion_rate must be computed correctly"
