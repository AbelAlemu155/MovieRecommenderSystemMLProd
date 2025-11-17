# test_model_quality.py

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from homealone.monitoring.model_quality_monitor import ModelQualityMonitor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def db_path():
    """Fixture for test database path"""
    return '/test/db/test.db'


@pytest.fixture
def reports_dir():
    """Fixture for test reports directory"""
    return '/test/reports'


@pytest.fixture
def mock_db_connection():
    """Fixture for mocked database connection"""
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn, mock_cursor


@pytest.fixture
def mock_makedirs():
    """Fixture for mocked os.makedirs"""
    with patch('os.makedirs') as mock:
        yield mock


@pytest.fixture
def monitor(db_path, reports_dir, mock_db_connection, mock_makedirs):
    """Fixture for ModelQualityMonitor instance"""
    return ModelQualityMonitor(db_path, reports_dir)


@pytest.fixture
def sample_recommendations():
    """Fixture for sample recommendations data"""
    return pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'recommendations': ['m1,m2,m3,m4', 'm5,m6,m7', 'm1,m8,m9', 'm2,m3,m4', 'm10,m11'],
        'timestamp': ['2025-10-27T10:00:00'] * 5,
        'model_version': ['svd'] * 5
    })


@pytest.fixture
def sample_watch_history():
    """Fixture for sample watch history data"""
    return pd.DataFrame({
        'user_id': [1, 1, 2, 3, 4],
        'movie_id': ['m1', 'm2', 'm5', 'm9', 'm3'],
        'start_time': ['2025-10-27T11:00:00'] * 5
    })


# ============================================================================
# Tests for Initialization
# ============================================================================

def test_init_creates_required_directories(monitor, mock_makedirs, reports_dir):
    """Test that initialization creates required directories"""
    assert mock_makedirs.called


def test_init_establishes_database_connection(monitor, mock_db_connection):
    """Test that initialization connects to database"""
    mock_conn, _ = mock_db_connection
    mock_conn.execute.assert_any_call('PRAGMA journal_mode=WAL')


def test_init_creates_tables(mock_db_connection):
    """Test that all required tables are created"""
    mock_conn, mock_cursor = mock_db_connection

    with patch('os.makedirs'):
        monitor = ModelQualityMonitor('/test/db.db', '/test/reports')

    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    assert any('CREATE TABLE IF NOT EXISTS model_quality_metrics' in call for call in execute_calls)
    assert any('CREATE TABLE IF NOT EXISTS quality_alerts' in call for call in execute_calls)


# ============================================================================
# Tests for Online Accuracy Computation
# ============================================================================

@patch('pandas.read_sql_query')
@patch('pandas.to_datetime')
def test_compute_online_accuracy_success(mock_to_datetime, mock_read_sql, monitor, sample_recommendations, sample_watch_history):
    """Test successful computation of online accuracy metrics"""
    # Mock datetime conversion
    mock_to_datetime.side_effect = lambda x, **kwargs: pd.to_datetime(x) if isinstance(x, pd.Series) else x

    mock_read_sql.side_effect = [sample_recommendations, sample_watch_history]

    result = monitor.compute_online_accuracy(window_hours=24, k=20)

    assert result is not None
    assert 'hit_rate' in result
    assert 'precision_at_k' in result
    assert 'engagement_rate' in result
    assert result['total_users'] == 5


@patch('pandas.read_sql_query')
def test_compute_online_accuracy_no_recommendations(mock_read_sql, monitor):
    """Test that no recommendations returns None"""
    mock_read_sql.return_value = pd.DataFrame()

    result = monitor.compute_online_accuracy(window_hours=24, k=20)

    assert result is None


@patch('pandas.read_sql_query')
@patch('pandas.to_datetime')
def test_compute_online_accuracy_calculates_hit_rate(mock_to_datetime, mock_read_sql, monitor):
    """Test hit rate calculation"""
    # User 1 watches m1, m2 (2 hits from recommendations m1,m2,m3,m4)
    # User 2 watches m5 (1 hit from recommendations m5,m6,m7)
    # User 3 watches nothing (0 hits)

    mock_to_datetime.side_effect = lambda x, **kwargs: pd.to_datetime(x) if isinstance(x, pd.Series) else x

    recommendations = pd.DataFrame({
        'user_id': [1, 2, 3],
        'recommendations': ['m1,m2,m3,m4', 'm5,m6,m7', 'm8,m9,m10'],
        'timestamp': ['2025-10-27T10:00:00'] * 3,
        'model_version': [None] * 3
    })

    watches = pd.DataFrame({
        'user_id': [1, 1, 2],
        'movie_id': ['m1', 'm2', 'm5'],
        'start_time': ['2025-10-27T11:00:00'] * 3
    })

    mock_read_sql.side_effect = [recommendations, watches]

    result = monitor.compute_online_accuracy(window_hours=24, k=20)

    # 2 out of 3 users had hits
    assert result['hit_rate'] == pytest.approx(0.6667, abs=0.01)


@patch('pandas.read_sql_query')
@patch('pandas.to_datetime')
def test_compute_online_accuracy_calculates_precision(mock_to_datetime, mock_read_sql, monitor):
    """Test precision@K calculation"""
    mock_to_datetime.side_effect = lambda x, **kwargs: pd.to_datetime(x) if isinstance(x, pd.Series) else x

    recommendations = pd.DataFrame({
        'user_id': [1],
        'recommendations': ['m1,m2,m3,m4,m5'],  # 5 recommendations
        'timestamp': ['2025-10-27T10:00:00'],
        'model_version': [None]
    })

    watches = pd.DataFrame({
        'user_id': [1, 1],
        'movie_id': ['m1', 'm3'],  # Watched 2 out of 5
        'start_time': ['2025-10-27T11:00:00'] * 2
    })

    mock_read_sql.side_effect = [recommendations, watches]

    result = monitor.compute_online_accuracy(window_hours=24, k=5)

    # Precision = 2/5 = 0.4
    assert result['precision_at_k'] == 0.4


# ============================================================================
# Tests for Model Version Quality
# ============================================================================

@patch('pandas.read_sql_query')
@patch('pandas.to_datetime')
def test_compute_model_version_quality(mock_to_datetime, mock_read_sql, monitor):
    """Test computation of per-model-version quality metrics"""
    mock_to_datetime.side_effect = lambda x, **kwargs: pd.to_datetime(x) if isinstance(x, pd.Series) else x

    # Mock query to get model versions
    versions_df = pd.DataFrame({'model_version': ['svd', 'knn']})

    # Mock recommendations for each version
    svd_recs = pd.DataFrame({
        'user_id': [1],
        'recommendations': ['m1,m2'],
        'timestamp': ['2025-10-27T10:00:00'],
        'model_version': ['svd']
    })

    knn_recs = pd.DataFrame({
        'user_id': [2],
        'recommendations': ['m3,m4'],
        'timestamp': ['2025-10-27T10:00:00'],
        'model_version': ['knn']
    })

    watches = pd.DataFrame({
        'user_id': [1, 2],
        'movie_id': ['m1', 'm3'],
        'start_time': ['2025-10-27T11:00:00'] * 2
    })

    # Setup side_effect for multiple calls
    mock_read_sql.side_effect = [
        versions_df,  # First call to get versions
        svd_recs, watches,  # SVD model
        knn_recs, watches   # KNN model
    ]

    result = monitor.compute_model_version_quality(window_hours=24, k=20)

    assert len(result) == 2
    assert result[0]['model_version'] == 'svd'
    assert result[1]['model_version'] == 'knn'


# ============================================================================
# Tests for Alert Checking
# ============================================================================

def test_check_quality_alerts_critical_low_hit_rate(monitor):
    """Test that critical alert is triggered for very low hit rate"""
    metrics = {
        'hit_rate': 0.08,
        'engagement_rate': 0.03,
        'model_version': 'svd'
    }

    alerts = monitor.check_quality_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['severity'] == 'critical' for a in alerts)
    assert any(a['alert_type'] == 'low_hit_rate' for a in alerts)


def test_check_quality_alerts_warning_hit_rate(monitor):
    """Test that warning alert is triggered for borderline hit rate"""
    metrics = {
        'hit_rate': 0.15,
        'engagement_rate': 0.08,
        'model_version': 'svd'
    }

    alerts = monitor.check_quality_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['severity'] == 'warning' for a in alerts)


def test_check_quality_alerts_low_engagement(monitor):
    """Test that alert is triggered for low engagement rate"""
    metrics = {
        'hit_rate': 0.25,
        'engagement_rate': 0.03,
        'model_version': 'svd'
    }

    alerts = monitor.check_quality_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['alert_type'] == 'low_engagement' for a in alerts)


def test_check_quality_alerts_no_alerts_for_good_metrics(monitor):
    """Test that no alerts are triggered for good metrics"""
    metrics = {
        'hit_rate': 0.35,
        'engagement_rate': 0.15,
        'model_version': 'svd'
    }

    alerts = monitor.check_quality_alerts(metrics)

    assert len(alerts) == 0


# ============================================================================
# Tests for Metrics Persistence
# ============================================================================

def test_save_quality_metrics(monitor, mock_db_connection):
    """Test that quality metrics are saved to database"""
    mock_conn, mock_cursor = mock_db_connection

    metrics = {
        'window_hours': 24,
        'model_version': 'svd',
        'hit_rate': 0.25,
        'precision_at_k': 0.05,
        'engagement_rate': 0.10,
        'avg_rank_watched': 5.2,
        'total_recommendations': 100,
        'total_users': 100,
        'k_value': 20
    }

    monitor.save_quality_metrics(metrics)

    mock_cursor.execute.assert_called()
    execute_args = mock_cursor.execute.call_args[0]
    assert 'INSERT INTO model_quality_metrics' in execute_args[0]
    mock_conn.commit.assert_called()


def test_save_quality_metrics_handles_none(monitor):
    """Test that None metrics are handled gracefully"""
    # Should not raise exception
    monitor.save_quality_metrics(None)


# ============================================================================
# Tests for Quality Report Generation
# ============================================================================

@patch('pandas.read_sql_query')
def test_generate_quality_report(mock_read_sql, monitor):
    """Test quality report generation"""
    mock_data = pd.DataFrame({
        'avg_hit_rate': [0.25, 0.22],
        'avg_precision': [0.05, 0.04],
        'avg_engagement': [0.08, 0.07],
        'avg_rank': [5.5, 6.0],
        'total_recommendations': [500, 300],
        'model_version': ['svd', 'knn']
    })

    mock_read_sql.return_value = mock_data

    result = monitor.generate_quality_report(hours=24)

    assert result['hours'] == 24
    assert 'overall' in result
    assert 'by_version' in result
    assert len(result['by_version']) == 2


@patch('pandas.read_sql_query')
def test_generate_quality_report_no_data(mock_read_sql, monitor):
    """Test quality report with no data"""
    mock_read_sql.return_value = pd.DataFrame()

    result = monitor.generate_quality_report(hours=24)

    assert result['overall'] == {}
    assert result['by_version'] == []


# ============================================================================
# Tests for Quality Trend
# ============================================================================

@patch('pandas.read_sql_query')
def test_get_quality_trend(mock_read_sql, monitor):
    """Test quality trend data retrieval"""
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-10-20', periods=10, freq='h'),
        'hit_rate': np.random.uniform(0.2, 0.3, 10),
        'precision_at_k': np.random.uniform(0.04, 0.06, 10),
        'engagement_rate': np.random.uniform(0.06, 0.10, 10),
        'avg_rank_watched': np.random.uniform(4, 7, 10),
        'model_version': ['svd'] * 10
    })

    mock_read_sql.return_value = mock_data

    result = monitor.get_quality_trend(hours=168)

    assert len(result) == 10
    assert 'timestamp' in result.columns
    assert 'hit_rate' in result.columns


# ============================================================================
# Tests for Connection Management
# ============================================================================

def test_close_closes_database_connection(monitor, mock_db_connection):
    """Test that close() properly closes database connection"""
    mock_conn, _ = mock_db_connection

    monitor.close()

    mock_conn.close.assert_called_once()


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("hit_rate,expected_severity", [
    (0.08, 'critical'),
    (0.15, 'warning'),
    (0.30, None),
])
def test_alert_severity_levels(monitor, hit_rate, expected_severity):
    """Test that correct alert severity is determined"""
    metrics = {
        'hit_rate': hit_rate,
        'engagement_rate': 0.10,
        'model_version': 'test'
    }

    alerts = monitor.check_quality_alerts(metrics)

    if expected_severity is None:
        low_hit_alerts = [a for a in alerts if a['alert_type'] == 'low_hit_rate']
        assert len(low_hit_alerts) == 0
    else:
        assert any(a['severity'] == expected_severity for a in alerts)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
