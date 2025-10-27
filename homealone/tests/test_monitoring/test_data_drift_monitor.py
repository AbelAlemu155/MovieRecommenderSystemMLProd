# test_data_drift_monitor.py

import pytest
from unittest.mock import Mock, MagicMock, patch, call, mock_open
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Assuming the module is in homealone.monitoring
from homealone.monitoring.data_drift_monitor import DataDriftMonitor, DriftMonitorScheduler


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
def sample_reference_data():
    """Fixture for sample reference data"""
    return pd.DataFrame({
        'user_id': range(200),
        'movie_id': range(200),
        'rating': np.random.uniform(1, 5, 200),
        'rated_at': ['2025-10-20'] * 200
    })


@pytest.fixture
def sample_current_data():
    """Fixture for sample current data"""
    return pd.DataFrame({
        'user_id': range(100),
        'movie_id': range(100),
        'rating': np.random.uniform(2, 5, 100),
        'rated_at': ['2025-10-27'] * 100
    })


@pytest.fixture
def monitor(db_path, reports_dir, mock_db_connection, mock_makedirs):
    """Fixture for DataDriftMonitor instance"""
    return DataDriftMonitor(db_path, reports_dir)


@pytest.fixture
def sample_drift_metrics():
    """Fixture for sample drift metrics"""
    return {
        'window_type': 'hourly',
        'metric_name': 'rating_distribution',
        'ks_statistic': 0.15,
        'ks_p_value': 0.03,
        'wasserstein_distance': 0.25,
        'normalized_wd': 0.20,
        'ref_mean': 3.0,
        'ref_std': 0.8,
        'curr_mean': 3.2,
        'curr_std': 0.9,
        'ref_sample_size': 200,
        'curr_sample_size': 100,
        'drift_detected': True,
        'drift_severity': 'medium',
        'details': json.dumps({'test': 'data'})
    }


# ============================================================================
# Tests for DataDriftMonitor Initialization
# ============================================================================

def test_init_creates_required_directories(monitor, mock_makedirs, reports_dir):
    """Test that initialization creates all required directories"""
    expected_calls = [
        call(reports_dir, exist_ok=True),
        call(os.path.join(reports_dir, 'hourly'), exist_ok=True),
        call(os.path.join(reports_dir, 'daily'), exist_ok=True),
        call(os.path.join(reports_dir, 'weekly'), exist_ok=True)
    ]
    mock_makedirs.assert_has_calls(expected_calls, any_order=False)


def test_init_establishes_database_connection(db_path, monitor, mock_db_connection):
    """Test that initialization connects to database"""
    mock_conn, _ = mock_db_connection
    mock_conn.execute.assert_any_call('PRAGMA journal_mode=WAL')


def test_init_sets_window_configurations(monitor):
    """Test that initialization sets correct window configurations"""
    assert 'hourly' in monitor.windows
    assert 'daily' in monitor.windows
    assert 'weekly' in monitor.windows
    assert monitor.windows['hourly']['hours'] == 1
    assert monitor.windows['daily']['hours'] == 24
    assert monitor.windows['weekly']['hours'] == 168


def test_init_initializes_cache(monitor):
    """Test that cache is initialized properly"""
    assert monitor.reference_cache == {}
    assert monitor.last_cache_update is None
    assert monitor.cache_ttl == timedelta(hours=24)


# ============================================================================
# Tests for Database Table Creation
# ============================================================================

def test_create_tables_creates_drift_metrics_table(mock_db_connection):
    """Test that drift_metrics table is created"""
    mock_conn, mock_cursor = mock_db_connection
    
    with patch('os.makedirs'):
        monitor = DataDriftMonitor('/test/db.db', '/test/reports')
    
    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    assert any('CREATE TABLE IF NOT EXISTS drift_metrics' in call for call in execute_calls)


def test_create_tables_creates_index(mock_db_connection):
    """Test that index is created on drift_metrics table"""
    mock_conn, mock_cursor = mock_db_connection
    
    with patch('os.makedirs'):
        monitor = DataDriftMonitor('/test/db.db', '/test/reports')
    
    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    assert any('CREATE INDEX IF NOT EXISTS idx_drift_timestamp' in call for call in execute_calls)


def test_create_tables_commits_changes(mock_db_connection):
    """Test that table creation commits to database"""
    mock_conn, _ = mock_db_connection
    
    with patch('os.makedirs'):
        monitor = DataDriftMonitor('/test/db.db', '/test/reports')
    
    assert mock_conn.commit.called


# ============================================================================
# Tests for Reference Data Management
# ============================================================================

@patch('pandas.read_sql_query')
def test_get_reference_data_queries_database(mock_read_sql, monitor, sample_reference_data):
    """Test that reference data is queried from database"""
    mock_read_sql.return_value = sample_reference_data
    
    result = monitor._get_reference_data()
    
    assert len(result) == 200
    mock_read_sql.assert_called_once()


@patch('pandas.read_sql_query')
def test_get_reference_data_caches_result(mock_read_sql, monitor, sample_reference_data):
    """Test that reference data is cached after first query"""
    mock_read_sql.return_value = sample_reference_data
    
    result1 = monitor._get_reference_data()
    result2 = monitor._get_reference_data()
    
    assert len(result1) == len(result2)
    assert mock_read_sql.call_count == 1


@patch('pandas.read_sql_query')
def test_get_reference_data_force_refresh_bypasses_cache(mock_read_sql, monitor, sample_reference_data):
    """Test that force_refresh parameter bypasses cache"""
    mock_read_sql.return_value = sample_reference_data
    
    monitor._get_reference_data()
    monitor._get_reference_data(force_refresh=True)
    
    assert mock_read_sql.call_count == 2


@patch('pandas.read_sql_query')
def test_get_reference_data_returns_none_for_insufficient_data(mock_read_sql, monitor):
    """Test that insufficient reference data returns None"""
    insufficient_data = pd.DataFrame({
        'user_id': range(50),
        'movie_id': range(50),
        'rating': [3.0] * 50,
        'rated_at': ['2025-10-20'] * 50
    })
    mock_read_sql.return_value = insufficient_data
    
    result = monitor._get_reference_data()
    
    assert result is None


# ============================================================================
# Tests for KS Test Computation
# ============================================================================

def test_compute_ks_test_returns_all_fields(monitor):
    """Test that KS test returns all required fields"""
    ref_data = np.random.normal(3.0, 0.5, 1000)
    curr_data = np.random.normal(3.0, 0.5, 1000)
    
    result = monitor.compute_ks_test(ref_data, curr_data)
    
    assert 'ks_statistic' in result
    assert 'ks_p_value' in result
    assert 'drift_detected' in result
    assert 'interpretation' in result


def test_compute_ks_test_detects_significant_drift(monitor):
    """Test that KS test detects drift between different distributions"""
    ref_data = np.random.normal(3.0, 0.5, 1000)
    curr_data = np.random.normal(4.0, 0.5, 1000)
    
    result = monitor.compute_ks_test(ref_data, curr_data)
    
    assert result['drift_detected'] is True
    assert result['ks_p_value'] < 0.05


def test_compute_ks_test_no_drift_for_similar_distributions(monitor):
    """Test that KS test does not detect drift for similar distributions"""
    np.random.seed(42)
    ref_data = np.random.normal(3.0, 0.5, 1000)
    curr_data = np.random.normal(3.0, 0.5, 1000)
    
    result = monitor.compute_ks_test(ref_data, curr_data)
    
    assert result['drift_detected'] is False
    assert result['ks_p_value'] > 0.05


# ============================================================================
# Tests for Wasserstein Distance Computation
# ============================================================================

def test_compute_wasserstein_distance_returns_all_fields(monitor):
    """Test that Wasserstein distance computation returns all fields"""
    ref_data = np.random.normal(3.0, 0.5, 1000)
    curr_data = np.random.normal(3.2, 0.5, 1000)
    
    result = monitor.compute_wasserstein_distance(ref_data, curr_data)
    
    assert 'wasserstein_distance' in result
    assert 'normalized_wd' in result
    assert 'drift_detected' in result
    assert 'severity' in result
    assert 'interpretation' in result


def test_compute_wasserstein_distance_severity_levels(monitor):
    """Test severity classification based on normalized WD"""
    ref_data = np.random.normal(3.0, 1.0, 1000)
    ref_std = 1.0
    
    # High drift (> 0.3 std)
    high_drift = np.random.normal(3.4, 1.0, 1000)
    result_high = monitor.compute_wasserstein_distance(ref_data, high_drift, ref_std)
    assert result_high['severity'] == 'high'
    
    # No drift (< 0.1 std)
    no_drift = np.random.normal(3.0, 1.0, 1000)
    result_none = monitor.compute_wasserstein_distance(ref_data, no_drift, ref_std)
    assert result_none['severity'] == 'none'


# ============================================================================
# Tests for Drift Detection
# ============================================================================

@patch('pandas.read_sql_query')
def test_detect_rating_drift_returns_none_for_insufficient_reference(mock_read_sql, monitor):
    """Test drift detection returns None when reference data is insufficient"""
    insufficient_ref = pd.DataFrame({
        'user_id': range(50),
        'movie_id': range(50),
        'rating': [3.0] * 50,
        'rated_at': ['2025-10-20'] * 50
    })
    mock_read_sql.return_value = insufficient_ref
    
    result = monitor.detect_rating_drift('hourly')
    
    assert result is None


@patch('pandas.read_sql_query')
def test_detect_rating_drift_success(mock_read_sql, monitor, sample_reference_data, sample_current_data):
    """Test successful rating drift detection"""
    mock_read_sql.side_effect = [sample_reference_data, sample_current_data]
    
    result = monitor.detect_rating_drift('hourly')
    
    assert result is not None
    assert 'timestamp' in result
    assert 'window_type' in result
    assert result['window_type'] == 'hourly'
    assert 'ks_p_value' in result
    assert 'normalized_wd' in result


# ============================================================================
# Tests for Metrics Persistence
# ============================================================================

def test_save_drift_metrics_inserts_into_database(monitor, sample_drift_metrics, mock_db_connection):
    """Test that drift metrics are inserted into database"""
    mock_conn, mock_cursor = mock_db_connection
    
    monitor.save_drift_metrics(sample_drift_metrics)
    
    mock_cursor.execute.assert_called()
    execute_args = mock_cursor.execute.call_args[0]
    assert 'INSERT INTO drift_metrics' in execute_args[0]
    mock_conn.commit.assert_called()


def test_save_drift_metrics_handles_none_gracefully(monitor):
    """Test that None metrics are handled without error"""
    # Should not raise exception
    monitor.save_drift_metrics(None)


# ============================================================================
# Tests for Visualization (Fixed)
# ============================================================================

@patch('pandas.read_sql_query')
@patch('matplotlib.pyplot.close')  # This works
def test_plot_drift_trends_creates_plot_files(mock_close, mock_read_sql, monitor):
    """Test that drift trend plotting creates and saves plot files"""
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-10-20', periods=10, freq='h'),  # Fixed: use 'h' instead of 'H'
        'ks_p_value': np.random.uniform(0.01, 0.5, 10),
        'normalized_wd': np.random.uniform(0, 0.4, 10),
        'ref_mean': [3.0] * 10,
        'curr_mean': np.random.uniform(2.8, 3.2, 10),
        'drift_detected': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        'drift_severity': ['none'] * 10
    })
    mock_read_sql.return_value = mock_data
    
    # Mock the figure's savefig method
    with patch('matplotlib.figure.Figure.savefig') as mock_savefig:
        result = monitor.plot_drift_trends('hourly', days=7, save=True)
        
        # Verify savefig was called once (main)
        assert mock_savefig.call_count == 1
        mock_close.assert_called_once()
        assert result is not None



@patch('pandas.read_sql_query')
def test_plot_drift_trends_handles_no_data(mock_read_sql, monitor):
    """Test that plotting handles empty data gracefully"""
    mock_read_sql.return_value = pd.DataFrame()
    
    result = monitor.plot_drift_trends('hourly', days=7, save=True)
    
    assert result is None


# ============================================================================
# Tests for Connection Management
# ============================================================================

def test_close_closes_database_connection(monitor, mock_db_connection):
    """Test that close() properly closes database connection"""
    mock_conn, _ = mock_db_connection
    
    monitor.close()
    
    mock_conn.close.assert_called_once()


# ============================================================================
# Tests for DriftMonitorScheduler (Fixed)
# ============================================================================

@pytest.fixture
def mock_monitor_class():
    """Fixture for mocked DataDriftMonitor class"""
    # Fix: Correct module path for patching
    with patch('homealone.monitoring.data_drift_monitor.DataDriftMonitor') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock, mock_instance


def test_scheduler_init_creates_monitor(db_path, reports_dir, mock_monitor_class):
    """Test that scheduler initialization creates a monitor instance"""
    mock_class, mock_instance = mock_monitor_class
    
    scheduler = DriftMonitorScheduler(db_path, reports_dir)
    
    assert scheduler.monitor is not None
    assert scheduler.running is False


def test_scheduler_init_test_mode_enabled(db_path, reports_dir, mock_monitor_class):
    """Test scheduler initialization with test mode enabled"""
    scheduler = DriftMonitorScheduler(db_path, reports_dir, test_mode=True)
    
    assert scheduler.test_mode is True


def test_scheduler_run_hourly_check(db_path, reports_dir, mock_monitor_class):
    """Test that hourly check executes drift detection"""
    _, mock_instance = mock_monitor_class
    scheduler = DriftMonitorScheduler(db_path, reports_dir)
    
    scheduler.run_hourly_check()
    
    mock_instance.run_drift_detection.assert_called_once_with(['hourly'])


def test_scheduler_run_daily_check(db_path, reports_dir, mock_monitor_class):
    """Test that daily check executes detection and visualization"""
    _, mock_instance = mock_monitor_class
    scheduler = DriftMonitorScheduler(db_path, reports_dir)
    
    scheduler.run_daily_check()
    
    mock_instance.run_drift_detection.assert_called_with(['daily'])
    assert mock_instance.plot_drift_trends.called
    assert mock_instance.generate_summary_report.called


def test_scheduler_stop(db_path, reports_dir, mock_monitor_class):
    """Test that stop() halts scheduler and closes monitor"""
    _, mock_instance = mock_monitor_class
    scheduler = DriftMonitorScheduler(db_path, reports_dir)
    scheduler.running = True
    
    scheduler.stop()
    
    assert scheduler.running is False
    mock_instance.close.assert_called_once()


# ============================================================================
# Parametrized Tests
# ============================================================================

@pytest.mark.parametrize("window_type,expected_hours", [
    ('hourly', 1),
    ('daily', 24),
    ('weekly', 168),
])
def test_window_configurations(monitor, window_type, expected_hours):
    """Test that window configurations are set correctly"""
    assert monitor.windows[window_type]['hours'] == expected_hours


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
