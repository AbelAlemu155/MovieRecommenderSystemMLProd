# test_service_health.py

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import numpy as np
from datetime import datetime
import json

from homealone.monitoring.service_health import ServiceHealthMonitor


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
    """Fixture for ServiceHealthMonitor instance"""
    return ServiceHealthMonitor(db_path, reports_dir)


# ============================================================================
# Tests for Initialization
# ============================================================================

def test_init_creates_required_directories(monitor, mock_makedirs, reports_dir):
    """Test that initialization creates required directories"""
    assert mock_makedirs.called
    # Should create reports_dir and reports_dir/health
    calls = mock_makedirs.call_args_list
    assert any(reports_dir in str(call) for call in calls)


def test_init_establishes_database_connection(db_path, monitor, mock_db_connection):
    """Test that initialization connects to database"""
    mock_conn, _ = mock_db_connection
    mock_conn.execute.assert_any_call('PRAGMA journal_mode=WAL')


def test_init_creates_tables(mock_db_connection):
    """Test that all required tables are created"""
    mock_conn, mock_cursor = mock_db_connection

    with patch('os.makedirs'):
        monitor = ServiceHealthMonitor('/test/db.db', '/test/reports')

    execute_calls = [str(call) for call in mock_cursor.execute.call_args_list]
    assert any('CREATE TABLE IF NOT EXISTS service_health_metrics' in call for call in execute_calls)
    assert any('CREATE TABLE IF NOT EXISTS recommendation_requests' in call for call in execute_calls)
    assert any('CREATE TABLE IF NOT EXISTS health_alerts' in call for call in execute_calls)


# ============================================================================
# Tests for Kafka Log Parsing
# ============================================================================

def test_parse_kafka_recommendation_log_valid_format(monitor):
    """Test parsing of valid Kafka recommendation log"""
    log_line = "2025-10-27T10:30:00,123,recommendation request server1, status 200, result: m1,m2,m3, 150ms"

    result = monitor.parse_kafka_recommendation_log(log_line)

    assert result is not None
    assert result['user_id'] == 123
    assert result['status'] == 200
    assert result['recommendations'] == 'm1,m2,m3'
    assert result['response_time_ms'] == 150.0


def test_parse_kafka_recommendation_log_with_space_before_ms(monitor):
    """Test parsing log with space before 'ms'"""
    log_line = "2025-10-27T10:30:00,456,recommendation request server2, status 200, result: m4,m5, 200 ms"

    result = monitor.parse_kafka_recommendation_log(log_line)

    assert result is not None
    assert result['response_time_ms'] == 200.0


def test_parse_kafka_recommendation_log_invalid_format(monitor):
    """Test that invalid log format returns None"""
    log_line = "invalid log format"

    result = monitor.parse_kafka_recommendation_log(log_line)

    assert result is None


def test_parse_kafka_recommendation_log_error_status(monitor):
    """Test parsing log with error status"""
    log_line = "2025-10-27T10:30:00,789,recommendation request server3, status 500, result: error, 100ms"

    result = monitor.parse_kafka_recommendation_log(log_line)

    assert result is not None
    assert result['status'] == 500
    assert result['recommendations'] == 'error'


# ============================================================================
# Tests for Availability Metrics Computation
# ============================================================================

@patch('pandas.read_sql_query')
def test_compute_availability_metrics_success(mock_read_sql, monitor):
    """Test successful computation of availability metrics"""
    # Mock aggregated data
    mock_agg_data = pd.DataFrame({
        'total_requests': [100],
        'successful_requests': [95],
        'failed_requests': [5],
        'avg_response_time': [150.0],
        'model_version': [None]
    })

    # Mock response time data
    mock_response_times = pd.DataFrame({
        'response_time_ms': [100, 150, 200, 180, 120]
    })

    mock_read_sql.side_effect = [mock_agg_data, mock_response_times]

    result = monitor.compute_availability_metrics(window_minutes=60)

    assert result is not None
    assert result['total_requests'] == 100
    assert result['successful_requests'] == 95
    assert result['availability_percentage'] == 95.0
    assert result['error_rate'] == 5.0


@patch('pandas.read_sql_query')
def test_compute_availability_metrics_no_data(mock_read_sql, monitor):
    """Test that empty data returns None"""
    mock_read_sql.return_value = pd.DataFrame()

    result = monitor.compute_availability_metrics(window_minutes=60)

    assert result is None


@patch('pandas.read_sql_query')
def test_compute_availability_metrics_calculates_percentiles(mock_read_sql, monitor):
    """Test that response time percentiles are calculated"""
    mock_agg_data = pd.DataFrame({
        'total_requests': [50],
        'successful_requests': [45],
        'failed_requests': [5],
        'avg_response_time': [200.0],
        'model_version': [None]
    })

    # Generate response time data
    response_times = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    mock_response_times = pd.DataFrame({'response_time_ms': response_times})

    mock_read_sql.side_effect = [mock_agg_data, mock_response_times]

    result = monitor.compute_availability_metrics(window_minutes=60)

    assert result['p95_response_time_ms'] is not None
    assert result['p99_response_time_ms'] is not None


# ============================================================================
# Tests for Model Version Metrics
# ============================================================================

@patch('pandas.read_sql_query')
def test_compute_model_version_metrics(mock_read_sql, monitor):
    """Test computation of per-model-version metrics"""
    mock_data = pd.DataFrame({
        'model_version': ['svd', 'knn'],
        'total_requests': [50, 30],
        'successful_requests': [48, 28],
        'avg_response_time': [150.0, 180.0],
        'unique_users': [25, 15]
    })

    mock_read_sql.return_value = mock_data

    result = monitor.compute_model_version_metrics(window_minutes=60)

    assert len(result) == 2
    assert result[0]['model_version'] == 'svd'
    assert result[0]['availability_percentage'] == 96.0
    assert result[1]['model_version'] == 'knn'


# ============================================================================
# Tests for Alert Checking
# ============================================================================

def test_check_alerts_critical_low_availability(monitor):
    """Test that critical alert is triggered for low availability"""
    metrics = {
        'availability_percentage': 65.0,
        'error_rate': 15.0,
        'p95_response_time_ms': 300.0
    }

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['severity'] == 'critical' for a in alerts)
    assert any(a['alert_type'] == 'low_availability' for a in alerts)


def test_check_alerts_warning_availability(monitor):
    """Test that warning alert is triggered for borderline availability"""
    metrics = {
        'availability_percentage': 85.0,
        'error_rate': 5.0,
        'p95_response_time_ms': 300.0
    }

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['severity'] == 'warning' for a in alerts)


def test_check_alerts_high_error_rate(monitor):
    """Test that alert is triggered for high error rate"""
    metrics = {
        'availability_percentage': 95.0,
        'error_rate': 12.0,
        'p95_response_time_ms': 300.0
    }

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['alert_type'] == 'high_error_rate' for a in alerts)


def test_check_alerts_slow_response(monitor):
    """Test that alert is triggered for slow response times"""
    metrics = {
        'availability_percentage': 98.0,
        'error_rate': 2.0,
        'p95_response_time_ms': 600.0
    }

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['alert_type'] == 'slow_response' for a in alerts)


def test_check_alerts_no_alerts_for_good_metrics(monitor):
    """Test that no alerts are triggered for good metrics"""
    metrics = {
        'availability_percentage': 99.0,
        'error_rate': 1.0,
        'p95_response_time_ms': 200.0
    }

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) == 0


# ============================================================================
# Tests for Metrics Persistence
# ============================================================================

def test_save_health_metrics(monitor, mock_db_connection):
    """Test that health metrics are saved to database"""
    mock_conn, mock_cursor = mock_db_connection

    metrics = {
        'window_minutes': 60,
        'total_requests': 100,
        'successful_requests': 95,
        'failed_requests': 5,
        'availability_percentage': 95.0,
        'avg_response_time_ms': 150.0,
        'p95_response_time_ms': 300.0,
        'p99_response_time_ms': 400.0,
        'error_rate': 5.0
    }

    monitor.save_health_metrics(metrics)

    mock_cursor.execute.assert_called()
    execute_args = mock_cursor.execute.call_args[0]
    assert 'INSERT INTO service_health_metrics' in execute_args[0]
    mock_conn.commit.assert_called()


def test_save_health_metrics_handles_none(monitor):
    """Test that None metrics are handled gracefully"""
    # Should not raise exception
    monitor.save_health_metrics(None)


# ============================================================================
# Tests for Health Summary
# ============================================================================

@patch('pandas.read_sql_query')
def test_get_health_summary(mock_read_sql, monitor):
    """Test health summary generation"""
    mock_data = pd.DataFrame({
        'avg_availability': [96.5],
        'min_availability': [92.0],
        'avg_error_rate': [3.5],
        'avg_response_time': [180.0],
        'total_requests': [5000]
    })

    mock_read_sql.return_value = mock_data

    result = monitor.get_health_summary(hours=24)

    assert result['hours'] == 24
    assert result['avg_availability'] == 96.5
    assert result['min_availability'] == 92.0
    assert result['total_requests'] == 5000


@patch('pandas.read_sql_query')
def test_get_health_summary_no_data(mock_read_sql, monitor):
    """Test health summary with no data"""
    mock_read_sql.return_value = pd.DataFrame()

    result = monitor.get_health_summary(hours=24)

    assert result['avg_availability'] == 0
    assert result['total_requests'] == 0


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

@pytest.mark.parametrize("availability,expected_severity", [
    (65.0, 'critical'),
    (85.0, 'warning'),
    (95.0, None),
])
def test_alert_severity_levels(monitor, availability, expected_severity):
    """Test that correct alert severity is determined"""
    metrics = {
        'availability_percentage': availability,
        'error_rate': 100 - availability,
        'p95_response_time_ms': 200.0
    }

    alerts = monitor.check_alerts(metrics)

    if expected_severity is None:
        low_avail_alerts = [a for a in alerts if a['alert_type'] == 'low_availability']
        assert len(low_avail_alerts) == 0
    else:
        assert any(a['severity'] == expected_severity for a in alerts)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
