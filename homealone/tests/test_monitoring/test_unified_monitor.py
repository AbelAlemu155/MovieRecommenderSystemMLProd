# test_unified_monitor.py

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import json

from homealone.monitoring.unified_monitor import UnifiedMonitor, AlertManager


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
def mock_monitors():
    """Fixture for mocked monitor classes"""
    with patch('homealone.monitoring.unified_monitor.ServiceHealthMonitor') as mock_health, \
         patch('homealone.monitoring.unified_monitor.ModelQualityMonitor') as mock_quality, \
         patch('homealone.monitoring.unified_monitor.DataDriftMonitor') as mock_drift:

        yield {
            'health': mock_health,
            'quality': mock_quality,
            'drift': mock_drift
        }


@pytest.fixture
def alert_manager():
    """Fixture for AlertManager instance"""
    with patch('os.makedirs'):
        return AlertManager()


@pytest.fixture
def unified_monitor(db_path, reports_dir, mock_monitors):
    """Fixture for UnifiedMonitor instance"""
    with patch('os.makedirs'):
        return UnifiedMonitor(db_path, reports_dir)


# ============================================================================
# Tests for AlertManager
# ============================================================================

def test_alert_manager_init(alert_manager):
    """Test AlertManager initialization"""
    assert alert_manager.alert_history == []
    assert alert_manager.reports_dir is not None


@patch('builtins.open', create=True)
def test_alert_manager_send_alert_logs_to_file(mock_open, alert_manager):
    """Test that alerts are logged to file"""
    alert = {
        'source': 'test',
        'alert_type': 'test_alert',
        'severity': 'warning',
        'message': 'Test alert message'
    }

    alert_manager.send_alert(alert)

    mock_open.assert_called()
    assert len(alert_manager.alert_history) == 1


def test_alert_manager_send_alert_adds_timestamp(alert_manager):
    """Test that timestamp is added to alerts"""
    alert = {
        'source': 'test',
        'alert_type': 'test_alert',
        'severity': 'warning',
        'message': 'Test message'
    }

    with patch('builtins.open', create=True):
        alert_manager.send_alert(alert)

    assert 'timestamp' in alert_manager.alert_history[0]


@patch('builtins.open', create=True)
@patch('requests.post')
def test_alert_manager_sends_slack_alert(mock_post, mock_open, alert_manager):
    """Test Slack webhook integration"""
    alert_manager.config = {'slack_webhook_url': 'https://hooks.slack.com/test'}

    alert = {
        'source': 'test',
        'alert_type': 'test_alert',
        'severity': 'critical',
        'message': 'Critical issue',
        'metric_value': 50.0,
        'threshold_value': 70.0
    }

    alert_manager.send_alert(alert)

    mock_post.assert_called_once()


def test_alert_manager_get_recent_alerts(alert_manager):
    """Test retrieving recent alerts"""
    with patch('builtins.open', create=True):
        for i in range(5):
            alert = {
                'source': 'test',
                'alert_type': f'alert_{i}',
                'severity': 'warning',
                'message': f'Alert {i}'
            }
            alert_manager.send_alert(alert)

    recent = alert_manager.get_recent_alerts(hours=24)

    assert len(recent) == 5


# ============================================================================
# Tests for UnifiedMonitor Initialization
# ============================================================================

def test_unified_monitor_init_creates_all_monitors(unified_monitor, mock_monitors):
    """Test that all monitoring components are initialized"""
    assert unified_monitor.health_monitor is not None
    assert unified_monitor.quality_monitor is not None
    assert unified_monitor.drift_monitor is not None
    assert unified_monitor.alert_manager is not None


def test_unified_monitor_init_with_alert_config():
    """Test initialization with alert configuration"""
    alert_config = {
        'email_enabled': True,
        'smtp': {'host': 'smtp.test.com'}
    }

    with patch('homealone.monitoring.unified_monitor.ServiceHealthMonitor'), \
         patch('homealone.monitoring.unified_monitor.ModelQualityMonitor'), \
         patch('homealone.monitoring.unified_monitor.DataDriftMonitor'), \
         patch('os.makedirs'):

        monitor = UnifiedMonitor('/test/db.db', '/test/reports', alert_config)

        assert monitor.alert_manager.config == alert_config


# ============================================================================
# Tests for Health Check
# ============================================================================

def test_run_health_check(unified_monitor):
    """Test health check execution"""
    mock_metrics = {
        'availability_percentage': 95.0,
        'total_requests': 100
    }

    unified_monitor.health_monitor.compute_availability_metrics.return_value = mock_metrics
    unified_monitor.health_monitor.check_alerts.return_value = []
    unified_monitor.health_monitor.compute_model_version_metrics.return_value = []

    unified_monitor.run_health_check()

    unified_monitor.health_monitor.compute_availability_metrics.assert_called_once()
    unified_monitor.health_monitor.save_health_metrics.assert_called_once_with(mock_metrics)


def test_run_health_check_triggers_alerts(unified_monitor):
    """Test that health check triggers alerts"""
    mock_metrics = {
        'availability_percentage': 60.0,
        'total_requests': 100
    }

    mock_alerts = [{
        'alert_type': 'low_availability',
        'severity': 'critical',
        'message': 'Low availability'
    }]

    unified_monitor.health_monitor.compute_availability_metrics.return_value = mock_metrics
    unified_monitor.health_monitor.check_alerts.return_value = mock_alerts
    unified_monitor.health_monitor.compute_model_version_metrics.return_value = []

    with patch.object(unified_monitor.alert_manager, 'send_alert') as mock_send:
        unified_monitor.run_health_check()

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert call_args['source'] == 'service_health'


def test_run_health_check_handles_no_metrics(unified_monitor):
    """Test health check when no metrics are available"""
    unified_monitor.health_monitor.compute_availability_metrics.return_value = None

    # Should not raise exception
    unified_monitor.run_health_check()


# ============================================================================
# Tests for Quality Check
# ============================================================================

def test_run_quality_check(unified_monitor):
    """Test quality check execution"""
    mock_metrics = {
        'hit_rate': 0.25,
        'precision_at_k': 0.05
    }

    unified_monitor.quality_monitor.compute_online_accuracy.return_value = mock_metrics
    unified_monitor.quality_monitor.check_quality_alerts.return_value = []
    unified_monitor.quality_monitor.compute_model_version_quality.return_value = []

    unified_monitor.run_quality_check()

    unified_monitor.quality_monitor.compute_online_accuracy.assert_called_once()
    unified_monitor.quality_monitor.save_quality_metrics.assert_called_once_with(mock_metrics)


def test_run_quality_check_triggers_alerts(unified_monitor):
    """Test that quality check triggers alerts"""
    mock_metrics = {
        'hit_rate': 0.08,
        'precision_at_k': 0.02
    }

    mock_alerts = [{
        'alert_type': 'low_hit_rate',
        'severity': 'critical',
        'message': 'Hit rate too low'
    }]

    unified_monitor.quality_monitor.compute_online_accuracy.return_value = mock_metrics
    unified_monitor.quality_monitor.check_quality_alerts.return_value = mock_alerts
    unified_monitor.quality_monitor.compute_model_version_quality.return_value = []

    with patch.object(unified_monitor.alert_manager, 'send_alert') as mock_send:
        unified_monitor.run_quality_check()

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert call_args['source'] == 'model_quality'


# ============================================================================
# Tests for Drift Check
# ============================================================================

def test_run_drift_check(unified_monitor):
    """Test drift check execution"""
    unified_monitor.drift_monitor.detect_rating_drift.return_value = {
        'drift_detected': False,
        'normalized_wd': 0.05
    }

    unified_monitor.run_drift_check()

    unified_monitor.drift_monitor.run_drift_detection.assert_called_once_with(['hourly', 'daily'])


def test_run_drift_check_triggers_alert_on_high_drift(unified_monitor):
    """Test that drift check triggers alert for high drift"""
    unified_monitor.drift_monitor.detect_rating_drift.return_value = {
        'drift_detected': True,
        'drift_severity': 'high',
        'normalized_wd': 0.35
    }

    with patch.object(unified_monitor.alert_manager, 'send_alert') as mock_send:
        unified_monitor.run_drift_check()

        mock_send.assert_called_once()
        call_args = mock_send.call_args[0][0]
        assert call_args['source'] == 'data_drift'
        assert call_args['severity'] == 'critical'


def test_run_drift_check_no_alert_on_low_drift(unified_monitor):
    """Test that no alert is triggered for low drift"""
    unified_monitor.drift_monitor.detect_rating_drift.return_value = {
        'drift_detected': False,
        'drift_severity': 'none',
        'normalized_wd': 0.05
    }

    with patch.object(unified_monitor.alert_manager, 'send_alert') as mock_send:
        unified_monitor.run_drift_check()

        mock_send.assert_not_called()


# ============================================================================
# Tests for Dashboard Generation
# ============================================================================

def test_generate_monitoring_dashboard(unified_monitor):
    """Test dashboard generation"""
    unified_monitor.health_monitor.get_health_summary.return_value = {'avg_availability': 95.0}
    unified_monitor.health_monitor.compute_availability_metrics.return_value = {'availability_percentage': 98.0}
    unified_monitor.quality_monitor.generate_quality_report.return_value = {'overall': {'avg_hit_rate': 0.25}}
    unified_monitor.drift_monitor.generate_drift_report.return_value = {'hourly': {}}

    with patch('builtins.open', create=True):
        dashboard = unified_monitor.generate_monitoring_dashboard()

    assert 'timestamp' in dashboard
    assert 'health' in dashboard
    assert 'quality' in dashboard
    assert 'drift' in dashboard
    assert 'alerts' in dashboard


@patch('builtins.open', create=True)
def test_generate_monitoring_dashboard_saves_to_file(mock_open, unified_monitor):
    """Test that dashboard is saved to file"""
    unified_monitor.health_monitor.get_health_summary.return_value = {}
    unified_monitor.health_monitor.compute_availability_metrics.return_value = {}
    unified_monitor.quality_monitor.generate_quality_report.return_value = {}
    unified_monitor.drift_monitor.generate_drift_report.return_value = {}

    unified_monitor.generate_monitoring_dashboard()

    mock_open.assert_called()
    # Verify JSON was written
    handle = mock_open()
    handle.write.assert_called()


def test_generate_monitoring_dashboard_handles_errors(unified_monitor):
    """Test that dashboard generation handles component errors gracefully"""
    unified_monitor.health_monitor.get_health_summary.side_effect = Exception("Health error")
    unified_monitor.quality_monitor.generate_quality_report.return_value = {}
    unified_monitor.drift_monitor.generate_drift_report.return_value = {}

    with patch('builtins.open', create=True):
        dashboard = unified_monitor.generate_monitoring_dashboard()

    assert 'health' in dashboard
    assert 'error' in dashboard['health']


# ============================================================================
# Tests for Monitoring Close
# ============================================================================

def test_close_closes_all_monitors(unified_monitor):
    """Test that close() closes all monitoring components"""
    unified_monitor.close()

    unified_monitor.health_monitor.close.assert_called_once()
    unified_monitor.quality_monitor.close.assert_called_once()
    unified_monitor.drift_monitor.close.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_monitoring_cycle(unified_monitor):
    """Test a full monitoring cycle with all checks"""
    # Setup mock returns
    unified_monitor.health_monitor.compute_availability_metrics.return_value = {'availability_percentage': 95.0}
    unified_monitor.health_monitor.check_alerts.return_value = []
    unified_monitor.health_monitor.compute_model_version_metrics.return_value = []

    unified_monitor.quality_monitor.compute_online_accuracy.return_value = {'hit_rate': 0.25}
    unified_monitor.quality_monitor.check_quality_alerts.return_value = []
    unified_monitor.quality_monitor.compute_model_version_quality.return_value = []

    unified_monitor.drift_monitor.detect_rating_drift.return_value = {'drift_detected': False}

    # Run all checks
    unified_monitor.run_health_check()
    unified_monitor.run_quality_check()
    unified_monitor.run_drift_check()

    # Verify all monitors were called
    assert unified_monitor.health_monitor.compute_availability_metrics.called
    assert unified_monitor.quality_monitor.compute_online_accuracy.called
    assert unified_monitor.drift_monitor.run_drift_detection.called


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
