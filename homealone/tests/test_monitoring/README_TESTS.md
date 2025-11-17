# Monitoring Infrastructure Tests

Comprehensive test suite for the monitoring infrastructure components.

## Test Files

### 1. **test_service_health.py** (350+ lines, 25 tests)
Tests for Service Health Monitor that analyzes Kafka logs.

**Test Coverage:**
- ✅ Initialization and database connection
- ✅ Table creation (service_health_metrics, recommendation_requests, health_alerts)
- ✅ Kafka log parsing (valid/invalid formats, error statuses)
- ✅ Availability metrics computation
- ✅ Response time percentile calculation (P95, P99)
- ✅ Per-model-version metrics (A/B testing support)
- ✅ Alert threshold checking (critical <70%, warning <90%)
- ✅ Metrics persistence to database
- ✅ Health summary generation
- ✅ Connection management

**Key Tests:**
- `test_parse_kafka_recommendation_log_valid_format()` - Verifies log parsing
- `test_compute_availability_metrics_calculates_percentiles()` - Tests metric calculation
- `test_check_alerts_critical_low_availability()` - Validates alerting
- `test_compute_model_version_metrics()` - Tests A/B testing support

### 2. **test_model_quality.py** (330+ lines, 22 tests)
Tests for Model Quality Monitor that tracks online accuracy.

**Test Coverage:**
- ✅ Initialization and database connection
- ✅ Table creation (model_quality_metrics, quality_alerts)
- ✅ Online accuracy computation (hit rate, precision@K)
- ✅ Recommendation-watch matching logic
- ✅ Per-model-version quality metrics (A/B testing)
- ✅ Alert threshold checking (critical <10%, warning <20%)
- ✅ Metrics persistence to database
- ✅ Quality report generation
- ✅ Quality trend data retrieval
- ✅ Connection management

**Key Tests:**
- `test_compute_online_accuracy_calculates_hit_rate()` - Tests hit rate calculation
- `test_compute_online_accuracy_calculates_precision()` - Tests precision@K
- `test_compute_model_version_quality()` - Tests A/B testing support
- `test_check_quality_alerts_critical_low_hit_rate()` - Validates alerting

### 3. **test_unified_monitor.py** (280+ lines, 18 tests)
Tests for Unified Monitor that orchestrates all monitoring components.

**Test Coverage:**
- ✅ AlertManager initialization and alerting
- ✅ Multi-channel alert delivery (file, Slack)
- ✅ UnifiedMonitor initialization
- ✅ Health check orchestration
- ✅ Quality check orchestration
- ✅ Drift check orchestration
- ✅ Alert triggering from each component
- ✅ Dashboard generation
- ✅ Error handling in dashboard generation
- ✅ Monitor cleanup
- ✅ Full monitoring cycle integration test

**Key Tests:**
- `test_alert_manager_send_alert_logs_to_file()` - Tests alert logging
- `test_run_health_check_triggers_alerts()` - Tests health monitoring
- `test_run_quality_check_triggers_alerts()` - Tests quality monitoring
- `test_run_drift_check_triggers_alert_on_high_drift()` - Tests drift monitoring
- `test_full_monitoring_cycle()` - Integration test

### 4. **test_data_drift_monitor.py** (490+ lines, existing)
Tests for Data Drift Monitor (already existed).

**Test Coverage:**
- ✅ Drift detection with KS test and Wasserstein distance
- ✅ Reference data caching
- ✅ Multi-window detection (hourly, daily, weekly)
- ✅ Visualization generation
- ✅ Scheduler functionality

## Running Tests

### Run All Monitoring Tests
```bash
pytest homealone/tests/test_monitoring/ -v
```

### Run Specific Test File
```bash
# Service health tests
pytest homealone/tests/test_monitoring/test_service_health.py -v

# Model quality tests
pytest homealone/tests/test_monitoring/test_model_quality.py -v

# Unified monitor tests
pytest homealone/tests/test_monitoring/test_unified_monitor.py -v

# Drift monitor tests (existing)
pytest homealone/tests/test_monitoring/test_data_drift_monitor.py -v
```

### Run with Coverage
```bash
pytest homealone/tests/test_monitoring/ --cov=homealone.monitoring --cov-report=term-missing
```

### Run Specific Test
```bash
pytest homealone/tests/test_monitoring/test_service_health.py::test_parse_kafka_recommendation_log_valid_format -v
```

## Test Statistics

**Total Test Files:** 4
**Total Tests:** ~95 tests
**Total Test Code:** ~1,450 lines

**Test Distribution:**
- Service Health: 25 tests (350 lines)
- Model Quality: 22 tests (330 lines)
- Unified Monitor: 18 tests (280 lines)
- Data Drift: 30 tests (490 lines)

## Test Coverage

The test suite covers:

### **Core Functionality**
- ✅ Database operations (connect, create tables, insert, query)
- ✅ Metric computation (availability, quality, drift)
- ✅ Alert threshold checking
- ✅ Data persistence
- ✅ Report generation

### **Edge Cases**
- ✅ No data scenarios
- ✅ Invalid input formats
- ✅ Error handling
- ✅ None value handling
- ✅ Empty result sets

### **Integration**
- ✅ Component orchestration
- ✅ Multi-channel alerting
- ✅ Full monitoring cycle

### **A/B Testing Support**
- ✅ Per-model-version health metrics
- ✅ Per-model-version quality metrics
- ✅ Model comparison capability

## Mocking Strategy

Tests use comprehensive mocking to avoid external dependencies:

- **Database:** `sqlite3.connect` mocked with MagicMock
- **File I/O:** `os.makedirs`, `open()` mocked
- **Network:** `requests.post` mocked for Slack alerts
- **External Monitors:** Component classes mocked in integration tests
- **Time:** `datetime` operations use fixed timestamps
- **Data Queries:** `pandas.read_sql_query` returns controlled test data

## Test Fixtures

Common fixtures used across all test files:

```python
@pytest.fixture
def db_path():
    """Test database path"""
    return '/test/db/test.db'

@pytest.fixture
def reports_dir():
    """Test reports directory"""
    return '/test/reports'

@pytest.fixture
def mock_db_connection():
    """Mocked database connection"""
    # Returns (mock_conn, mock_cursor)

@pytest.fixture
def monitor():
    """Monitor instance with mocked dependencies"""
```

## Parametrized Tests

Several tests use parametrization to test multiple scenarios:

```python
@pytest.mark.parametrize("availability,expected_severity", [
    (65.0, 'critical'),
    (85.0, 'warning'),
    (95.0, None),
])
def test_alert_severity_levels(monitor, availability, expected_severity):
    """Test alert severity determination"""
```

## Test Organization

```
homealone/tests/test_monitoring/
├── __init__.py
├── test_service_health.py          # Service health monitor tests
├── test_model_quality.py           # Model quality monitor tests
├── test_unified_monitor.py         # Unified monitor tests
├── test_data_drift_monitor.py      # Drift monitor tests (existing)
└── README_TESTS.md                 # This file
```

## Continuous Integration

Tests are run automatically via GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run tests with coverage
  run: |
    pytest --cov=homealone --cov-report=term-missing homealone/tests --cov-fail-under=70
```

## Expected Test Results

All tests should pass with the following characteristics:

```
======================== test session starts =========================
collected 95 items

test_service_health.py::test_init_creates_required_directories PASSED
test_service_health.py::test_parse_kafka_recommendation_log_valid_format PASSED
test_service_health.py::test_compute_availability_metrics_success PASSED
...
test_model_quality.py::test_compute_online_accuracy_success PASSED
test_model_quality.py::test_compute_online_accuracy_calculates_hit_rate PASSED
...
test_unified_monitor.py::test_run_health_check PASSED
test_unified_monitor.py::test_generate_monitoring_dashboard PASSED
...
test_data_drift_monitor.py::test_compute_ks_test_detects_significant_drift PASSED
...

======================== 95 passed in 2.50s ==========================
```

## Common Test Patterns

### Testing Database Operations
```python
def test_save_metrics(monitor, mock_db_connection):
    mock_conn, mock_cursor = mock_db_connection

    monitor.save_health_metrics(metrics)

    mock_cursor.execute.assert_called()
    assert 'INSERT INTO' in mock_cursor.execute.call_args[0][0]
    mock_conn.commit.assert_called()
```

### Testing Alert Triggering
```python
def test_check_alerts_critical(monitor):
    metrics = {'availability_percentage': 65.0}

    alerts = monitor.check_alerts(metrics)

    assert len(alerts) > 0
    assert any(a['severity'] == 'critical' for a in alerts)
```

### Testing Metric Computation
```python
@patch('pandas.read_sql_query')
def test_compute_metrics(mock_sql, monitor):
    mock_sql.return_value = test_data

    result = monitor.compute_availability_metrics()

    assert result is not None
    assert result['availability_percentage'] > 0
```

## Debugging Failed Tests

If tests fail:

1. **Check mock setup:**
   ```bash
   pytest homealone/tests/test_monitoring/test_service_health.py::test_name -vv
   ```

2. **View captured output:**
   ```bash
   pytest homealone/tests/test_monitoring/ -v -s
   ```

3. **Run with debugger:**
   ```bash
   pytest homealone/tests/test_monitoring/ --pdb
   ```

4. **Check coverage gaps:**
   ```bash
   pytest homealone/tests/test_monitoring/ --cov=homealone.monitoring --cov-report=html
   open htmlcov/index.html
   ```

## Adding New Tests

When adding new monitoring features:

1. Add tests to appropriate test file
2. Follow existing test patterns
3. Mock external dependencies
4. Test both success and failure cases
5. Add parametrized tests for multiple scenarios
6. Update this README with new test descriptions

## Dependencies for Testing

```python
# Required packages
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
```

Install with:
```bash
pip install pytest pytest-cov pytest-mock
```

## Test Quality Metrics

**Target Metrics:**
- Code Coverage: >80% for monitoring modules
- Test Pass Rate: 100%
- Test Execution Time: <5 seconds
- Tests per Module: Minimum 15 tests per major component

**Current Status:**
- ✅ Service Health Monitor: 25 tests
- ✅ Model Quality Monitor: 22 tests
- ✅ Unified Monitor: 18 tests
- ✅ Data Drift Monitor: 30 tests
- ✅ Total: 95 tests

---

**Last Updated:** 2025-11-08
**Test Suite Version:** 1.0
**Minimum Required Coverage:** 70%
