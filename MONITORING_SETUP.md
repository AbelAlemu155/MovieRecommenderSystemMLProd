# Monitoring Infrastructure Setup Guide

## Quick Start

### 1. Build and Start Monitoring Service

```bash
# Build the monitoring container
docker-compose build monitoring

# Start all services (including monitoring)
docker-compose up -d

# Check monitoring service logs
docker-compose logs -f monitoring
```

### 2. Access Monitoring Endpoints

- **Prometheus Metrics:** http://localhost:8000/metrics
- **Recommendation API:** http://localhost:8082/recommend/<user_id>
- **Generated Reports:** `./homealone/reports/`

### 3. Verify Monitoring is Working

```bash
# Check if monitoring container is running
docker-compose ps monitoring

# View real-time monitoring logs
docker-compose logs -f monitoring

# Check generated reports
ls -lh homealone/reports/
```

## What Was Implemented

### ✅ Service Health Monitoring
- **Location:** [homealone/monitoring/service_health.py](homealone/monitoring/service_health.py)
- **What it does:**
  - Parses Kafka logs to track recommendation request success/failure
  - Computes availability percentage (% of successful responses)
  - Tracks response times (avg, P95, P99)
  - Monitors error rates
  - **Supports A/B testing:** Separate metrics per model version
- **Alerts:**
  - Critical: Availability < 70%
  - Warning: Availability < 90%
  - Warning: P95 response time > 500ms

### ✅ Model Quality Monitoring
- **Location:** [homealone/monitoring/model_quality_monitor.py](homealone/monitoring/model_quality_monitor.py)
- **What it does:**
  - Computes online accuracy by matching recommendations with actual user watches
  - Metrics: Hit Rate, Precision@K, Engagement Rate
  - **Supports A/B testing:** Separate quality metrics per model version
- **Alerts:**
  - Critical: Hit rate < 10%
  - Warning: Hit rate < 20% or Engagement < 5%

### ✅ Data Drift Monitoring
- **Location:** [homealone/monitoring/data_drift_monitor.py](homealone/monitoring/data_drift_monitor.py) (already existed, now integrated)
- **What it does:**
  - Detects distribution changes using KS test and Wasserstein distance
  - Multi-window detection (hourly, daily, weekly)
  - Generates visualizations (time series plots, heatmaps)
  - Creates HTML summary reports

### ✅ Unified Monitoring System
- **Location:** [homealone/monitoring/unified_monitor.py](homealone/monitoring/unified_monitor.py)
- **What it does:**
  - Orchestrates all monitoring components
  - Centralized alert management
  - Generates comprehensive monitoring dashboard (JSON)
  - Configurable scheduling (production vs test mode)
  - **Alert notifications:** Console, file, email (optional), Slack (optional)

### ✅ Prometheus Metrics Exporter
- **Location:** [homealone/monitoring/prometheus_exporter.py](homealone/monitoring/prometheus_exporter.py)
- **What it does:**
  - Exports all metrics in Prometheus format on port 8000
  - Ready for Grafana visualization
  - Metrics include: availability, quality, drift, alerts

### ✅ Docker Integration
- **Updated:** [docker-compose.yml](docker-compose.yml:52-70)
- Service name: `monitoring`
- Ports: 8000 (Prometheus metrics)
- Depends on: `db_writer`, `inference`
- Volumes: Shared database and reports directory

### ✅ Telemetry Updates
- **Updated:** [homealone/inference/telemetry.py](homealone/inference/telemetry.py)
- Now supports `model_version` field for A/B testing
- Recommendations table updated to track model versions

## Monitoring Dashboard Structure

```
homealone/reports/
├── monitoring_dashboard.json    # Unified dashboard (updated every 15 min)
├── health/                       # Service health reports
├── quality/                      # Model quality reports
├── drift/                        # Drift detection visualizations
│   ├── hourly/
│   │   └── drift_trends_hourly.png
│   ├── daily/
│   │   └── drift_trends_daily.png
│   └── weekly/
│       └── drift_trends_weekly.png
└── alerts/
    └── alerts.jsonl              # All alerts log
```

## Monitoring Schedule

**Production Mode (default):**
- Health check: Every 5 minutes
- Quality check: Every hour (at :10)
- Drift check: Every hour (at :15)
- Dashboard update: Every 15 minutes

**Test Mode (TEST_MODE=true):**
- All checks: Every 5 minutes

## Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Availability | < 90% | < 70% |
| Error Rate | - | > 10% |
| Response Time (P95) | > 500ms | - |
| Hit Rate | < 20% | < 10% |
| Engagement Rate | < 5% | - |
| Data Drift (Norm. WD) | > 0.15 | > 0.3 |

## Database Schema

### New Tables Created

1. **service_health_metrics** - Health metrics over time
2. **recommendation_requests** - Parsed Kafka logs
3. **health_alerts** - Health-related alerts
4. **model_quality_metrics** - Quality metrics over time
5. **quality_alerts** - Quality-related alerts

### Updated Tables

- **recommendations** - Added `model_version` field (removed UNIQUE constraint on user_id)

## A/B Testing Support

The monitoring system fully supports A/B testing:

### How to Use:

1. **Tag model versions in inference service:**
   ```python
   # In server.py, when logging telemetry:
   telemetry.log('recommendation', {
       'user_id': user_id,
       'recommended_items': recommendations,
       'timestamp': timestamp,
       'model_version': 'svd'  # or 'knn', 'nmf', 'v1', 'v2', etc.
   })
   ```

2. **Monitor per-model metrics:**
   ```python
   # Health metrics per model
   health_monitor.compute_model_version_metrics(window_minutes=60)

   # Quality metrics per model
   quality_monitor.compute_model_version_quality(window_hours=24)
   ```

3. **Compare models in Grafana:**
   - Group by `model_version` label in Prometheus queries
   - Create comparison dashboards

## Optional: Email/Slack Alerts

### Email Configuration

Edit [homealone/monitoring/unified_monitor.py](homealone/monitoring/unified_monitor.py:863):

```python
alert_config = {
    'email_enabled': True,
    'smtp': {
        'host': 'smtp.gmail.com',
        'port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',  # Use app password, not account password
        'from_email': 'your-email@gmail.com',
        'to_email': 'team@example.com'
    }
}
```

### Slack Configuration

1. Create Slack webhook: https://api.slack.com/messaging/webhooks
2. Add to config:
   ```python
   alert_config = {
       'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
   }
   ```

## Grafana Setup (Optional but Recommended)

### 1. Install Grafana

Add to [docker-compose.yml](docker-compose.yml):

```yaml
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      - monitoring

volumes:
  grafana-storage:
```

### 2. Configure Prometheus Data Source

1. Access Grafana: http://localhost:3000 (admin/admin)
2. Add data source: Configuration → Data Sources → Add Prometheus
3. URL: `http://monitoring:8000`
4. Save & Test

### 3. Create Dashboards

**Recommended Panels:**

1. **Service Availability Gauge**
   ```promql
   recommendation_service_availability_percentage{window="60m"}
   ```

2. **Response Time Graph**
   ```promql
   histogram_quantile(0.95, recommendation_response_time_ms)
   ```

3. **Model Comparison (Hit Rate)**
   ```promql
   recommendation_model_hit_rate
   ```
   - Group by: `model_version`

4. **Data Drift Heatmap**
   ```promql
   recommendation_data_drift_wasserstein_normalized
   ```

5. **Active Alerts**
   ```promql
   sum(recommendation_active_alerts)
   ```

## Viewing Monitoring Data

### 1. Check Prometheus Metrics

```bash
curl http://localhost:8000/metrics | grep recommendation
```

### 2. View JSON Dashboard

```bash
cat homealone/reports/monitoring_dashboard.json | jq .
```

### 3. Query Database Directly

```bash
sqlite3 homealone/db/home_alone.db

-- Recent availability
SELECT timestamp, availability_percentage, error_rate
FROM service_health_metrics
ORDER BY timestamp DESC LIMIT 10;

-- Recent quality
SELECT timestamp, model_version, hit_rate, precision_at_k
FROM model_quality_metrics
ORDER BY timestamp DESC LIMIT 10;

-- Recent drift
SELECT timestamp, window_type, normalized_wd, drift_detected
FROM drift_metrics
ORDER BY timestamp DESC LIMIT 10;

-- Active alerts
SELECT * FROM health_alerts WHERE resolved = 0;
SELECT * FROM quality_alerts WHERE resolved = 0;
```

### 4. View Generated Visualizations

```bash
# Open drift plots in browser
open homealone/reports/drift/hourly/drift_trends_hourly.png
open homealone/reports/drift/drift_report.html
```

## Troubleshooting

### Monitoring container keeps restarting

**Check logs:**
```bash
docker-compose logs monitoring
```

**Common issues:**
- Database not accessible: Ensure db_writer is running first
- Missing Kafka credentials: Check .env file
- Port 8000 already in use: Change in docker-compose.yml

### No metrics appearing

**Verify:**
1. Inference service is running and serving requests
2. Telemetry is being written to database:
   ```bash
   sqlite3 homealone/db/home_alone.db "SELECT COUNT(*) FROM recommendations;"
   ```
3. Kafka logs contain recommendation requests

### Alerts not triggering

**Check:**
1. Alert thresholds in monitor classes
2. Data exists in monitoring tables
3. Alert manager logs for errors

## Next Steps

### For Milestone 3 Report

Document the following:

1. **Monitoring Infrastructure (0.5 pages text max, excluding figures)**
   - Briefly describe the three monitoring components (health, quality, drift)
   - Include screenshot of monitoring dashboard or Prometheus metrics
   - Provide links to implementation files

2. **What We Monitor:**
   - **(a) Service Availability:** Analyzed from Kafka logs, not just server telemetry
   - **(b) Model Quality:** Online accuracy using hit rate and precision@K from M2
   - **(c) Data Drift:** KS test and Wasserstein distance from M2

3. **Alerts:**
   - Automated alerts for: low availability (<70%), high error rate (>10%), low hit rate (<10%), critical drift (>0.3)
   - Alert logs saved to `/app/reports/alerts/alerts.jsonl`
   - Optional email/Slack notifications configured

4. **A/B Testing Support:**
   - All metrics tracked separately per model version
   - Model version comparison available via Prometheus/Grafana
   - Telemetry updated to include model_version field

## Files Created/Modified

### New Files
- `homealone/monitoring/service_health.py` (413 lines)
- `homealone/monitoring/model_quality_monitor.py` (404 lines)
- `homealone/monitoring/unified_monitor.py` (437 lines)
- `homealone/monitoring/prometheus_exporter.py` (339 lines)
- `homealone/monitoring/kafka_log_reader.py` (73 lines)
- `homealone/monitoring/README.md` (comprehensive documentation)

### Modified Files
- `homealone/monitoring/Dockerfile` (updated for unified monitoring)
- `homealone/monitoring/requirements.txt` (added dependencies)
- `homealone/inference/telemetry.py` (added model_version support)
- `docker-compose.yml` (renamed drift_monitor → monitoring, added port 8000)

## Resources

- [Full Monitoring Documentation](homealone/monitoring/README.md)
- [Prometheus Metrics Reference](http://localhost:8000/metrics)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)

---

**Questions?** Check the [monitoring README](homealone/monitoring/README.md) or contact the team.
