# Monitoring Infrastructure

Comprehensive monitoring system for the Movie Recommendation Service that tracks service health, model quality, and data drift.

## Overview

The monitoring infrastructure consists of three main components:

1. **Service Health Monitor** - Tracks availability, response times, and error rates
2. **Model Quality Monitor** - Tracks prediction quality using online metrics
3. **Data Drift Monitor** - Detects distribution changes in incoming data

All components are orchestrated by a **Unified Monitor** that provides centralized alerting and reporting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Monitor                          │
│  (Orchestrates all monitoring & alerting)                   │
└────────────┬────────────┬────────────┬─────────────────────┘
             │            │            │
     ┌───────┴─────┐ ┌────┴────┐ ┌────┴────────┐
     │   Service   │ │  Model  │ │    Data     │
     │   Health    │ │ Quality │ │    Drift    │
     └──────┬──────┘ └────┬────┘ └─────┬───────┘
            │             │             │
            └─────────────┴─────────────┴──────────────┐
                                                        │
                            ┌───────────────────────────┴────┐
                            │  SQLite Database               │
                            │  - Metrics Storage             │
                            │  - Telemetry Data              │
                            │  - Alert History               │
                            └────────────────────────────────┘
```

## Components

### 1. Service Health Monitor ([service_health.py](service_health.py))

**Purpose:** Monitor availability and performance of the recommendation service by analyzing Kafka logs.

**Metrics Tracked:**
- **Availability:** Percentage of successful responses (status 200) from recommendation service
- **Response Time:** Average, P95, and P99 response times
- **Error Rate:** Percentage of failed/timeout requests
- **Request Volume:** Total number of requests per time window
- **Model Version Performance:** Separate metrics for each model version (A/B testing support)

**Key Features:**
- Parses Kafka logs to determine actual user-perceived availability
- Tracks per-model-version metrics for A/B testing comparisons
- Configurable time windows (1h, 24h, etc.)
- Automatic alert generation for low availability or high error rates

**Alerts:**
- **Critical:** Availability < 70% or Error Rate > 10%
- **Warning:** Availability < 90% or P95 response time > 500ms

**Database Tables:**
- `service_health_metrics` - Historical health metrics
- `recommendation_requests` - Parsed Kafka recommendation logs
- `health_alerts` - Health-related alerts

### 2. Model Quality Monitor ([model_quality_monitor.py](model_quality_monitor.py))

**Purpose:** Track prediction quality in production using actual user behavior.

**Metrics Tracked:**
- **Hit Rate:** Percentage of users who watched at least one recommended movie
- **Precision@K:** Average proportion of recommended movies that were watched
- **Engagement Rate:** Percentage of recommended movies that were watched
- **Average Rank of Watched:** Average position of watched movies in recommendation list
- **Per-Model Metrics:** Separate tracking for each model version (A/B testing)

**Key Features:**
- Matches recommendations with actual watch behavior
- Configurable time windows and K values
- Supports A/B testing with per-model-version quality tracking
- Automatic alert generation for quality degradation

**Alerts:**
- **Critical:** Hit rate < 10%
- **Warning:** Hit rate < 20% or Engagement rate < 5%

**Database Tables:**
- `model_quality_metrics` - Historical quality metrics
- `quality_alerts` - Quality-related alerts

### 3. Data Drift Monitor ([data_drift_monitor.py](data_drift_monitor.py))

**Purpose:** Detect distribution changes in incoming data (ratings, user activity).

**Metrics Tracked:**
- **KS Test p-value:** Statistical test for distribution differences
- **Wasserstein Distance:** Normalized measure of distribution shift
- **Drift Severity:** Categorized as none/low/medium/high
- **Multi-window Detection:** Hourly, daily, and weekly windows

**Key Features:**
- Two-sample Kolmogorov-Smirnov test for distribution comparison
- Wasserstein distance for measuring drift magnitude
- Reference window: 7-14 days ago (stable baseline)
- Automated visualization generation (time series plots, heatmaps)
- HTML summary reports

**Drift Thresholds:**
- **High:** Normalized WD > 0.3 or KS p-value < 0.01
- **Medium:** Normalized WD > 0.15 or KS p-value < 0.05
- **Low:** Normalized WD > 0.1

**Database Tables:**
- `drift_metrics` - Historical drift metrics

### 4. Unified Monitor ([unified_monitor.py](unified_monitor.py))

**Purpose:** Orchestrate all monitoring components with centralized alerting.

**Features:**
- **Coordinated Scheduling:** Runs health/quality/drift checks at optimal intervals
- **Centralized Alerting:** Aggregates alerts from all components
- **Alert Notifications:** Supports console logging, file logging, email (SMTP), and Slack
- **Monitoring Dashboard:** JSON export of all current metrics
- **Test Mode:** Accelerated checks for development/testing

**Alert Manager:**
- Logs all alerts to `/app/reports/alerts/alerts.jsonl`
- Sends email notifications for critical alerts (if configured)
- Posts to Slack webhook (if configured)
- Maintains alert history for analysis

### 5. Prometheus Exporter ([prometheus_exporter.py](prometheus_exporter.py))

**Purpose:** Export metrics in Prometheus format for Grafana visualization.

**Exposed Metrics:**
```
# Service Health
recommendation_service_availability_percentage{window}
recommendation_requests_total{status}
recommendation_response_time_ms (histogram)
recommendation_service_error_rate{window}

# Model Quality
recommendation_model_hit_rate{model_version, window}
recommendation_model_precision_at_k{model_version, k, window}
recommendation_model_engagement_rate{model_version, window}

# Data Drift
recommendation_data_drift_ks_pvalue{window_type, metric_name}
recommendation_data_drift_wasserstein_normalized{window_type, metric_name}
recommendation_data_drift_detected{window_type, severity}

# Alerts
recommendation_alerts_total{alert_type, severity, source}
recommendation_active_alerts{severity}
```

**Port:** 8000 (exposed for Prometheus scraping)

## Deployment

### Docker Deployment

The monitoring system runs as a Docker container defined in [Dockerfile](Dockerfile).

**Build and run:**
```bash
docker-compose up -d monitoring
```

**Access Prometheus metrics:**
```
http://localhost:8000/metrics
```

**View generated reports:**
```
./homealone/reports/
├── health/           # Service health reports
├── quality/          # Model quality reports
├── drift/            # Drift detection reports
│   ├── hourly/
│   ├── daily/
│   └── weekly/
└── alerts/           # Alert logs
    └── alerts.jsonl
```

### Configuration

**Environment Variables:**
- `TEST_MODE=true|false` - Run in test mode (accelerated checks)
- `TUNNEL_USER`, `TUNNEL_IP`, `TUNNEL_PASSWORD`, `TUNNEL_PORT` - Kafka connection for log reading

**Alert Configuration:**

Edit [unified_monitor.py](unified_monitor.py:863) to configure alerts:

```python
alert_config = {
    'email_enabled': True,  # Enable email alerts
    'smtp': {
        'host': 'smtp.gmail.com',
        'port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from_email': 'your-email@gmail.com',
        'to_email': 'team@example.com'
    },
    'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
}
```

## Monitoring Schedule

**Production Mode:**
- **Health Check:** Every 5 minutes
- **Quality Check:** Every hour (at :10)
- **Drift Check:** Every hour (at :15)
- **Dashboard Update:** Every 15 minutes

**Test Mode:**
- All checks run every 5 minutes

## A/B Testing Support

All monitoring components support tracking multiple model versions simultaneously:

1. **Model Version Tagging:** Update [server.py](../inference/server.py) to include model version in telemetry:
   ```python
   telemetry.log('recommendation', {
       'user_id': user_id,
       'recommended_items': recommendations,
       'timestamp': timestamp,
       'model_version': 'v1'  # or 'v2', 'svd', 'knn', etc.
   })
   ```

2. **Separate Metrics:** Health, quality, and drift are tracked separately per model version

3. **Comparison:** Use `compute_model_version_metrics()` and `compute_model_version_quality()` to compare models

## Database Schema

### service_health_metrics
```sql
CREATE TABLE service_health_metrics (
    timestamp DATETIME,
    window_minutes INTEGER,
    total_requests INTEGER,
    successful_requests INTEGER,
    failed_requests INTEGER,
    availability_percentage REAL,
    avg_response_time_ms REAL,
    p95_response_time_ms REAL,
    p99_response_time_ms REAL,
    error_rate REAL,
    model_version TEXT
);
```

### model_quality_metrics
```sql
CREATE TABLE model_quality_metrics (
    timestamp DATETIME,
    window_hours INTEGER,
    model_version TEXT,
    hit_rate REAL,
    precision_at_k REAL,
    recall_at_k REAL,
    ndcg_at_k REAL,
    avg_rank_watched REAL,
    engagement_rate REAL,
    total_recommendations INTEGER,
    total_users INTEGER,
    k_value INTEGER
);
```

### drift_metrics
```sql
CREATE TABLE drift_metrics (
    timestamp DATETIME,
    window_type TEXT,  -- 'hourly', 'daily', 'weekly'
    metric_name TEXT,  -- 'rating_distribution', 'user_activity'
    ks_statistic REAL,
    ks_p_value REAL,
    wasserstein_distance REAL,
    normalized_wd REAL,
    drift_detected BOOLEAN,
    drift_severity TEXT  -- 'none', 'low', 'medium', 'high'
);
```

## Accessing Monitoring Data

### 1. Via Database Queries

```python
import sqlite3
conn = sqlite3.connect('/app/db/home_alone.db')

# Get recent availability
cursor.execute("""
    SELECT timestamp, availability_percentage
    FROM service_health_metrics
    WHERE timestamp >= datetime('now', '-24 hours')
    ORDER BY timestamp DESC
""")
```

### 2. Via JSON Dashboard

```bash
cat ./homealone/reports/monitoring_dashboard.json
```

### 3. Via Prometheus

```bash
curl http://localhost:8000/metrics
```

### 4. Via Grafana

1. Add Prometheus data source: `http://monitoring:8000`
2. Import pre-built dashboards or create custom visualizations

## Grafana Dashboard Setup

**Recommended Panels:**

1. **Service Availability** (Gauge)
   - Query: `recommendation_service_availability_percentage{window="60m"}`
   - Thresholds: Red < 70%, Yellow < 90%, Green >= 90%

2. **Response Time** (Graph)
   - Query: `histogram_quantile(0.95, recommendation_response_time_ms)`
   - Alert: > 500ms

3. **Model Quality Comparison** (Graph)
   - Query: `recommendation_model_hit_rate`
   - Group by: `model_version`

4. **Data Drift Detection** (Heatmap)
   - Query: `recommendation_data_drift_wasserstein_normalized`
   - Group by: `window_type`

5. **Active Alerts** (Stat)
   - Query: `sum(recommendation_active_alerts)`

## Testing

Run monitoring in test mode for faster checks:

```bash
export TEST_MODE=true
python unified_monitor.py
```

## Troubleshooting

### No metrics appearing

**Check:**
1. Database connection: Ensure `/app/db/home_alone.db` exists and is accessible
2. Telemetry logging: Verify inference service is logging to `recommendations` table
3. Kafka logs: Confirm recommendation requests are being logged to Kafka

### Alerts not firing

**Check:**
1. Alert thresholds in each monitor class
2. Alert manager configuration
3. SMTP/Slack credentials (if using external notifications)

### Drift always detected

**Possible causes:**
1. Insufficient reference data (< 100 samples)
2. High natural variability in ratings
3. Thresholds too sensitive - adjust in [data_drift_monitor.py](data_drift_monitor.py:186)

### Prometheus metrics not updating

**Check:**
1. Exporter is running on port 8000
2. Database connection is active
3. Metrics are being written to database tables

## Performance Considerations

- **Database Load:** Monitoring queries are optimized with indexes
- **Memory:** Drift visualization can use ~200MB for plot generation
- **CPU:** Minimal (<5%) when running on schedule
- **Storage:** Reports and visualizations stored in `/app/reports/` (~10MB per week)

## Extending the Monitoring System

### Adding New Metrics

1. Define metric in appropriate monitor class
2. Add database column if persisting
3. Update Prometheus exporter with new gauge/counter
4. Add visualization to Grafana dashboard

### Adding New Alert Types

1. Add alert check in `check_alerts()` or `check_quality_alerts()`
2. Define thresholds and severity levels
3. Alert manager will automatically handle notification

### Integrating with Other Tools

- **PagerDuty:** Add webhook call in `AlertManager.send_alert()`
- **CloudWatch:** Export metrics via Prometheus remote write
- **DataDog:** Use StatsD integration with Prometheus

## References

- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- [Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)

## Support

For issues or questions about the monitoring infrastructure, contact the team or create an issue in the project repository.
