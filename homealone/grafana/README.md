# Grafana Setup for Recommendation System Monitoring

This directory contains Grafana configuration for visualizing monitoring metrics.

## Quick Start

### 1. Start Grafana

```bash
docker-compose up -d grafana
```

### 2. Access Grafana

Open your browser and navigate to:
```
http://localhost:3000
```

**Default Credentials:**
- Username: `admin`
- Password: `admin`

You'll be prompted to change the password on first login (optional).

### 3. View Dashboard

The dashboard is automatically provisioned and available at:
```
http://localhost:3000/d/recommendation-monitoring/movie-recommendation-system-monitoring
```

Or navigate through the UI:
- Click "Dashboards" (left sidebar)
- Select "Movie Recommendation System Monitoring"

## Dashboard Panels

The pre-configured dashboard includes:

### 1. **Service Availability (1h)** - Gauge
- Shows current 1-hour availability percentage
- Color-coded thresholds:
  - Red: < 70% (Critical)
  - Orange: 70-90% (Warning)
  - Yellow: 90-95%
  - Green: > 95% (Healthy)

### 2. **Response Time (P95, P99)** - Time Series
- Tracks 95th and 99th percentile response times
- Helps identify performance degradation
- Alert threshold: P95 > 500ms

### 3. **Active Alerts** - Stat
- Displays current number of active unresolved alerts
- Turns red when alerts are present

### 4. **Model Hit Rate by Version** - Time Series
- Compares hit rate across different model versions
- Essential for A/B testing
- Thresholds:
  - Critical: < 10%
  - Warning: < 20%

### 5. **Model Precision@K by Version** - Time Series
- Shows precision@K metrics per model version
- Tracks recommendation quality over time

### 6. **Data Drift (Wasserstein Distance)** - Time Series
- Monitors data distribution changes
- Separate lines for hourly/daily windows
- Thresholds:
  - Yellow: > 0.15 (Medium drift)
  - Red: > 0.3 (High drift)

### 7. **Drift Detection Status** - Time Series
- Binary indicator of drift detection
- Shows which time windows detected drift

### 8. **Alert Rate by Type** - Time Series (Stacked Bars)
- Displays alert frequency by severity and type
- Helps identify recurring issues

## Customization

### Adding New Panels

1. Click "Add" → "Visualization"
2. Select "Prometheus" as data source
3. Enter PromQL query (see available metrics below)
4. Configure visualization type and thresholds
5. Save dashboard

### Available Metrics

All metrics are prefixed with `recommendation_`:

**Service Health:**
```promql
recommendation_service_availability_percentage{window="60m"}
recommendation_requests_total{status="200"}
recommendation_response_time_ms (histogram)
recommendation_service_error_rate{window="60m"}
```

**Model Quality:**
```promql
recommendation_model_hit_rate{model_version="v1", window="24h"}
recommendation_model_precision_at_k{model_version="v1", k="20"}
recommendation_model_engagement_rate{model_version="v1"}
```

**Data Drift:**
```promql
recommendation_data_drift_ks_pvalue{window_type="hourly"}
recommendation_data_drift_wasserstein_normalized{window_type="daily"}
recommendation_data_drift_detected{window_type="hourly", severity="high"}
```

**Alerts:**
```promql
recommendation_alerts_total{alert_type="low_availability", severity="critical"}
recommendation_active_alerts{severity="critical"}
```

### Example Queries

**Service availability over time (5-minute average):**
```promql
avg_over_time(recommendation_service_availability_percentage{window="60m"}[5m])
```

**Request rate per second:**
```promql
rate(recommendation_requests_total[5m])
```

**Error rate percentage:**
```promql
(rate(recommendation_requests_total{status!="200"}[5m]) / rate(recommendation_requests_total[5m])) * 100
```

**Compare model versions:**
```promql
recommendation_model_hit_rate{window="24h"} by (model_version)
```

## Configuration Files

### `provisioning/datasources/prometheus.yml`
- Configures Prometheus data source
- Points to monitoring container on port 8000
- Auto-provisioned on Grafana startup

### `provisioning/dashboards/dashboard.yml`
- Dashboard provider configuration
- Makes dashboards auto-load from files

### `provisioning/dashboards/recommendation_monitoring.json`
- Pre-built dashboard definition
- Includes all monitoring panels
- Editable through Grafana UI

## Alerting (Optional)

### Configure Alert Notifications

1. Go to "Alerting" → "Contact points"
2. Add notification channel:
   - Email (SMTP)
   - Slack
   - PagerDuty
   - Webhook

### Create Alert Rules

Example: Alert on low availability

1. Edit "Service Availability" panel
2. Click "Alert" tab
3. Create alert rule:
   ```
   WHEN avg() OF query(A, 5m, now) IS BELOW 90
   ```
4. Set notification channel

## Troubleshooting

### Grafana not starting

```bash
# Check logs
docker-compose logs grafana

# Recreate container
docker-compose up -d --force-recreate grafana
```

### Data source not working

```bash
# Verify monitoring container is running
docker-compose ps monitoring

# Test Prometheus endpoint
curl http://localhost:8000/metrics

# Check Grafana can reach monitoring
docker exec -it group-project-f25-home-alone-grafana-1 sh
wget -O- http://monitoring:8000/metrics
```

### No data in panels

- **Cause**: No recommendation requests have been made yet
- **Solution**: Make some requests to the inference service:
  ```bash
  curl http://localhost:8082/recommend/123
  ```
- Wait a few minutes for metrics to populate

### Dashboard not appearing

```bash
# Check provisioning logs
docker-compose logs grafana | grep -i dashboard

# Manually import dashboard
# Go to Dashboards → Import → Upload JSON file
# Use: homealone/grafana/provisioning/dashboards/recommendation_monitoring.json
```

## Backup and Migration

### Export Dashboard

1. Go to dashboard settings (gear icon)
2. Click "JSON Model"
3. Copy JSON
4. Save to file

### Export Data Source Configuration

```bash
# Via Grafana UI
Settings → Data Sources → Prometheus → Export

# Or copy from container
docker cp group-project-f25-home-alone-grafana-1:/var/lib/grafana ./grafana-backup
```

## Advanced Configuration

### Change Refresh Rate

Edit dashboard settings:
- Top-right: Click refresh dropdown
- Select interval (10s, 30s, 1m, 5m, etc.)
- Or set in JSON: `"refresh": "30s"`

### Add Variables

Create template variables for dynamic filtering:

1. Dashboard settings → Variables → Add variable
2. Example: `model_version` variable
   - Type: Query
   - Data source: Prometheus
   - Query: `label_values(recommendation_model_hit_rate, model_version)`
3. Use in queries: `{model_version="$model_version"}`

### Set Up Snapshots

Share dashboard state:
1. Click "Share" icon
2. "Snapshot" tab
3. Publish to raintank.io or export locally

## References

- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)

---

**Access URLs:**
- Grafana UI: http://localhost:3000
- Prometheus Metrics: http://localhost:8000/metrics
- Dashboard Direct Link: http://localhost:3000/d/recommendation-monitoring
