# Monitoring Infrastructure - Complete Guide

Comprehensive monitoring system for the Movie Recommendation Service that tracks service health, model quality, and data drift with automated alerting to `pgithiri@andrew.cmu.edu`.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Configuration](#configuration)
- [Testing End-to-End](#testing-end-to-end)
- [Grafana Dashboard](#grafana-dashboard)
- [Database Schema](#database-schema)
- [Troubleshooting](#troubleshooting)

---

## Overview

The monitoring infrastructure consists of three main components that work together to provide comprehensive observability:

1. **Service Health Monitor** - Tracks availability, response times, and error rates from the `recommendations` table (populated by inference telemetry)
2. **Model Quality Monitor** - Tracks prediction quality using online metrics by matching recommendations with actual watch behavior
3. **Data Drift Monitor** - Detects distribution changes in incoming data using statistical tests

All components are orchestrated by a **Unified Monitor** that provides centralized alerting via email, Slack, and file logs.

### Key Features

✅ **Single Source of Truth:** All monitoring reads from the `recommendations` table (no data duplication)
✅ **Real-time Metrics:** Prometheus exporter on port 8000
✅ **Automated Alerts:** Email to `pgithiri@andrew.cmu.edu`, Slack webhooks, file logging
✅ **A/B Testing Support:** Track multiple model versions simultaneously
✅ **Grafana Integration:** Pre-built dashboards for visualization
✅ **Containerized:** Fully Dockerized with docker-compose

---

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
                                   ┌────────────────────┴────────┐
                                   │  SQLite Database            │
                                   │  /app/db/home_alone.db      │
                                   │                             │
                                   │  ┌───────────────────────┐  │
                                   │  │ recommendations       │  │
                                   │  │ (telemetry from       │  │
                                   │  │  inference server)    │  │
                                   │  └───────────────────────┘  │
                                   │                             │
                                   │  Monitoring reads from      │
                                   │  this single table          │
                                   └─────────────────────────────┘
                                                ↓
                                   Prometheus Exporter (port 8000)
                                                ↓
                                   Grafana Dashboards (port 3000)
```

### Data Flow

```
Inference Request → Server (measures time/status) → Telemetry
                                                         ↓
                                         recommendations table
                                         (user_id, items, status,
                                          response_time_ms, server)
                                                         ↓
                                         ServiceHealthMonitor
                                         ModelQualityMonitor
                                         DataDriftMonitor
                                                         ↓
                                         Prometheus Metrics
                                                         ↓
                                         Grafana + Alerts
```

---

## Quick Start

### Prerequisites

1. **Docker** and **Docker Compose** installed
2. **Database** with watch history and ratings data
3. **Trained models** in `/app/models/` directory
4. **Email credentials** (Gmail or CMU SMTP)

### Step 1: Configure Email Alerts

Edit `.env` file in project root:

```bash
# Email Alert Configuration
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_TO=pgithiri@andrew.cmu.edu
ALERT_EMAIL_FROM=homealone-monitoring@cmu.edu

# Choose one SMTP option:

# Option A: Gmail (Recommended)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password

# Option B: CMU SMTP
# SMTP_HOST=smtp.andrew.cmu.edu
# SMTP_PORT=587
# SMTP_USERNAME=pgithiri
# SMTP_PASSWORD=your-andrew-password

# Test Mode (run checks every 5 minutes)
TEST_MODE=false
```

**Gmail App Password Setup:**
1. Go to https://myaccount.google.com/security
2. Enable 2-Factor Authentication
3. Search for "App Passwords"
4. Generate password for "Mail"
5. Use that 16-character password in `SMTP_PASSWORD`

### Step 2: Build and Start Services

```bash
# From project root
docker-compose build
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected services:
- `db_writer` - Database writer (Kafka consumer)
- `inference` - Inference server (port 8082)
- `monitoring` - Monitoring service (port 8000)
- `grafana` - Grafana dashboards (port 3000)

### Step 3: Generate Test Data

```powershell
# PowerShell - Generate 50 recommendation requests
for ($i=1; $i -le 50; $i++) {
    $userId = Get-Random -Minimum 1 -Maximum 1000
    Invoke-WebRequest -Uri "http://localhost:8082/recommend/$userId"
    Write-Host "Request $i completed"
}

# Generate some errors (for testing error metrics)
for ($i=1; $i -le 10; $i++) {
    try { Invoke-WebRequest -Uri "http://localhost:8082/recommend/invalid" } catch {}
}
```

### Step 4: Verify Metrics

```bash
# Check Prometheus metrics
curl http://localhost:8000/metrics

# Open Grafana
# Browser: http://localhost:3000
# Login: admin / admin
```

---

## Components

### 1. Service Health Monitor

**File:** [service_health.py](service_health.py)

**Purpose:** Monitor availability and performance of the recommendation service.

**Data Source:** Reads directly from `recommendations` table (populated by inference server telemetry)

**Metrics Tracked:**
- **Availability:** `(successful_requests / total_requests) * 100`
  - Successful = HTTP 200 status
  - Failed = status != 200
- **Response Time:** Average, P95, P99 percentiles (from `response_time_ms` column)
- **Error Rate:** `(failed_requests / total_requests) * 100`
- **Request Volume:** Total requests per time window
- **Model Version Performance:** Separate metrics per model version (A/B testing)

**Key Features:**
- ✅ Uses existing `recommendations` table (no Kafka parsing, no duplicates)
- ✅ Tracks per-model-version metrics
- ✅ Configurable time windows (1h, 24h, etc.)
- ✅ Automatic alert generation

**Alerts:**
- **Critical:** Availability < 70% OR Error Rate > 10%
- **Warning:** Availability < 90% OR P95 response time > 500ms

**Database Tables Created:**
- `service_health_metrics` - Historical aggregated metrics
- `health_alerts` - Health-related alerts

**Database Tables Used (from inference):**
- `recommendations` - Source of all health data

**Example Query:**
```sql
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful,
    AVG(CASE WHEN status = 200 THEN response_time_ms END) as avg_time
FROM recommendations
WHERE timestamp >= datetime('now', '-60 minutes');
```

---

### 2. Model Quality Monitor

**File:** [model_quality_monitor.py](model_quality_monitor.py)

**Purpose:** Track prediction quality in production using actual user behavior.

**Data Source:** Joins `recommendations` table with `watch_history` table

**Metrics Tracked:**
- **Hit Rate:** `(users_who_watched_>=1_rec / total_users) * 100`
  - % of users who watched at least one recommended movie
- **Precision@K:** `(avg of watched_items_in_recs / k)`
  - Average proportion of recommendations that were watched
- **Engagement Rate:** `(total_watched_from_recs / total_recommended) * 100`
- **NDCG@K:** Normalized Discounted Cumulative Gain (ranking quality)
- **Average Rank of Watched:** Average position of watched movies in recommendation list
- **Per-Model Metrics:** Separate tracking for each model version

**Key Features:**
- Matches recommendations with actual watch behavior (2-hour buffer for delayed watching)
- Configurable time windows and K values (default K=20)
- Supports A/B testing with per-model-version quality tracking
- Automatic alert generation for quality degradation

**Alerts:**
- **Critical:** Hit rate < 10%
- **Warning:** Hit rate < 20% OR Engagement rate < 5%

**Database Tables Created:**
- `model_quality_metrics` - Historical quality metrics
- `quality_alerts` - Quality-related alerts

**Database Tables Used:**
- `recommendations` - What was recommended
- `watch_history` - What was actually watched

**Example Query:**
```sql
-- Check if user watched recommended movies
SELECT
    r.user_id,
    r.recommended_items,
    COUNT(DISTINCT w.movie_id) as watched_count
FROM recommendations r
LEFT JOIN watch_history w ON
    r.user_id = w.user_id AND
    INSTR(r.recommended_items, w.movie_id) > 0 AND
    w.start_time BETWEEN r.timestamp AND datetime(r.timestamp, '+2 hours')
WHERE r.timestamp >= datetime('now', '-24 hours')
GROUP BY r.user_id;
```

---

### 3. Data Drift Monitor

**File:** [data_drift_monitor.py](data_drift_monitor.py)

**Purpose:** Detect distribution changes in incoming data (ratings, user activity).

**Data Source:** Reads from `ratings` table

**Metrics Tracked:**
- **KS Test p-value:** Statistical test for distribution differences
  - p < 0.05 indicates significant drift
- **Wasserstein Distance:** Normalized measure of distribution shift
  - Measures "distance" between distributions
- **Drift Severity:** Categorized as none/low/medium/high
- **Multi-window Detection:** Hourly, daily, and weekly windows

**Statistical Methods:**

1. **Kolmogorov-Smirnov Test:**
   ```python
   from scipy.stats import ks_2samp
   ks_stat, p_value = ks_2samp(reference_data, current_data)
   ```

2. **Wasserstein Distance:**
   ```python
   from scipy.stats import wasserstein_distance
   wd = wasserstein_distance(reference_data, current_data)
   normalized_wd = wd / reference_std
   ```

**Reference Window:** 7-14 days ago (stable baseline)

**Drift Thresholds:**
- **High:** Normalized WD > 0.3 OR KS p-value < 0.01
- **Medium:** Normalized WD > 0.15 OR KS p-value < 0.05
- **Low:** Normalized WD > 0.1

**Key Features:**
- Automated visualization generation (time series plots, heatmaps)
- HTML summary reports
- Multi-window analysis for different time scales

**Database Tables Created:**
- `drift_metrics` - Historical drift metrics with severity

**Visualization Outputs:**
- `/app/reports/drift/hourly/` - 1-hour window plots
- `/app/reports/drift/daily/` - 24-hour window plots
- `/app/reports/drift/weekly/` - 7-day window plots

---

### 4. Unified Monitor

**File:** [unified_monitor.py](unified_monitor.py)

**Purpose:** Orchestrate all monitoring components with centralized alerting.

**Features:**
- **Coordinated Scheduling:** Runs health/quality/drift checks at optimal intervals
- **Centralized Alerting:** Aggregates alerts from all components
- **Alert Notifications:** Console, file, email (SMTP), Slack webhooks
- **Monitoring Dashboard:** JSON export of all current metrics
- **Configuration Management:** Reads from environment variables via [config.py](config.py)

**Monitoring Schedule:**

**Production Mode (default):**
- Health Check: Every 5 minutes
- Quality Check: Every hour (at :10)
- Drift Check: Every hour (at :15)
- Dashboard Update: Every 15 minutes

**Test Mode (`TEST_MODE=true`):**
- All checks: Every 5 minutes (for faster testing)

**Alert Manager Features:**

1. **Console Logging:**
   ```
   🚨 ALERT [CRITICAL] - service_health: Service availability dropped to 65.50%
   ```

2. **File Logging (JSONL):**
   ```bash
   /app/reports/alerts/alerts.jsonl
   ```
   Each alert as JSON line:
   ```json
   {"timestamp":"2025-11-16T10:30:00","source":"service_health","alert_type":"low_availability","severity":"critical","message":"Service availability dropped to 65.50%","metric_value":65.5,"threshold_value":70.0}
   ```

3. **Email Alerts (SMTP):**
   Sends to `pgithiri@andrew.cmu.edu` for critical alerts:
   ```
   Subject: [CRITICAL] Movie Recommendation System Alert

   Alert Type: low_availability
   Source: service_health
   Severity: critical
   Time: 2025-11-16 10:30:00

   Message: Service availability dropped to 65.50%

   Metric Value: 65.5
   Threshold: 70.0
   ```

4. **Slack Webhooks (Optional):**
   Color-coded messages with metric details

---

### 5. Prometheus Exporter

**File:** [prometheus_exporter.py](prometheus_exporter.py)

**Purpose:** Export metrics in Prometheus format for Grafana visualization.

**Port:** 8000 (exposed for Prometheus scraping)

**Update Frequency:** Every 30 seconds (configurable)

**Exposed Metrics:**

```prometheus
# Service Health
recommendation_service_availability_percentage{window="60m"} 95.5
recommendation_requests_total{status="200"} 150
recommendation_requests_total{status="400"} 5
recommendation_response_time_ms_bucket{le="100"} 120
recommendation_response_time_ms_bucket{le="500"} 150
recommendation_service_error_rate{window="60m"} 3.2

# Model Quality
recommendation_model_hit_rate{model_version="svd",window="24h"} 0.25
recommendation_model_precision_at_k{model_version="svd",k="20",window="24h"} 0.15
recommendation_model_engagement_rate{model_version="svd",window="24h"} 0.08

# Data Drift
recommendation_data_drift_ks_pvalue{window_type="hourly",metric_name="rating_ks"} 0.45
recommendation_data_drift_wasserstein_normalized{window_type="daily",metric_name="rating_wd"} 0.12
recommendation_data_drift_detected{window_type="hourly",severity="high"} 0

# Alerts
recommendation_alerts_total{alert_type="low_availability",severity="critical",source="service_health"} 2
recommendation_active_alerts{severity="critical"} 1
```

**Access Metrics:**
```bash
curl http://localhost:8000/metrics
```

---

## Configuration

### Environment Variables (`.env` file)

```bash
# Database
DB_PATH=/app/db/home_alone.db

# Reports Directory
REPORTS_DIR=/app/reports

# Prometheus Port
PROMETHEUS_PORT=8000

# Test Mode (faster checks for testing)
TEST_MODE=false

# Email Alerts
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_TO=pgithiri@andrew.cmu.edu
ALERT_EMAIL_FROM=homealone-monitoring@cmu.edu

# SMTP Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Slack (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Alert Thresholds (Optional - defaults shown)
ALERT_AVAILABILITY_CRITICAL=70
ALERT_AVAILABILITY_WARNING=90
ALERT_ERROR_RATE_CRITICAL=10
ALERT_RESPONSE_TIME_P95=500
ALERT_HIT_RATE_CRITICAL=10
ALERT_HIT_RATE_WARNING=20
ALERT_ENGAGEMENT_RATE_WARNING=5
ALERT_DRIFT_WD_HIGH=0.3
```

### Configuration Class

**File:** [config.py](config.py)

Manages all configuration from environment variables with validation and printing:

```python
from config import MonitoringConfig

# Print current configuration
MonitoringConfig.print_config()

# Validate configuration
if MonitoringConfig.validate():
    print("Configuration is valid")

# Access values
db_path = MonitoringConfig.DB_PATH
alert_config = MonitoringConfig.ALERT_CONFIG
```

---

## Testing End-to-End

### Automated Test (PowerShell)

Use the included test script:

```powershell
# From project root
.\test_monitoring.ps1
```

This script automatically:
1. ✅ Checks Docker is running
2. ✅ Builds Docker images
3. ✅ Starts all services
4. ✅ Waits for inference server
5. ✅ Generates 20 valid requests
6. ✅ Generates 5 error requests
7. ✅ Verifies Prometheus metrics
8. ✅ Checks database records

### Manual Testing Steps

#### 1. Start Services

```bash
docker-compose up -d
docker-compose ps  # Verify all services running
```

#### 2. Check Service Logs

```bash
# Inference server
docker-compose logs -f inference

# Monitoring
docker-compose logs -f monitoring

# All services
docker-compose logs -f
```

Look for:
- ✅ "All resources loaded successfully" (inference)
- ✅ "Service health monitoring tables initialized" (monitoring)
- ✅ "Prometheus exporter started on port 8000" (monitoring)

#### 3. Generate Test Requests

```powershell
# Valid requests
for ($i=1; $i -le 50; $i++) {
    $userId = Get-Random -Minimum 1 -Maximum 1000
    Invoke-WebRequest -Uri "http://localhost:8082/recommend/$userId"
}

# Error requests (test error handling)
for ($i=1; $i -le 10; $i++) {
    try {
        Invoke-WebRequest -Uri "http://localhost:8082/recommend/invalid"
    } catch {}
}
```

#### 4. Verify Database Records

```bash
# Access database
docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db

# Check recommendations table
sqlite> SELECT COUNT(*) FROM recommendations;
sqlite> SELECT status, COUNT(*) FROM recommendations GROUP BY status;
sqlite> SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT 5;
sqlite> .schema recommendations

# Verify new columns exist
sqlite> PRAGMA table_info(recommendations);
# Should show: status, response_time_ms, server

sqlite> .quit
```

Expected output:
- Records with `status = 200` (successful)
- Records with `status = 400` (errors)
- `response_time_ms` populated
- `server = 'inference-server'`

#### 5. Check Prometheus Metrics

```bash
# Browser
http://localhost:8000/metrics

# OR curl
curl http://localhost:8000/metrics | grep recommendation
```

Look for:
- `recommendation_service_availability_percentage`
- `recommendation_requests_total{status="200"}`
- `recommendation_response_time_ms_bucket`

#### 6. Trigger Test Alert

```bash
# Get monitoring container
docker ps | grep monitoring

# Create test critical alert
docker exec -it <container-id> python3 << 'EOF'
import sqlite3
conn = sqlite3.connect('/app/db/home_alone.db')
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO health_alerts (alert_type, severity, message, metric_value, threshold_value)
    VALUES ('low_availability', 'critical', 'TEST: Availability dropped to 65%', 65.0, 70.0)
""")
conn.commit()
print("Test alert created!")
EOF
```

Check email at `pgithiri@andrew.cmu.edu` for alert notification.

#### 7. View Alert Logs

```bash
docker exec $(docker ps -q -f name=monitoring) cat /app/reports/alerts/alerts.jsonl
```

#### 8. Access Monitoring Dashboard

```bash
docker exec $(docker ps -q -f name=monitoring) cat /app/reports/monitoring_dashboard.json
```

---

## Grafana Dashboard

### Access Grafana

1. Open browser: http://localhost:3000
2. Login: `admin` / `admin` (change on first login)

### Configure Prometheus Data Source

1. Go to **Configuration** (gear icon) → **Data Sources**
2. Click **Add data source**
3. Select **Prometheus**
4. URL: `http://monitoring:8000` (Docker internal network)
5. Click **Save & Test** → Should show "Data source is working"

### Import Pre-Built Dashboard

**Option A: Auto-Provisioned**
- Dashboard should be auto-loaded at startup
- Go to **Dashboards** → **Browse**
- Look for "Recommendation System Monitoring"

**Option B: Manual Import**
1. Go to **Dashboards** → **Import**
2. Upload: `homealone/grafana/provisioning/dashboards/recommendation_monitoring.json`
3. Select Prometheus data source
4. Click **Import**

### Dashboard Panels

**Row 1 - Service Health:**
- **Availability (1h)** - Gauge (red < 70%, yellow < 90%, green >= 90%)
- **Response Time (P95/P99)** - Line graph with alert line at 500ms
- **Error Rate** - Time series graph
- **Active Alerts** - Stat panel showing count

**Row 2 - Model Quality:**
- **Hit Rate by Model Version** - Compare model performance over time
- **Precision@K** - Quality of recommendations
- **Engagement Rate** - User interaction with recommendations

**Row 3 - Data Drift:**
- **Wasserstein Distance (Hourly/Daily/Weekly)** - Drift magnitude
- **Drift Detection Status** - Binary time series (0 = no drift, 1 = drift)
- **KS Test P-Values** - Statistical significance

**Row 4 - Alerts:**
- **Alert Rate by Type** - Breakdown of alert types
- **Recent Alerts** - Timeline of alerts

### Recommended PromQL Queries

```promql
# Service availability (gauge)
recommendation_service_availability_percentage{window="60m"}

# Response time P95 (graph)
histogram_quantile(0.95, rate(recommendation_response_time_ms_bucket[5m]))

# Compare models (graph)
recommendation_model_hit_rate{window="24h"} by (model_version)

# Data drift (heatmap)
recommendation_data_drift_wasserstein_normalized by (window_type)

# Active alerts (stat)
sum(recommendation_active_alerts{severity="critical"})

# Request rate (graph)
rate(recommendation_requests_total[5m]) by (status)
```

---

## A/B Testing Support

All monitoring components support tracking multiple model versions simultaneously.

### 1. Tag Model Versions in Inference Server

Currently, the inference server doesn't tag model versions. To enable A/B testing monitoring:

**Edit [server.py](../inference/server.py):**

```python
# Option 1: Random assignment
import random
model_version = random.choice(['svd', 'knn', 'nmf'])

# Option 2: User-based assignment
model_version = 'svd' if int(user_id) % 2 == 0 else 'knn'

# Option 3: Time-based
import datetime
hour = datetime.datetime.now().hour
model_version = 'v1' if hour < 12 else 'v2'

# Then include in telemetry
telemetry.log('recommendation', {
    'user_id': raw_user_id,
    'recommended_items': recommendation_response,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'status': status_code,
    'response_time_ms': response_time_ms,
    'server': 'inference-server',
    'model_version': model_version  # ADD THIS
})
```

### 2. View Per-Model Metrics

**Health Metrics:**
```python
from service_health import ServiceHealthMonitor
monitor = ServiceHealthMonitor()
version_metrics = monitor.compute_model_version_metrics(window_minutes=60)
```

**Quality Metrics:**
```python
from model_quality_monitor import ModelQualityMonitor
monitor = ModelQualityMonitor()
version_quality = monitor.compute_model_version_quality(window_hours=24, k=20)
```

**Grafana Comparison:**
```promql
recommendation_model_hit_rate{window="24h"} by (model_version)
```

### 3. Test A/B Monitoring (Manual Data)

```bash
docker exec -it $(docker ps -q -f name=monitoring) python3 << 'EOF'
import sqlite3
from datetime import datetime
conn = sqlite3.connect('/app/db/home_alone.db')
cursor = conn.cursor()

# Insert test data for 3 model versions
models = ['svd', 'knn', 'nmf']
for i in range(30):
    model = models[i % 3]
    cursor.execute("""
        INSERT INTO recommendations
        (user_id, recommended_items, timestamp, model_version, status, response_time_ms, server)
        VALUES (?, ?, datetime('now'), ?, 200, ?, 'inference-server')
    """, (str(100 + i), '1,2,3,4,5', model, 50 + (i % 3) * 10))

conn.commit()
print(f"Inserted 30 test recommendations across 3 model versions")
EOF
```

---

## Database Schema

### recommendations (Source Table - Created by Telemetry)

```sql
CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    recommended_items TEXT NOT NULL,  -- Comma-separated movie IDs
    model_version TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status INTEGER DEFAULT 200,  -- HTTP status code
    response_time_ms REAL,       -- Response time in milliseconds
    server TEXT                  -- Server identifier
);

CREATE INDEX idx_recommendations_timestamp ON recommendations(timestamp);
```

### service_health_metrics (Created by Health Monitor)

```sql
CREATE TABLE service_health_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    window_minutes INTEGER NOT NULL,
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    timeout_requests INTEGER DEFAULT 0,
    availability_percentage REAL,
    avg_response_time_ms REAL,
    p95_response_time_ms REAL,
    p99_response_time_ms REAL,
    error_rate REAL,
    model_version TEXT,
    details TEXT
);
```

### model_quality_metrics (Created by Quality Monitor)

```sql
CREATE TABLE model_quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    window_hours INTEGER NOT NULL,
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

### drift_metrics (Created by Drift Monitor)

```sql
CREATE TABLE drift_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    window_type TEXT NOT NULL,  -- 'hourly', 'daily', 'weekly'
    metric_name TEXT NOT NULL,  -- 'rating_distribution', 'user_activity'
    ks_statistic REAL,
    ks_p_value REAL,
    wasserstein_distance REAL,
    normalized_wd REAL,
    ref_mean REAL,
    ref_std REAL,
    curr_mean REAL,
    curr_std REAL,
    ref_sample_size INTEGER,
    curr_sample_size INTEGER,
    drift_detected BOOLEAN,
    drift_severity TEXT  -- 'none', 'low', 'medium', 'high'
);
```

### health_alerts / quality_alerts (Created by Monitors)

```sql
CREATE TABLE health_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,  -- 'warning', 'critical'
    message TEXT NOT NULL,
    metric_value REAL,
    threshold_value REAL,
    resolved BOOLEAN DEFAULT 0,
    resolved_at DATETIME
);
```

---

## Troubleshooting

### Issue: No Metrics in Grafana

**Symptoms:**
- Grafana dashboard shows "No data"
- Prometheus data source test fails

**Solutions:**

1. **Check Prometheus endpoint:**
   ```bash
   curl http://localhost:8000/metrics
   ```
   Should return Prometheus-formatted metrics

2. **Verify monitoring container is running:**
   ```bash
   docker-compose ps monitoring
   ```

3. **Check monitoring logs:**
   ```bash
   docker-compose logs monitoring
   ```
   Look for errors or "Prometheus exporter started"

4. **Restart monitoring service:**
   ```bash
   docker-compose restart monitoring
   ```

5. **Verify database has data:**
   ```bash
   docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db "SELECT COUNT(*) FROM recommendations;"
   ```

---

### Issue: Email Alerts Not Sending

**Symptoms:**
- Alerts appear in logs but no email received
- SMTP errors in monitoring logs

**Solutions:**

1. **Verify environment variables:**
   ```bash
   docker-compose config | grep -A 10 ALERT_EMAIL
   ```

2. **Check SMTP credentials:**
   - Gmail: Use app password, not regular password
   - CMU: Ensure Andrew credentials are correct

3. **Test SMTP connection manually:**
   ```python
   import smtplib
   server = smtplib.SMTP('smtp.gmail.com', 587)
   server.starttls()
   server.login('your-email@gmail.com', 'app-password')
   print("SMTP connection successful!")
   server.quit()
   ```

4. **Check monitoring logs for email errors:**
   ```bash
   docker-compose logs monitoring | grep -i email
   docker-compose logs monitoring | grep -i smtp
   ```

5. **Verify `ALERT_EMAIL_ENABLED=true` in `.env`**

6. **Check spam folder** in pgithiri@andrew.cmu.edu inbox

---

### Issue: No Data in recommendations Table

**Symptoms:**
- Monitoring shows "No requests in last X minutes"
- Database query returns 0 rows

**Solutions:**

1. **Make inference requests:**
   ```bash
   curl http://localhost:8082/recommend/123
   ```

2. **Check inference server is logging:**
   ```bash
   docker-compose logs inference | grep -i telemetry
   ```

3. **Verify database connection:**
   ```bash
   docker exec -it $(docker ps -q -f name=inference) ls -la /app/db/home_alone.db
   ```

4. **Check table schema:**
   ```bash
   docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db ".schema recommendations"
   ```
   Should show columns: status, response_time_ms, server

5. **Verify new columns exist:**
   ```bash
   docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db "PRAGMA table_info(recommendations);"
   ```

---

### Issue: Monitoring Shows "No requests in last X minutes"

**Symptoms:**
- Health metrics return None
- Grafana shows gaps in data

**Solutions:**

1. **Generate more requests:**
   ```powershell
   for ($i=1; $i -le 50; $i++) {
       Invoke-WebRequest -Uri "http://localhost:8082/recommend/$((Get-Random -Min 1 -Max 1000))"
   }
   ```

2. **Check time zone consistency:**
   - Inference server uses `time.gmtime()` (UTC)
   - Monitoring queries use `datetime('now')` (local time)
   - Solution: Ensure both use same time zone

3. **Verify timestamp format:**
   ```bash
   docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db "SELECT timestamp FROM recommendations LIMIT 5;"
   ```

4. **Check monitoring query window:**
   ```python
   # In service_health.py, try larger window
   metrics = monitor.compute_availability_metrics(window_minutes=120)
   ```

---

### Issue: Drift Always Detected

**Symptoms:**
- Every drift check shows "high" severity
- KS p-value always < 0.01

**Possible Causes & Solutions:**

1. **Insufficient reference data:**
   - Need at least 100 samples in reference window
   - Check: `SELECT COUNT(*) FROM ratings WHERE rated_at BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days');`

2. **High natural variability:**
   - Ratings data is inherently noisy
   - Solution: Adjust thresholds in [config.py](config.py)
   ```python
   DRIFT_THRESHOLDS = {
       'wasserstein_high': 0.5,  # Increase from 0.3
   }
   ```

3. **Data collection just started:**
   - Need historical data for comparison
   - Wait for 14+ days of data

4. **Actual drift occurring:**
   - This might be legitimate!
   - Investigate what changed in user behavior or ratings

---

### Issue: Prometheus Metrics Not Updating

**Symptoms:**
- Metrics endpoint shows stale data
- Grafana graphs are flat

**Solutions:**

1. **Check exporter is running:**
   ```bash
   docker-compose logs monitoring | grep -i prometheus
   ```
   Should see "Prometheus exporter started on port 8000"

2. **Verify database connection:**
   ```bash
   docker-compose logs monitoring | grep -i database
   ```

3. **Check update loop:**
   ```bash
   docker-compose logs monitoring | grep -i "Updating Prometheus metrics"
   ```
   Should happen every 30 seconds

4. **Restart monitoring:**
   ```bash
   docker-compose restart monitoring
   ```

5. **Verify metrics are being written to database:**
   ```bash
   docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db "SELECT * FROM service_health_metrics ORDER BY timestamp DESC LIMIT 1;"
   ```

---

## Performance Considerations

- **Database Load:** Optimized queries with indexes on timestamp columns
- **Memory:** ~200MB for drift visualization plot generation
- **CPU:** < 5% when running on scheduled intervals
- **Storage:** Reports and visualizations ~10MB per week in `/app/reports/`
- **Network:** Minimal - only Prometheus scraping on port 8000

## Key Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Inference API | http://localhost:8082 | - |
| Prometheus Metrics | http://localhost:8000/metrics | - |
| Grafana Dashboard | http://localhost:3000 | admin/admin |

## Key Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f monitoring
docker-compose logs -f inference

# Generate test requests
curl http://localhost:8082/recommend/123

# Check database
docker exec -it $(docker ps -q -f name=db_writer) sqlite3 /app/db/home_alone.db

# Check metrics
curl http://localhost:8000/metrics

# Stop services
docker-compose down
```

## Support

For issues or questions about the monitoring infrastructure:
- Check logs: `docker-compose logs monitoring`
- Check database: Connect to `/app/db/home_alone.db`
- Review alert logs: `/app/reports/alerts/alerts.jsonl`
- Contact: pgithiri@andrew.cmu.edu
