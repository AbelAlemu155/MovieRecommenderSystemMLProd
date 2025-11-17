"""
Service Health Monitor - Tracks availability and health of the recommendation service
Monitors recommendation data from the database populated by inference telemetry
"""
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from db_utils import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceHealthMonitor:
    """
    Monitors recommendation service health by analyzing Kafka logs.
    Tracks availability, response times, error rates, and model version performance.
    """

    def __init__(self,
                 db_path: str = '/app/db/home_alone.db',
                 reports_dir: str = '/app/reports'):
        self.db_path = db_path
        self.reports_dir = reports_dir
        self.conn = None

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'health'), exist_ok=True)

        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection with proper configuration"""
        self.conn = get_db_connection(self.db_path, timeout=60.0)
        logger.info(f"Connected to database: {self.db_path}")

    def _create_tables(self):
        """Create tables for health metrics (recommendations table is created by telemetry)"""
        cursor = self.conn.cursor()

        # Service availability metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS service_health_metrics (
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
            )
        ''')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_value REAL,
                threshold_value REAL,
                resolved BOOLEAN DEFAULT 0,
                resolved_at DATETIME
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp
            ON health_alerts(timestamp, resolved)
        ''')

        self.conn.commit()
        logger.info("Service health monitoring tables initialized (using existing recommendations table)")


    def compute_availability_metrics(self, window_minutes: int = 60) -> Dict:
        """
        Compute service availability metrics for the specified time window.
        Uses the recommendations table populated by inference telemetry.
        """
        query = f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status != 200 THEN 1 ELSE 0 END) as failed_requests,
                AVG(CASE WHEN status = 200 THEN response_time_ms END) as avg_response_time,
                model_version
            FROM recommendations
            WHERE timestamp >= datetime('now', '-{window_minutes} minutes')
            GROUP BY model_version
        """

        df = pd.read_sql_query(query, self.conn)

        if len(df) == 0:
            logger.warning(f"No requests in last {window_minutes} minutes")
            return None

        # Aggregate across all model versions
        total_requests = df['total_requests'].sum()
        successful_requests = df['successful_requests'].sum()
        failed_requests = df['failed_requests'].sum()

        availability = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0

        # Get response time percentiles
        response_times_query = f"""
            SELECT response_time_ms
            FROM recommendations
            WHERE timestamp >= datetime('now', '-{window_minutes} minutes')
              AND status = 200
            ORDER BY response_time_ms
        """

        response_times = pd.read_sql_query(response_times_query, self.conn)

        if len(response_times) > 0:
            avg_response_time = response_times['response_time_ms'].mean()
            p95_response_time = response_times['response_time_ms'].quantile(0.95)
            p99_response_time = response_times['response_time_ms'].quantile(0.99)
        else:
            avg_response_time = p95_response_time = p99_response_time = None

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': window_minutes,
            'total_requests': int(total_requests),
            'successful_requests': int(successful_requests),
            'failed_requests': int(failed_requests),
            'availability_percentage': round(availability, 2),
            'error_rate': round(error_rate, 2),
            'avg_response_time_ms': round(avg_response_time, 2) if avg_response_time else None,
            'p95_response_time_ms': round(p95_response_time, 2) if p95_response_time else None,
            'p99_response_time_ms': round(p99_response_time, 2) if p99_response_time else None,
            'model_versions': df.to_dict('records')
        }

        logger.info(
            f"Availability [{window_minutes}min]: {availability:.2f}%, "
            f"Requests: {total_requests}, "
            f"Errors: {failed_requests} ({error_rate:.2f}%)"
        )

        return metrics

    def compute_model_version_metrics(self, window_minutes: int = 60) -> List[Dict]:
        """
        Compute separate metrics for each model version (for A/B testing).
        Uses the recommendations table populated by inference telemetry.
        """
        query = f"""
            SELECT
                model_version,
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 200 THEN 1 ELSE 0 END) as successful_requests,
                AVG(CASE WHEN status = 200 THEN response_time_ms END) as avg_response_time,
                COUNT(DISTINCT user_id) as unique_users
            FROM recommendations
            WHERE timestamp >= datetime('now', '-{window_minutes} minutes')
              AND model_version IS NOT NULL
            GROUP BY model_version
        """

        df = pd.read_sql_query(query, self.conn)

        results = []
        for _, row in df.iterrows():
            availability = (row['successful_requests'] / row['total_requests'] * 100) if row['total_requests'] > 0 else 0

            results.append({
                'model_version': row['model_version'],
                'total_requests': int(row['total_requests']),
                'successful_requests': int(row['successful_requests']),
                'availability_percentage': round(availability, 2),
                'avg_response_time_ms': round(row['avg_response_time'], 2) if row['avg_response_time'] else None,
                'unique_users': int(row['unique_users'])
            })

        return results

    def save_health_metrics(self, metrics: Dict):
        """Save computed health metrics to database"""
        if metrics is None:
            return

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO service_health_metrics (
                window_minutes, total_requests, successful_requests, failed_requests,
                availability_percentage, avg_response_time_ms,
                p95_response_time_ms, p99_response_time_ms, error_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['window_minutes'],
            metrics['total_requests'],
            metrics['successful_requests'],
            metrics['failed_requests'],
            metrics['availability_percentage'],
            metrics['avg_response_time_ms'],
            metrics['p95_response_time_ms'],
            metrics['p99_response_time_ms'],
            metrics['error_rate']
        ))
        self.conn.commit()

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """
        Check if any alert thresholds are violated.
        """
        alerts = []

        # Alert: Low availability
        if metrics['availability_percentage'] < 70:
            alerts.append({
                'alert_type': 'low_availability',
                'severity': 'critical',
                'message': f"Service availability dropped to {metrics['availability_percentage']:.2f}%",
                'metric_value': metrics['availability_percentage'],
                'threshold_value': 70.0
            })
        elif metrics['availability_percentage'] < 90:
            alerts.append({
                'alert_type': 'low_availability',
                'severity': 'warning',
                'message': f"Service availability at {metrics['availability_percentage']:.2f}%",
                'metric_value': metrics['availability_percentage'],
                'threshold_value': 90.0
            })

        # Alert: High error rate
        if metrics['error_rate'] > 10:
            alerts.append({
                'alert_type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate is {metrics['error_rate']:.2f}%",
                'metric_value': metrics['error_rate'],
                'threshold_value': 10.0
            })

        # Alert: Slow response time
        if metrics['p95_response_time_ms'] and metrics['p95_response_time_ms'] > 500:
            alerts.append({
                'alert_type': 'slow_response',
                'severity': 'warning',
                'message': f"P95 response time is {metrics['p95_response_time_ms']:.0f}ms",
                'metric_value': metrics['p95_response_time_ms'],
                'threshold_value': 500.0
            })

        # Log and save alerts
        for alert in alerts:
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
            self._save_alert(alert)

        return alerts

    def _save_alert(self, alert: Dict):
        """Save alert to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO health_alerts
            (alert_type, severity, message, metric_value, threshold_value)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            alert['alert_type'],
            alert['severity'],
            alert['message'],
            alert['metric_value'],
            alert['threshold_value']
        ))
        self.conn.commit()

    def get_health_summary(self, hours: int = 24) -> Dict:
        """
        Get summary of service health over specified hours.
        """
        query = f"""
            SELECT
                AVG(availability_percentage) as avg_availability,
                MIN(availability_percentage) as min_availability,
                AVG(error_rate) as avg_error_rate,
                AVG(avg_response_time_ms) as avg_response_time,
                SUM(total_requests) as total_requests
            FROM service_health_metrics
            WHERE timestamp >= datetime('now', '-{hours} hours')
        """

        result = pd.read_sql_query(query, self.conn)

        if len(result) == 0 or result.iloc[0]['avg_availability'] is None:
            return {
                'hours': hours,
                'avg_availability': 0,
                'min_availability': 0,
                'avg_error_rate': 0,
                'avg_response_time': 0,
                'total_requests': 0
            }

        row = result.iloc[0]
        return {
            'hours': hours,
            'avg_availability': round(row['avg_availability'], 2),
            'min_availability': round(row['min_availability'], 2),
            'avg_error_rate': round(row['avg_error_rate'], 2),
            'avg_response_time': round(row['avg_response_time'], 2),
            'total_requests': int(row['total_requests'])
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    logger.info("Starting Service Health Monitor")

    monitor = ServiceHealthMonitor()

    try:
        while True:
            # Compute metrics every 5 minutes
            logger.info("Computing health metrics...")

            # 1-hour window metrics
            metrics_1h = monitor.compute_availability_metrics(window_minutes=60)
            if metrics_1h:
                monitor.save_health_metrics(metrics_1h)
                monitor.check_alerts(metrics_1h)

            # Model version metrics (for A/B testing)
            version_metrics = monitor.compute_model_version_metrics(window_minutes=60)
            if version_metrics:
                logger.info(f"Model version metrics: {version_metrics}")

            # Health summary
            summary_24h = monitor.get_health_summary(hours=24)
            logger.info(f"24h Health Summary: {summary_24h}")

            # Wait 5 minutes
            time.sleep(300)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        monitor.close()
    except Exception as e:
        logger.error(f"Error in health monitor: {e}", exc_info=True)
        monitor.close()
