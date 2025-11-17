"""
Prometheus Metrics Exporter
Exposes monitoring metrics in Prometheus format for Grafana visualization
"""
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import time
import logging
import os
from db_utils import get_db_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Exports monitoring metrics to Prometheus.
    Runs HTTP server on port 8000 for Prometheus to scrape.
    """

    def __init__(self, db_path: str = '/app/db/home_alone.db', port: int = 8000):
        self.db_path = db_path
        self.port = port
        self.conn = None

        # Define Prometheus metrics
        logger.info("Initializing Prometheus metrics...")

        # Service Health Metrics
        self.service_availability = Gauge(
            'recommendation_service_availability_percentage',
            'Service availability percentage',
            ['window']
        )

        self.service_requests_total = Counter(
            'recommendation_requests_total',
            'Total number of recommendation requests',
            ['status']
        )

        self.service_response_time = Histogram(
            'recommendation_response_time_ms',
            'Response time in milliseconds',
            buckets=[10, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
        )

        self.service_error_rate = Gauge(
            'recommendation_service_error_rate',
            'Service error rate percentage',
            ['window']
        )

        # Model Quality Metrics - Single online metric: hit_rate
        self.model_hit_rate = Gauge(
            'recommendation_model_hit_rate',
            'Model hit rate (proportion of successful recommendations)',
            ['model_version', 'window']
        )

        # Data Drift Metrics - Single drift metric: Wasserstein distance
        self.data_drift_wasserstein = Gauge(
            'recommendation_data_drift_wasserstein_normalized',
            'Normalized Wasserstein distance for drift detection',
            ['window_type']
        )

        # Alert Metrics
        self.alerts_total = Counter(
            'recommendation_alerts_total',
            'Total number of alerts triggered',
            ['alert_type', 'severity', 'source']
        )

        self.active_alerts = Gauge(
            'recommendation_active_alerts',
            'Number of active unresolved alerts',
            ['severity']
        )

        # System Info
        self.system_info = Info(
            'recommendation_system',
            'Information about the recommendation system'
        )

        self.system_info.info({
            'version': '1.0',
            'environment': os.getenv('ENVIRONMENT', 'production')
        })

        logger.info("Prometheus metrics initialized")

    def connect_db(self):
        """Connect to database with proper configuration"""
        if self.conn is None:
            self.conn = get_db_connection(self.db_path, timeout=60.0)

    def update_health_metrics(self):
        """Update service health metrics from database"""
        try:
            self.connect_db()
            cursor = self.conn.cursor()

            # Get latest health metrics
            cursor.execute('''
                SELECT
                    window_minutes,
                    availability_percentage,
                    error_rate,
                    avg_response_time_ms,
                    successful_requests,
                    failed_requests
                FROM service_health_metrics
                ORDER BY timestamp DESC
                LIMIT 1
            ''')

            row = cursor.fetchone()
            if row:
                window, availability, error_rate, avg_response, success, failed = row

                self.service_availability.labels(window=f"{window}m").set(availability or 0)
                self.service_error_rate.labels(window=f"{window}m").set(error_rate or 0)

                # Update response time histogram (approximation)
                if avg_response:
                    self.service_response_time.observe(avg_response)

                logger.debug(f"Updated health metrics: availability={availability}%")

        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")

    def update_quality_metrics(self):
        """Update model quality metrics from database"""
        try:
            self.connect_db()
            cursor = self.conn.cursor()

            # Get latest quality metrics per model version
            cursor.execute('''
                SELECT
                    model_version,
                    window_hours,
                    hit_rate
                FROM model_quality_metrics
                WHERE timestamp >= datetime('now', '-2 hours')
                ORDER BY timestamp DESC
            ''')

            rows = cursor.fetchall()
            for row in rows:
                model_ver, window, hit_rate = row

                model_label = model_ver or 'default'

                self.model_hit_rate.labels(
                    model_version=model_label,
                    window=f"{window}h"
                ).set(hit_rate or 0)

                logger.debug(f"Updated quality metrics for {model_label}: hit_rate={hit_rate}")

        except Exception as e:
            logger.error(f"Error updating quality metrics: {e}")

    def update_drift_metrics(self):
        """Update data drift metrics from database"""
        try:
            self.connect_db()
            cursor = self.conn.cursor()

            # Get latest drift metrics - only Wasserstein distance
            cursor.execute('''
                SELECT
                    window_type,
                    normalized_wd
                FROM drift_metrics
                WHERE timestamp >= datetime('now', '-2 hours')
                  AND metric_name = 'rating_distribution'
                ORDER BY timestamp DESC
                LIMIT 10
            ''')

            rows = cursor.fetchall()

            # Track which window types we've seen
            seen_windows = set()

            for row in rows:
                window_type, norm_wd = row
                seen_windows.add(window_type)

                self.data_drift_wasserstein.labels(
                    window_type=window_type
                ).set(norm_wd or 0)

                logger.debug(f"Updated drift metric: {window_type} WD={norm_wd}")

            # If no data found, set default values for common window types
            # This ensures the dashboard shows 0 instead of "No data"
            if not rows:
                for window_type in ['hourly', 'daily']:
                    self.data_drift_wasserstein.labels(
                        window_type=window_type
                    ).set(0.0)
                logger.debug("No drift data found, setting default values to 0")

        except Exception as e:
            logger.error(f"Error updating drift metrics: {e}")
            # On error, still set default values to avoid "No data" in dashboard
            for window_type in ['hourly', 'daily']:
                self.data_drift_wasserstein.labels(
                    window_type=window_type
                ).set(0.0)

    def update_alert_metrics(self):
        """Update alert metrics from database"""
        try:
            self.connect_db()
            cursor = self.conn.cursor()

            # Count unresolved alerts by severity
            cursor.execute('''
                SELECT severity, COUNT(*)
                FROM (
                    SELECT severity, resolved FROM health_alerts
                    WHERE timestamp >= datetime('now', '-24 hours')
                    UNION ALL
                    SELECT severity, resolved FROM quality_alerts
                    WHERE timestamp >= datetime('now', '-24 hours')
                )
                WHERE resolved = 0
                GROUP BY severity
            ''')

            for severity, count in cursor.fetchall():
                self.active_alerts.labels(severity=severity).set(count)

            logger.debug("Updated alert metrics")

        except Exception as e:
            logger.error(f"Error updating alert metrics: {e}")

    def update_all_metrics(self):
        """Update all metrics"""
        logger.info("Updating all Prometheus metrics...")

        self.update_health_metrics()
        self.update_quality_metrics()
        self.update_drift_metrics()
        self.update_alert_metrics()

        logger.info("All Prometheus metrics updated")

    def start(self, update_interval: int = 30):
        """
        Start Prometheus exporter HTTP server and update loop.
        """
        logger.info(f"Starting Prometheus exporter on port {self.port}...")

        # Start HTTP server for Prometheus to scrape
        start_http_server(self.port)
        logger.info(f"Prometheus metrics available at http://localhost:{self.port}/metrics")

        # Continuously update metrics
        while True:
            try:
                self.update_all_metrics()
                time.sleep(update_interval)
            except KeyboardInterrupt:
                logger.info("Prometheus exporter stopped")
                break
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}", exc_info=True)
                time.sleep(update_interval)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    logger.info("Starting Prometheus Exporter for Recommendation Monitoring")

    exporter = PrometheusExporter()

    try:
        exporter.start(update_interval=30)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        exporter.close()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exporter.close()
