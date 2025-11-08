"""
Unified Monitoring System
Orchestrates all monitoring components: health, quality, and drift
Provides centralized alerting and reporting
"""
import logging
import time
import schedule
import json
import os
from datetime import datetime
from threading import Thread
from typing import Dict, List
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from service_health import ServiceHealthMonitor
from model_quality_monitor import ModelQualityMonitor
from data_drift_monitor import DataDriftMonitor
from prometheus_exporter import PrometheusExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts from all monitoring components.
    Sends notifications when critical issues are detected.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alert_history = []
        self.reports_dir = '/app/reports/alerts'
        os.makedirs(self.reports_dir, exist_ok=True)

    def send_alert(self, alert: Dict):
        """
        Send alert notification.
        Currently logs to file and console. Can be extended to send emails/Slack/PagerDuty.
        """
        logger.warning(f"🚨 ALERT [{alert['severity'].upper()}] - {alert['source']}: {alert['message']}")

        # Save to alert log file
        alert_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            **alert
        }

        alert_file = os.path.join(self.reports_dir, 'alerts.jsonl')
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert_with_timestamp) + '\n')

        self.alert_history.append(alert_with_timestamp)

        # Send critical alerts via email (if configured)
        if alert['severity'] == 'critical' and self.config.get('email_enabled'):
            self._send_email_alert(alert)

        # Send to Slack (if configured)
        if self.config.get('slack_webhook_url'):
            self._send_slack_alert(alert)

    def _send_email_alert(self, alert: Dict):
        """Send email alert (requires SMTP configuration)"""
        try:
            smtp_config = self.config.get('smtp', {})
            if not smtp_config:
                return

            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('to_email')
            msg['Subject'] = f"[{alert['severity'].upper()}] Movie Recommendation System Alert"

            body = f"""
Alert Type: {alert['alert_type']}
Source: {alert['source']}
Severity: {alert['severity']}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert['message']}

Metric Value: {alert.get('metric_value', 'N/A')}
Threshold: {alert.get('threshold_value', 'N/A')}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                server.starttls()
                server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)

            logger.info(f"Email alert sent for {alert['alert_type']}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack_alert(self, alert: Dict):
        """Send Slack alert via webhook"""
        try:
            import requests

            webhook_url = self.config.get('slack_webhook_url')
            if not webhook_url:
                return

            emoji = "🚨" if alert['severity'] == 'critical' else "⚠️"
            color = "danger" if alert['severity'] == 'critical' else "warning"

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"{emoji} {alert['source']} Alert",
                    "text": alert['message'],
                    "fields": [
                        {"title": "Alert Type", "value": alert['alert_type'], "short": True},
                        {"title": "Severity", "value": alert['severity'], "short": True},
                        {"title": "Metric Value", "value": str(alert.get('metric_value', 'N/A')), "short": True},
                        {"title": "Threshold", "value": str(alert.get('threshold_value', 'N/A')), "short": True}
                    ],
                    "footer": "Movie Recommendation Monitoring",
                    "ts": int(time.time())
                }]
            }

            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info(f"Slack alert sent for {alert['alert_type']}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from last N hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff_time
        ]

        return recent_alerts


class UnifiedMonitor:
    """
    Unified monitoring system that orchestrates all monitoring components.
    """

    def __init__(self,
                 db_path: str = '/app/db/home_alone.db',
                 reports_dir: str = '/app/reports',
                 alert_config: Dict = None):

        self.db_path = db_path
        self.reports_dir = reports_dir

        # Initialize monitoring components
        logger.info("Initializing monitoring components...")

        self.health_monitor = ServiceHealthMonitor(db_path, reports_dir)
        self.quality_monitor = ModelQualityMonitor(db_path, reports_dir)
        self.drift_monitor = DataDriftMonitor(db_path, reports_dir)

        self.alert_manager = AlertManager(alert_config or {})

        # Initialize Prometheus exporter
        self.prometheus_exporter = PrometheusExporter(db_path=db_path, port=8000)

        # Start Prometheus exporter in background thread
        self.exporter_thread = Thread(target=self.prometheus_exporter.start, args=(30,), daemon=True)
        self.exporter_thread.start()
        logger.info("Prometheus exporter started on port 8000")

        logger.info("All monitoring components initialized")

    def run_health_check(self):
        """Run service health monitoring"""
        try:
            logger.info("Running service health check...")

            # Compute health metrics
            metrics_1h = self.health_monitor.compute_availability_metrics(window_minutes=60)

            if metrics_1h:
                # Save metrics
                self.health_monitor.save_health_metrics(metrics_1h)

                # Check for alerts
                alerts = self.health_monitor.check_alerts(metrics_1h)

                for alert in alerts:
                    self.alert_manager.send_alert({
                        'source': 'service_health',
                        **alert
                    })

                # Log model version metrics
                version_metrics = self.health_monitor.compute_model_version_metrics(window_minutes=60)
                if version_metrics:
                    logger.info(f"Model version health: {json.dumps(version_metrics, indent=2)}")

        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)

    def run_quality_check(self):
        """Run model quality monitoring"""
        try:
            logger.info("Running model quality check...")

            # Compute quality metrics
            metrics_24h = self.quality_monitor.compute_online_accuracy(window_hours=24, k=20)

            if metrics_24h:
                # Save metrics
                self.quality_monitor.save_quality_metrics(metrics_24h)

                # Check for alerts
                alerts = self.quality_monitor.check_quality_alerts(metrics_24h)

                for alert in alerts:
                    self.alert_manager.send_alert({
                        'source': 'model_quality',
                        **alert
                    })

                # Log model version quality
                version_metrics = self.quality_monitor.compute_model_version_quality(window_hours=24, k=20)
                if version_metrics:
                    logger.info(f"Model version quality: {json.dumps(version_metrics, indent=2)}")

        except Exception as e:
            logger.error(f"Error in quality check: {e}", exc_info=True)

    def run_drift_check(self):
        """Run data drift monitoring"""
        try:
            logger.info("Running drift check...")

            # Run drift detection
            self.drift_monitor.run_drift_detection(['hourly', 'daily'])

            # Check for critical drift
            rating_drift = self.drift_monitor.detect_rating_drift('hourly')

            if rating_drift and rating_drift['drift_detected']:
                severity = 'critical' if rating_drift['drift_severity'] == 'high' else 'warning'

                self.alert_manager.send_alert({
                    'source': 'data_drift',
                    'alert_type': 'data_drift_detected',
                    'severity': severity,
                    'message': f"Data drift detected: Normalized WD={rating_drift['normalized_wd']:.3f}",
                    'metric_value': rating_drift['normalized_wd'],
                    'threshold_value': 0.15
                })

        except Exception as e:
            logger.error(f"Error in drift check: {e}", exc_info=True)

    def generate_monitoring_dashboard(self) -> Dict:
        """
        Generate comprehensive monitoring dashboard data.
        """
        logger.info("Generating monitoring dashboard...")

        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'health': {},
            'quality': {},
            'drift': {},
            'alerts': {}
        }

        try:
            # Health metrics
            health_summary = self.health_monitor.get_health_summary(hours=24)
            dashboard['health'] = {
                '24h_summary': health_summary,
                'current_1h': self.health_monitor.compute_availability_metrics(window_minutes=60)
            }

        except Exception as e:
            logger.error(f"Error getting health metrics: {e}")
            dashboard['health'] = {'error': str(e)}

        try:
            # Quality metrics
            quality_report = self.quality_monitor.generate_quality_report(hours=24)
            dashboard['quality'] = quality_report

        except Exception as e:
            logger.error(f"Error getting quality metrics: {e}")
            dashboard['quality'] = {'error': str(e)}

        try:
            # Drift metrics
            drift_report = self.drift_monitor.generate_drift_report(days=7)
            dashboard['drift'] = drift_report

        except Exception as e:
            logger.error(f"Error getting drift metrics: {e}")
            dashboard['drift'] = {'error': str(e)}

        try:
            # Recent alerts
            dashboard['alerts'] = {
                'recent_24h': self.alert_manager.get_recent_alerts(hours=24),
                'count': len(self.alert_manager.get_recent_alerts(hours=24))
            }

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            dashboard['alerts'] = {'error': str(e)}

        # Save dashboard to file
        dashboard_file = os.path.join(self.reports_dir, 'monitoring_dashboard.json')
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)

        logger.info(f"Dashboard saved to {dashboard_file}")

        return dashboard

    def start_monitoring(self, test_mode: bool = False):
        """
        Start continuous monitoring with scheduled checks.
        """
        logger.info("Starting unified monitoring system...")

        if test_mode:
            logger.info("TEST MODE: Running all checks every 5 minutes")
            schedule.every(5).minutes.do(self.run_health_check)
            schedule.every(5).minutes.do(self.run_quality_check)
            schedule.every(5).minutes.do(self.run_drift_check)
            schedule.every(5).minutes.do(self.generate_monitoring_dashboard)
        else:
            logger.info("PRODUCTION MODE: Scheduling checks at optimal intervals")
            # Health check every 5 minutes
            schedule.every(5).minutes.do(self.run_health_check)

            # Quality check every hour
            schedule.every().hour.at(":10").do(self.run_quality_check)

            # Drift check every hour
            schedule.every().hour.at(":15").do(self.run_drift_check)

            # Dashboard update every 15 minutes
            schedule.every(15).minutes.do(self.generate_monitoring_dashboard)

        # Run initial checks
        logger.info("Running initial monitoring checks...")
        self.run_health_check()
        self.run_quality_check()
        self.run_drift_check()
        self.generate_monitoring_dashboard()

        # Run monitoring loop
        logger.info("Monitoring system is running...")
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Monitoring system stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(60)

    def close(self):
        """Close all monitoring components"""
        logger.info("Closing monitoring components...")
        self.health_monitor.close()
        self.quality_monitor.close()
        self.drift_monitor.close()


if __name__ == "__main__":
    logger.info("Starting Unified Monitoring System")

    # Check for test mode
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'

    # Load alert configuration (optional)
    alert_config = {
        'email_enabled': False,  # Set to True and configure SMTP settings
        'smtp': {
            # 'host': 'smtp.gmail.com',
            # 'port': 587,
            # 'username': 'your-email@gmail.com',
            # 'password': 'your-password',
            # 'from_email': 'your-email@gmail.com',
            # 'to_email': 'team@example.com'
        },
        'slack_webhook_url': None  # Add Slack webhook URL here if desired
    }

    monitor = UnifiedMonitor(alert_config=alert_config)

    try:
        monitor.start_monitoring(test_mode=test_mode)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        monitor.close()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        monitor.close()
