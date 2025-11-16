"""
Monitoring Configuration
Loads settings from environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MonitoringConfig:
    """Configuration for the monitoring system"""

    # Database
    DB_PATH = os.getenv('DB_PATH', '/app/db/home_alone.db')

    # Reports Directory
    REPORTS_DIR = os.getenv('REPORTS_DIR', '/app/reports')

    # Prometheus
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '8000'))

    # Test Mode (runs all checks every 5 minutes)
    TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'

    # Alert Configuration
    ALERT_CONFIG = {
        # Email Alerts
        'email_enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
        'smtp': {
            'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('ALERT_EMAIL_FROM', 'monitoring@homealone.com'),
            'to_email': os.getenv('ALERT_EMAIL_TO', 'pgithiri@andrew.cmu.edu'),
        },

        # Slack Alerts
        'slack_webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
    }

    # Alert Thresholds
    HEALTH_THRESHOLDS = {
        'availability_critical': float(os.getenv('ALERT_AVAILABILITY_CRITICAL', '70')),
        'availability_warning': float(os.getenv('ALERT_AVAILABILITY_WARNING', '90')),
        'error_rate_critical': float(os.getenv('ALERT_ERROR_RATE_CRITICAL', '10')),
        'response_time_p95_warning': float(os.getenv('ALERT_RESPONSE_TIME_P95', '500')),
    }

    QUALITY_THRESHOLDS = {
        'hit_rate_critical': float(os.getenv('ALERT_HIT_RATE_CRITICAL', '10')),
        'hit_rate_warning': float(os.getenv('ALERT_HIT_RATE_WARNING', '20')),
        'engagement_rate_warning': float(os.getenv('ALERT_ENGAGEMENT_RATE_WARNING', '5')),
    }

    DRIFT_THRESHOLDS = {
        'ks_p_value': float(os.getenv('ALERT_DRIFT_KS_PVALUE', '0.05')),
        'wasserstein_medium': float(os.getenv('ALERT_DRIFT_WD_MEDIUM', '0.15')),
        'wasserstein_high': float(os.getenv('ALERT_DRIFT_WD_HIGH', '0.3')),
    }

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if cls.ALERT_CONFIG['email_enabled']:
            smtp = cls.ALERT_CONFIG['smtp']
            if not smtp['username'] or not smtp['password']:
                print("WARNING: Email alerts enabled but SMTP credentials not configured")
                return False
        return True

    @classmethod
    def print_config(cls):
        """Print current configuration (mask sensitive data)"""
        print("=" * 60)
        print("MONITORING CONFIGURATION")
        print("=" * 60)
        print(f"Database Path: {cls.DB_PATH}")
        print(f"Reports Directory: {cls.REPORTS_DIR}")
        print(f"Prometheus Port: {cls.PROMETHEUS_PORT}")
        print(f"Test Mode: {cls.TEST_MODE}")
        print()
        print("ALERT CONFIGURATION:")
        print(f"  Email Enabled: {cls.ALERT_CONFIG['email_enabled']}")
        if cls.ALERT_CONFIG['email_enabled']:
            print(f"    SMTP Host: {cls.ALERT_CONFIG['smtp']['host']}")
            print(f"    SMTP Port: {cls.ALERT_CONFIG['smtp']['port']}")
            print(f"    From: {cls.ALERT_CONFIG['smtp']['from_email']}")
            print(f"    To: {cls.ALERT_CONFIG['smtp']['to_email']}")
            print(f"    Username: {'*' * len(cls.ALERT_CONFIG['smtp']['username'])}")
            print(f"    Password: {'*' * 8 if cls.ALERT_CONFIG['smtp']['password'] else 'NOT SET'}")
        print(f"  Slack Enabled: {bool(cls.ALERT_CONFIG['slack_webhook_url'])}")
        print()
        print("THRESHOLDS:")
        print(f"  Availability Critical: < {cls.HEALTH_THRESHOLDS['availability_critical']}%")
        print(f"  Availability Warning: < {cls.HEALTH_THRESHOLDS['availability_warning']}%")
        print(f"  Error Rate Critical: > {cls.HEALTH_THRESHOLDS['error_rate_critical']}%")
        print(f"  P95 Response Time Warning: > {cls.HEALTH_THRESHOLDS['response_time_p95_warning']}ms")
        print(f"  Hit Rate Critical: < {cls.QUALITY_THRESHOLDS['hit_rate_critical']}%")
        print(f"  Drift WD High: > {cls.DRIFT_THRESHOLDS['wasserstein_high']}")
        print("=" * 60)


if __name__ == '__main__':
    # Test configuration
    MonitoringConfig.print_config()
    if MonitoringConfig.validate():
        print("\n✅ Configuration is valid")
    else:
        print("\n⚠️  Configuration has warnings")
