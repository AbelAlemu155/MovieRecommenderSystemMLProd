"""
Model Quality Monitor - Tracks online prediction quality
Monitors model accuracy in production using actual user behavior
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelQualityMonitor:
    """
    Monitors recommendation model quality in production.
    Computes online accuracy metrics based on user engagement with recommendations.
    """

    def __init__(self,
                 db_path: str = '/app/db/home_alone.db',
                 reports_dir: str = '/app/reports'):
        self.db_path = db_path
        self.reports_dir = reports_dir
        self.conn = None

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'quality'), exist_ok=True)

        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.execute('PRAGMA journal_mode=WAL')
        logger.info(f"Connected to database: {self.db_path}")

    def _create_tables(self):
        """Create tables for model quality metrics"""
        cursor = self.conn.cursor()

        # Model quality metrics over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_quality_metrics (
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
                k_value INTEGER DEFAULT 20,
                details TEXT
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_quality_timestamp
            ON model_quality_metrics(timestamp, model_version)
        ''')

        # Quality alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_value REAL,
                threshold_value REAL,
                model_version TEXT,
                resolved BOOLEAN DEFAULT 0,
                resolved_at DATETIME
            )
        ''')

        self.conn.commit()
        logger.info("Model quality monitoring tables initialized")

    def compute_online_accuracy(self, window_hours: int = 24, k: int = 20, model_version: Optional[str] = None) -> Dict:
        """
        Compute online accuracy metrics by matching recommendations with actual user behavior.

        Metrics:
        - Hit Rate: % of users who watched at least one recommended movie
        - Precision@K: Average proportion of recommended movies that were watched
        - Engagement Rate: % of recommended movies that were watched
        - Avg Rank of Watched: Average position of watched movies in recommendation list
        """

        # Get recommendations from the past window
        rec_query = f"""
            SELECT user_id, recommendations, timestamp, model_version
            FROM recommendations
            WHERE timestamp >= datetime('now', '-{window_hours} hours')
        """

        if model_version:
            rec_query += f" AND model_version = '{model_version}'"

        recommendations_df = pd.read_sql_query(rec_query, self.conn)

        if len(recommendations_df) == 0:
            logger.warning(f"No recommendations found in last {window_hours} hours")
            return None

        # Get watch history from the same time period (+ some buffer for delayed watching)
        watch_query = f"""
            SELECT user_id, movie_id, start_time
            FROM watch_history
            WHERE start_time >= datetime('now', '-{window_hours + 2} hours')
        """

        watch_df = pd.read_sql_query(watch_query, self.conn)

        if len(watch_df) == 0:
            logger.warning("No watch history found for evaluation period")
            return None

        # Compute metrics
        hit_count = 0
        precision_sum = 0
        total_recommended = 0
        total_watched_from_rec = 0
        rank_sum = 0
        rank_count = 0
        total_users = len(recommendations_df)

        for _, rec_row in recommendations_df.iterrows():
            user_id = rec_row['user_id']
            rec_timestamp = pd.to_datetime(rec_row['timestamp'])

            # Parse recommended movie IDs
            recommended_movies = []
            if rec_row['recommendations']:
                try:
                    recommended_movies = rec_row['recommendations'].split(',')
                except:
                    continue

            if not recommended_movies:
                continue

            # Get movies watched by this user after the recommendation
            user_watches = watch_df[
                (watch_df['user_id'] == user_id) &
                (pd.to_datetime(watch_df['start_time']) >= rec_timestamp)
            ]['movie_id'].tolist()

            # Convert to string for comparison
            user_watches = [str(mid) for mid in user_watches]
            recommended_movies_str = [str(mid).strip() for mid in recommended_movies]

            # Hit Rate: did user watch any recommended movie?
            watched_from_rec = set(user_watches) & set(recommended_movies_str)
            if watched_from_rec:
                hit_count += 1

            # Precision: proportion of recommendations that were watched
            if len(recommended_movies_str) > 0:
                precision = len(watched_from_rec) / min(len(recommended_movies_str), k)
                precision_sum += precision
                total_recommended += len(recommended_movies_str)
                total_watched_from_rec += len(watched_from_rec)

            # Average rank of watched movies
            for watched_movie in watched_from_rec:
                try:
                    rank = recommended_movies_str.index(watched_movie) + 1
                    rank_sum += rank
                    rank_count += 1
                except ValueError:
                    pass

        # Calculate final metrics
        hit_rate = (hit_count / total_users) if total_users > 0 else 0
        precision_at_k = (precision_sum / total_users) if total_users > 0 else 0
        engagement_rate = (total_watched_from_rec / total_recommended) if total_recommended > 0 else 0
        avg_rank_watched = (rank_sum / rank_count) if rank_count > 0 else None

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'window_hours': window_hours,
            'model_version': model_version,
            'hit_rate': round(hit_rate, 4),
            'precision_at_k': round(precision_at_k, 4),
            'engagement_rate': round(engagement_rate, 4),
            'avg_rank_watched': round(avg_rank_watched, 2) if avg_rank_watched else None,
            'total_recommendations': total_users,
            'total_users': total_users,
            'k_value': k
        }

        logger.info(
            f"Model Quality [{window_hours}h, k={k}]: "
            f"Hit Rate={hit_rate:.2%}, "
            f"Precision@{k}={precision_at_k:.4f}, "
            f"Engagement={engagement_rate:.2%}"
        )

        return metrics

    def compute_model_version_quality(self, window_hours: int = 24, k: int = 20) -> List[Dict]:
        """
        Compute quality metrics separately for each model version (for A/B testing).
        """
        # Get all model versions from recommendations
        query = f"""
            SELECT DISTINCT model_version
            FROM recommendations
            WHERE timestamp >= datetime('now', '-{window_hours} hours')
              AND model_version IS NOT NULL
        """

        versions_df = pd.read_sql_query(query, self.conn)

        results = []
        for version in versions_df['model_version']:
            metrics = self.compute_online_accuracy(
                window_hours=window_hours,
                k=k,
                model_version=version
            )
            if metrics:
                results.append(metrics)

        return results

    def save_quality_metrics(self, metrics: Dict):
        """Save computed quality metrics to database"""
        if metrics is None:
            return

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO model_quality_metrics (
                window_hours, model_version, hit_rate, precision_at_k,
                engagement_rate, avg_rank_watched, total_recommendations,
                total_users, k_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['window_hours'],
            metrics['model_version'],
            metrics['hit_rate'],
            metrics['precision_at_k'],
            metrics['engagement_rate'],
            metrics['avg_rank_watched'],
            metrics['total_recommendations'],
            metrics['total_users'],
            metrics['k_value']
        ))
        self.conn.commit()

    def check_quality_alerts(self, metrics: Dict) -> List[Dict]:
        """
        Check if any quality alert thresholds are violated.
        """
        alerts = []

        # Alert: Low hit rate
        if metrics['hit_rate'] < 0.10:  # Less than 10% hit rate
            alerts.append({
                'alert_type': 'low_hit_rate',
                'severity': 'critical',
                'message': f"Hit rate dropped to {metrics['hit_rate']:.2%}",
                'metric_value': metrics['hit_rate'],
                'threshold_value': 0.10,
                'model_version': metrics['model_version']
            })
        elif metrics['hit_rate'] < 0.20:  # Less than 20% hit rate
            alerts.append({
                'alert_type': 'low_hit_rate',
                'severity': 'warning',
                'message': f"Hit rate at {metrics['hit_rate']:.2%}",
                'metric_value': metrics['hit_rate'],
                'threshold_value': 0.20,
                'model_version': metrics['model_version']
            })

        # Alert: Low engagement
        if metrics['engagement_rate'] < 0.05:  # Less than 5% engagement
            alerts.append({
                'alert_type': 'low_engagement',
                'severity': 'warning',
                'message': f"Engagement rate at {metrics['engagement_rate']:.2%}",
                'metric_value': metrics['engagement_rate'],
                'threshold_value': 0.05,
                'model_version': metrics['model_version']
            })

        # Log and save alerts
        for alert in alerts:
            logger.warning(f"QUALITY ALERT [{alert['severity']}]: {alert['message']}")
            self._save_alert(alert)

        return alerts

    def _save_alert(self, alert: Dict):
        """Save quality alert to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO quality_alerts
            (alert_type, severity, message, metric_value, threshold_value, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert['alert_type'],
            alert['severity'],
            alert['message'],
            alert['metric_value'],
            alert['threshold_value'],
            alert.get('model_version')
        ))
        self.conn.commit()

    def get_quality_trend(self, hours: int = 168, model_version: Optional[str] = None) -> pd.DataFrame:
        """
        Get quality metrics trend over time.
        """
        query = f"""
            SELECT
                timestamp,
                hit_rate,
                precision_at_k,
                engagement_rate,
                avg_rank_watched,
                model_version
            FROM model_quality_metrics
            WHERE timestamp >= datetime('now', '-{hours} hours')
        """

        if model_version:
            query += f" AND model_version = '{model_version}'"

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn)
        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def generate_quality_report(self, hours: int = 24) -> Dict:
        """
        Generate summary quality report.
        """
        query = f"""
            SELECT
                AVG(hit_rate) as avg_hit_rate,
                AVG(precision_at_k) as avg_precision,
                AVG(engagement_rate) as avg_engagement,
                AVG(avg_rank_watched) as avg_rank,
                SUM(total_recommendations) as total_recommendations,
                model_version
            FROM model_quality_metrics
            WHERE timestamp >= datetime('now', '-{hours} hours')
            GROUP BY model_version
        """

        df = pd.read_sql_query(query, self.conn)

        if len(df) == 0:
            return {
                'hours': hours,
                'overall': {},
                'by_version': []
            }

        # Overall metrics (across all versions)
        overall = {
            'avg_hit_rate': round(df['avg_hit_rate'].mean(), 4),
            'avg_precision': round(df['avg_precision'].mean(), 4),
            'avg_engagement': round(df['avg_engagement'].mean(), 4),
            'avg_rank': round(df['avg_rank'].mean(), 2) if df['avg_rank'].notna().any() else None,
            'total_recommendations': int(df['total_recommendations'].sum())
        }

        # By version
        by_version = df.to_dict('records')

        return {
            'hours': hours,
            'overall': overall,
            'by_version': by_version
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    logger.info("Starting Model Quality Monitor")

    monitor = ModelQualityMonitor()

    try:
        while True:
            logger.info("Computing model quality metrics...")

            # 24-hour quality metrics
            metrics_24h = monitor.compute_online_accuracy(window_hours=24, k=20)
            if metrics_24h:
                monitor.save_quality_metrics(metrics_24h)
                monitor.check_quality_alerts(metrics_24h)

            # Model version quality (for A/B testing)
            version_metrics = monitor.compute_model_version_quality(window_hours=24, k=20)
            if version_metrics:
                logger.info(f"Model version quality: {version_metrics}")
                for vm in version_metrics:
                    monitor.save_quality_metrics(vm)

            # Quality report
            report = monitor.generate_quality_report(hours=24)
            logger.info(f"24h Quality Report: {report}")

            # Wait 1 hour before next check
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        monitor.close()
    except Exception as e:
        logger.error(f"Error in quality monitor: {e}", exc_info=True)
        monitor.close()
