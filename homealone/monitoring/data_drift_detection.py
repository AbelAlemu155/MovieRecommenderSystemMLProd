import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """
    Computes drift metrics over multiple time windows using KS test and Wasserstein Distance.
    """
    def __init__(self, db_path: str = '/app/db/home_alone.db'):
        self.db_path = db_path
        self.conn = none

        self.windows = {
            'hourly': {'hours': 1, 'check_interval_minutes': 5},
            'daily': {'hours': 24, 'check_interval_minutes': 60},
            'weekly': {'hours': 168, 'check_interval_minutes': 240},
        }

        self.reference_cache = {}
        self.cache_ttl = timedelta(hours=24)
        self.last_cache_update = None

        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, timeout=30)
        self.conn.execute('PRAGMA journal_mode=WAL')
        logger.info(f"Connected to database: {self.db_path}")

    def _create_tables(self):
        """Create tables for drift metrics and visualization"""
        cursor = self.conn.cursor()
        
        # Table to store drift metrics over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                window_type TEXT NOT NULL,  -- hourly, daily, weekly
                metric_name TEXT NOT NULL,  -- rating_ks, rating_wd, activity_ks, etc.
                ks_statistic REAL,
                ks_p_value REAL,
                wasserstein_distance REAL,
                normalized_wd REAL,  -- WD normalized by std dev
                ref_mean REAL,
                ref_std REAL,
                curr_mean REAL,
                curr_std REAL,
                ref_sample_size INTEGER,
                curr_sample_size INTEGER,
                drift_detected BOOLEAN,
                drift_severity TEXT,  -- none, low, medium, high
                details TEXT  -- JSON with additional info
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drift_timestamp 
            ON drift_metrics(timestamp, window_type, metric_name)
        ''')
        
        self.conn.commit()
        logger.info("Drift monitoring tables initialized")

    def _get_reference_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load reference data with caching for efficiency.
        Reference: Data from 7-14 days ago (stable baseline)
        """
        cache_expired = (
            self.last_cache_update is None or 
            datetime.now() - self.last_cache_update > self.cache_ttl
        )
        
        if force_refresh or cache_expired or 'ratings' not in self.reference_cache:
            logger.info("Loading reference data from database...")
            
            query = """
                SELECT user_id, movie_id, rating, rated_at
                FROM ratings
                WHERE rated_at BETWEEN 
                    datetime('now', '-14 days') AND datetime('now', '-7 days')
            """
            
            ref_data = pd.read_sql_query(query, self.conn)
            
            if len(ref_data) < 100:
                logger.warning("Insufficient reference data for drift detection")
                return None
            
            self.reference_cache['ratings'] = ref_data
            self.last_cache_update = datetime.now()
            logger.info(f"Reference data cached: {len(ref_data)} ratings")
        
        return self.reference_cache['ratings']

    def _get_current_data(self, window_hours: int) -> pd.DataFrame:
        """Get current data for specified time window"""
        query = f"""
            SELECT user_id, movie_id, rating, rated_at
            FROM ratings
            WHERE rated_at >= datetime('now', '-{window_hours} hours')
        """
        return pd.read_sql_query(query, self.conn)

    def compute_ks_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict:
        """
        Compute Kolmogorov-Smirnov test
        Returns: statistic, p-value, and interpretation
        """
        ks_statistic, p_value = stats.ks_2samp(ref_data, curr_data)
        
        if p_value < 0.01:
            interpretation = "Strong evidence of drift"
            drift_detected = True
        elif p_value < 0.05:
            interpretation = "Moderate evidence of drift"
            drift_detected = True
        else:
            interpretation = "No significant drift"
            drift_detected = False
        
        return {
            'ks_statistic': float(ks_statistic),
            'ks_p_value': float(p_value),
            'drift_detected': drift_detected,
            'interpretation': interpretation
        }

    def compute_wasserstein_distance(self, ref_data: np.ndarray, curr_data: np.ndarray, ref_std: float = None) -> Dict:
        """
        Compute Wasserstein Distance (Earth Mover's Distance)
        Normalized by standard deviation for interpretability[web:283]
        """
        wd = wasserstein_distance(ref_data, curr_data)
        
        if ref_std is None:
            ref_std = np.std(ref_data)
        
        normalized_wd = wd / ref_std if ref_std > 0 else wd
        
        if normalized_wd > 0.3:
            interpretation = "High drift (>0.3 std dev)"
            drift_detected = True
            severity = "high"
        elif normalized_wd > 0.15:
            interpretation = "Moderate drift (>0.15 std dev)"
            drift_detected = True
            severity = "medium"
        elif normalized_wd > 0.1:
            interpretation = "Low drift (>0.1 std dev)"
            drift_detected = True
            severity = "low"
        else:
            interpretation = "No significant drift"
            drift_detected = False
            severity = "none"
        
        return {
            'wasserstein_distance': float(wd),
            'normalized_wd': float(normalized_wd),
            'drift_detected': drift_detected,
            'severity': severity,
            'interpretation': interpretation
        }

    def detect_rating_drift(self, window_type: str = 'hourly') -> Optional[Dict]:
        """
        Main drift detection for rating distribution
        Combines KS test and Wasserstein distance
        """
        window_hours = self.windows[window_type]['hours']
        
        # Get reference and current data
        ref_data = self._get_reference_data()
        if ref_data is None or len(ref_data) < 100:
            logger.warning("Insufficient reference data")
            return None
        
        curr_data = self._get_current_data(window_hours)
        if len(curr_data) < 50:
            logger.warning(f"Insufficient current data for {window_type} window")
            return None
        
        # Extract rating values
        ref_ratings = ref_data['rating'].values
        curr_ratings = curr_data['rating'].values
        
        # Compute statistics
        ref_mean = float(np.mean(ref_ratings))
        ref_std = float(np.std(ref_ratings))
        curr_mean = float(np.mean(curr_ratings))
        curr_std = float(np.std(curr_ratings))
        
        # Compute drift metrics
        ks_results = self.compute_ks_test(ref_ratings, curr_ratings)
        wd_results = self.compute_wasserstein_distance(ref_ratings, curr_ratings, ref_std)
        
        # Combine results
        drift_detected = ks_results['drift_detected'] or wd_results['drift_detected']
        
        # Determine overall severity
        if wd_results['severity'] == 'high' or ks_results['ks_p_value'] < 0.01:
            severity = 'high'
        elif wd_results['severity'] == 'medium' or ks_results['ks_p_value'] < 0.05:
            severity = 'medium'
        elif wd_results['severity'] == 'low':
            severity = 'low'
        else:
            severity = 'none'
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'window_type': window_type,
            'window_hours': window_hours,
            'metric_name': 'rating_distribution',
            
            # KS Test results
            'ks_statistic': ks_results['ks_statistic'],
            'ks_p_value': ks_results['ks_p_value'],
            
            # Wasserstein Distance results
            'wasserstein_distance': wd_results['wasserstein_distance'],
            'normalized_wd': wd_results['normalized_wd'],
            
            # Statistics
            'ref_mean': ref_mean,
            'ref_std': ref_std,
            'curr_mean': curr_mean,
            'curr_std': curr_std,
            'mean_shift': abs(curr_mean - ref_mean),
            
            # Sample sizes
            'ref_sample_size': len(ref_ratings),
            'curr_sample_size': len(curr_ratings),
            
            # Detection results
            'drift_detected': drift_detected,
            'drift_severity': severity,
            
            # Details
            'details': json.dumps({
                'ks_interpretation': ks_results['interpretation'],
                'wd_interpretation': wd_results['interpretation']
            })
        }
        
        # Log results
        if drift_detected:
            logger.warning(
                f"DRIFT DETECTED [{window_type}]: "
                f"KS p-value={ks_results['ks_p_value']:.4f}, "
                f"Normalized WD={wd_results['normalized_wd']:.3f}, "
                f"Severity={severity}"
            )
        else:
            logger.info(
                f"No drift [{window_type}]: "
                f"KS p-value={ks_results['ks_p_value']:.4f}, "
                f"Normalized WD={wd_results['normalized_wd']:.3f}"
            )
        
        return result

    