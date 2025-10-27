import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from datetime import datetime, timedelta
import logging
import json
import schedule
import time
from threading import Thread
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """
    Computes drift metrics over multiple time windows using KS test and Wasserstein Distance.
    """
    def __init__(self, 
        db_path: str = '/app/db/home_alone.db', 
        reports_dir: str = '/app/reports'
        ):
        self.db_path = db_path
        self.reports_dir = reports_dir
        self.conn = None

        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'hourly'), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'daily'), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, 'weekly'), exist_ok=True)

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
                    datetime('now', '-8 days') AND datetime('now', '-1 days')
            """
            # query = """
            #     SELECT user_id, movie_id, rating, rated_at
            #     FROM ratings
            #     WHERE rated_at BETWEEN 
            #         datetime('now', '-14 days') AND datetime('now', '-7 days')
            # """
            
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

    def plot_drift_trends(
        self, 
        window_type: str = 'hourly',
        days: int = 7,
        save: bool = True
    ):
        """
        Generate comprehensive drift visualization
        Automatically saves to shared reports folder
        """
        logger.info(f"Generating drift plots for {window_type} window (last {days} days)...")
        
        # Get drift history for ratings
        query = f"""
            SELECT 
                timestamp,
                ks_p_value,
                normalized_wd,
                ref_mean,
                curr_mean,
                drift_detected,
                drift_severity
            FROM drift_metrics
            WHERE window_type = ?
                AND metric_name = 'rating_distribution'
                AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(window_type,))
        
        if len(df) == 0:
            logger.warning(f"No data available for {window_type} drift plots")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        
        # Subplot 1: Normalized Wasserstein Distance
        ax1 = fig.add_subplot(gs[0])
        
        # Plot WD with drift regions
        ax1.plot(df['timestamp'], df['normalized_wd'], 
                marker='o', linewidth=2, markersize=4, 
                color='#2E86AB', label='Normalized WD')
        
        # Add threshold lines
        ax1.axhline(y=0.1, color='#F6AE2D', linestyle='--', 
                   linewidth=1.5, label='Low drift (0.1)')
        ax1.axhline(y=0.15, color='#F26419', linestyle='--', 
                   linewidth=1.5, label='Moderate drift (0.15)')
        ax1.axhline(y=0.3, color='#C1121F', linestyle='--', 
                   linewidth=1.5, label='High drift (0.3)')
        
        # Highlight drift periods
        drift_points = df[df['drift_detected'] == 1]
        if len(drift_points) > 0:
            ax1.scatter(drift_points['timestamp'], drift_points['normalized_wd'],
                       color='red', s=100, zorder=5, marker='X', 
                       label='Drift detected')
        
        ax1.set_ylabel('Normalized Wasserstein Distance', fontsize=11, fontweight='bold')
        ax1.set_title(f'Data Drift Monitoring - {window_type.capitalize()} Window', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_ylim(bottom=0)
        
        # Subplot 2: KS Test p-value
        ax2 = fig.add_subplot(gs[1])
        
        ax2.plot(df['timestamp'], df['ks_p_value'], 
                marker='s', linewidth=2, markersize=4, 
                color='#55A630', label='KS p-value')
        
        # Significance threshold
        ax2.axhline(y=0.05, color='#C1121F', linestyle='--', 
                   linewidth=1.5, label='Significance (p=0.05)')
        ax2.axhline(y=0.01, color='#780000', linestyle='--', 
                   linewidth=1.5, label='Strong significance (p=0.01)')
        
        ax2.set_ylabel('KS Test p-value', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.set_ylim(bottom=0, top=max(0.1, df['ks_p_value'].max() * 1.1))
        
        # Subplot 3: Mean Rating Shift
        ax3 = fig.add_subplot(gs[2])
        
        ax3.plot(df['timestamp'], df['ref_mean'], 
                linewidth=2, color='#2E86AB', 
                label='Reference mean', linestyle='--', alpha=0.7)
        ax3.plot(df['timestamp'], df['curr_mean'], 
                linewidth=2, color='#F26419', 
                label='Current mean')
        
        ax3.set_ylabel('Average Rating', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.set_ylim(1, 5)  # Rating scale 1-5
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', rotation=45)
        
        # plt.tight_layout()
        
        # Save to shared folder
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'drift_trends_{window_type}.png'
            filepath = os.path.join(self.reports_dir, window_type, filename)
            
            # Also save timestamped version for history
            # filename_ts = f'drift_trends_{window_type}_{timestamp}.png'
            # filepath_ts = os.path.join(self.reports_dir, window_type, filename_ts)
            
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            # fig.savefig(filepath_ts, dpi=150, bbox_inches='tight')
            
            logger.info(f"Drift plot saved: {filepath}")
            plt.close(fig)
            
            return filepath
        
        return fig

    def plot_drift_heatmap(self, days: int = 7, save: bool = True):
        """
        Create heatmap showing drift intensity over time and windows
        """
        logger.info("Generating drift heatmap...")
        
        # Get all drift data
        query = f"""
            SELECT 
                timestamp,
                window_type,
                normalized_wd,
                drift_severity
            FROM drift_metrics
            WHERE metric_name = 'rating_distribution'
                AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) == 0:
            logger.warning("No data for heatmap")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            values='normalized_wd', 
            index='window_type',
            columns='date',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(d) for d in pivot.columns], rotation=45)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized WD', rotation=270, labelpad=15)
        
        ax.set_title('Drift Intensity Heatmap (All Windows)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Window Type', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            filename = 'drift_heatmap.png'
            filepath = os.path.join(self.reports_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Heatmap saved: {filepath}")
            plt.close(fig)
            return filepath
        
        return fig

    def generate_summary_report(self, days: int = 7, save: bool = True):
        """
        Generate HTML summary report with statistics
        """
        logger.info("Generating summary report...")
        
        report = self.generate_drift_report(days)
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Monitoring Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2E86AB;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .stats {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #2E86AB;
                    color: white;
                }}
                .alert {{
                    background-color: #ff4444;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .ok {{
                    background-color: #00C851;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Data Drift Monitoring Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Monitoring period: Last {days} days</p>
            </div>
        """
        
        # Add statistics for each window
        for window_type, stats in report.items():
            drift_pct = stats['drift_percentage']
            alert_class = 'alert' if drift_pct > 30 else 'ok'
            
            html += f"""
            <div class="stats">
                <h2>{window_type.capitalize()} Window</h2>
                <div class="{alert_class}">
                    Drift detected in {drift_pct:.1f}% of checks 
                    ({stats['drift_count']}/{stats['total_checks']})
                </div>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Average Normalized WD</td>
                        <td>{stats['avg_normalized_wd']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Maximum Normalized WD</td>
                        <td>{stats['max_normalized_wd']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Average KS p-value</td>
                        <td>{stats['avg_ks_p_value']:.4f}</td>
                    </tr>
                </table>
                <p><img src="{window_type}/drift_trends_{window_type}.png" 
                        style="max-width:100%; height:auto;"></p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        if save:
            filepath = os.path.join(self.reports_dir, 'drift_report.html')
            with open(filepath, 'w') as f:
                f.write(html)
            logger.info(f"HTML report saved: {filepath}")
            return filepath
        
        return html

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


    def detect_activity_drift(self, window_type: str = 'hourly') -> Optional[Dict]:
        """Detect drift in user activity patterns (ratings per user)"""
        window_hours = self.windows[window_type]['hours']
        
        ref_data = self._get_reference_data()
        if ref_data is None:
            return None
        
        curr_data = self._get_current_data(window_hours)
        if len(curr_data) < 30:
            return None
        
        # Calculate ratings per user
        ref_activity = ref_data.groupby('user_id').size().values
        curr_activity = curr_data.groupby('user_id').size().values
        
        if len(curr_activity) < 20:
            return None
        
        # Compute drift metrics
        ks_results = self.compute_ks_test(ref_activity, curr_activity)
        wd_results = self.compute_wasserstein_distance(
            ref_activity, 
            curr_activity,
            np.std(ref_activity)
        )
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'window_type': window_type,
            'window_hours': window_hours,
            'metric_name': 'user_activity',
            'ks_statistic': ks_results['ks_statistic'],
            'ks_p_value': ks_results['ks_p_value'],
            'wasserstein_distance': wd_results['wasserstein_distance'],
            'normalized_wd': wd_results['normalized_wd'],
            'ref_mean': float(np.mean(ref_activity)),
            'ref_std': float(np.std(ref_activity)),
            'curr_mean': float(np.mean(curr_activity)),
            'curr_std': float(np.std(curr_activity)),
            'ref_sample_size': len(ref_activity),
            'curr_sample_size': len(curr_activity),
            'drift_detected': ks_results['drift_detected'] or wd_results['drift_detected'],
            'drift_severity': wd_results['severity'],
            'details': json.dumps({
                'ks_interpretation': ks_results['interpretation'],
                'wd_interpretation': wd_results['interpretation']
            })
        }
        
        return result

    def save_drift_metrics(self, metrics: Dict):
        """Save computed drift metrics to database for visualization"""
        if metrics is None:
            return
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO drift_metrics (
                window_type, metric_name,
                ks_statistic, ks_p_value,
                wasserstein_distance, normalized_wd,
                ref_mean, ref_std, curr_mean, curr_std,
                ref_sample_size, curr_sample_size,
                drift_detected, drift_severity, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['window_type'],
            metrics['metric_name'],
            metrics['ks_statistic'],
            metrics['ks_p_value'],
            metrics['wasserstein_distance'],
            metrics['normalized_wd'],
            metrics['ref_mean'],
            metrics['ref_std'],
            metrics['curr_mean'],
            metrics['curr_std'],
            metrics['ref_sample_size'],
            metrics['curr_sample_size'],
            metrics['drift_detected'],
            metrics['drift_severity'],
            metrics['details']
        ))
        self.conn.commit()

    def run_drift_detection(self, window_types: List[str] = None):
        """
        Run drift detection for specified windows
        This is the main entry point for scheduled runs
        """
        if window_types is None:
            window_types = ['hourly', 'daily', 'weekly']
        
        logger.info(f"Starting drift detection for windows: {window_types}")
        
        for window_type in window_types:
            # Check rating distribution drift
            rating_metrics = self.detect_rating_drift(window_type)
            if rating_metrics:
                self.save_drift_metrics(rating_metrics)
            
            # Check user activity drift
            activity_metrics = self.detect_activity_drift(window_type)
            if activity_metrics:
                self.save_drift_metrics(activity_metrics)
        
        logger.info("Generating visualizations...")
        for window_type in window_types:
            self.plot_drift_trends(window_type, days=7, save=True)
        
        # Generate heatmap
        self.plot_drift_heatmap(days=7, save=True)
        
        # Generate HTML report
        self.generate_summary_report(days=7, save=True)
        
        logger.info("Drift detection and visualization complete!")

    def get_drift_history(self, window_type: str = 'hourly', metric_name: str = 'rating_distribution', days: int = 7
    ) -> pd.DataFrame:
        """
        Get drift history for visualization
        Returns time series of drift metrics (rating_distribution or user_activity)
        """
        query = f"""
            SELECT 
                timestamp,
                ks_p_value,
                normalized_wd,
                curr_mean,
                drift_detected,
                drift_severity
            FROM drift_metrics
            WHERE window_type = ?
                AND metric_name = ?
                AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(window_type, metric_name))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def generate_drift_report(self, days: int = 7) -> Dict:
        """Generate summary report of drift over past N days"""
        report = {}
        
        for window_type in ['hourly', 'daily', 'weekly']:
            query = f"""
                SELECT 
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN drift_detected = 1 THEN 1 ELSE 0 END) as drift_count,
                    AVG(normalized_wd) as avg_wd,
                    MAX(normalized_wd) as max_wd,
                    AVG(ks_p_value) as avg_ks_p_value
                FROM drift_metrics
                WHERE window_type = ?
                    AND timestamp >= datetime('now', '-{days} days')
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, (window_type,))
            row = cursor.fetchone()
            
            if row[0] > 0:
                report[window_type] = {
                    'total_checks': row[0],
                    'drift_count': row[1],
                    'drift_percentage': (row[1] / row[0]) * 100 if row[0] > 0 else 0,
                    'avg_normalized_wd': row[2],
                    'max_normalized_wd': row[3],
                    'avg_ks_p_value': row[4]
                }
        
        return report

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class DriftMonitorScheduler:
    """
    Scheduler to run drift detection at optimal intervals
    Runs as separate background process
    """
    
    def __init__(self, db_path: str = '/app/db/home_alone.db', 
    reports_dir: str = '/app/reports',
    test_mode: bool = False):
        self.monitor = DataDriftMonitor(db_path)
        self.running = False
        self.test_mode = test_mode
    
    def run_hourly_check(self):
        """Lightweight hourly check"""
        logger.info("Running hourly drift check...")
        self.monitor.run_drift_detection(['hourly'])
    
    def run_daily_check(self):
        """More comprehensive daily check"""
        logger.info("Running daily drift check...")
        self.monitor.run_drift_detection(['daily'])
        
        logger.info("Refreshing all visualizations...")
        for window_type in ['hourly', 'daily']:
            self.monitor.plot_drift_trends(window_type, days=7, save=True)
        
        self.monitor.generate_summary_report(days=7, save=True)
    
    def run_weekly_check(self):
        """Comprehensive weekly check"""
        logger.info("Running weekly drift check...")
        self.monitor.run_drift_detection(['weekly'])
        
        # Refresh with extended history
        for window_type in ['hourly', 'daily', 'weekly']:
            self.monitor.plot_drift_trends(window_type, days=30, save=True)
        
        self.monitor.plot_drift_heatmap(days=30, save=True)
        self.monitor.generate_summary_report(days=30, save=True)
        
        # Refresh reference data cache
        self.monitor._get_reference_data(force_refresh=True)
        report = self.monitor.generate_drift_report(days=7)
        logger.info(f"Weekly drift report: {report}")
    
    def start(self):
        """Start scheduled drift monitoring"""
        logger.info("Starting drift monitoring scheduler...")

        if self.test_mode:
            logger.info("TEST MODE: Running all checks every 5 minutes")
            schedule.every(5).minutes.do(self.run_hourly_check)
            schedule.every(5).minutes.do(self.run_daily_check)
            schedule.every(5).minutes.do(self.run_weekly_check)
        else:
            logger.info("PRODUCTION MODE: Scheduling checks at optimal intervals")
            # Schedule checks at optimal intervals
            schedule.every().hour.at(":05").do(self.run_hourly_check)  # Run at 5 min past hour
            schedule.every().day.at("02:00").do(self.run_daily_check)  # Run at 2 AM
            schedule.every().monday.at("03:00").do(self.run_weekly_check)  # Run Monday 3 AM
        
        # Run initial check immediately
        logger.info("Running initial drift check...")
        self.run_hourly_check()

        self.running = True
        
        # Run in loop
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start_in_background(self):
        """Start scheduler in background thread"""
        thread = Thread(target=self.start, daemon=True)
        thread.start()
        logger.info("Drift monitoring scheduler started in background")
        return thread
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        self.monitor.close()
        logger.info("Drift monitoring scheduler stopped")


# entry point for standalone drift monitoring service
if __name__ == "__main__":
    logger.info("Starting Drift Monitoring Service")
    test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
    if test_mode:
        logger.info("DRIFT MONITORING RUNNING IN TEST MODE")
    
    scheduler = DriftMonitorScheduler(test_mode=test_mode)
    
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        scheduler.stop()
