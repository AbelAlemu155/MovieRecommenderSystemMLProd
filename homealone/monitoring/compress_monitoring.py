import os
from dotenv import load_dotenv
import re
import logging
import time
from kafka_reader import KafkaSSHClient
from datetime import datetime
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import NoBrokersAvailable
from prometheus_client import Gauge, Counter, start_http_server
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_message(message):
    """Parse recommendation request from Kafka message."""
    pattern = r'^([^,]+),([^,]+),recommendation request ([^,]+), status (\d+), result: (.+), (\d+)\s*ms$'
    match = re.match(pattern, message)

    if match:
        time, userid, server, status, recommendations, responsetime = match.groups()
        responsetime_str = responsetime.strip().replace(' ms', '')
        return {
            "time": time,
            "userid": userid,
            "server": server,
            "status": status,
            "recommendations": recommendations,
            "responsetime": float(responsetime_str)
        }
    return None


def compute_drift_metrics(db_path, drift_metrics):
    """
    Compute data drift metrics in background thread.
    Updates Prometheus gauges with drift detection results.
    """
    logger.info("Starting drift monitoring thread...")

    while True:
        try:
            # Connect to database
            conn = sqlite3.connect(db_path, timeout=60.0)
            conn.execute('PRAGMA journal_mode=WAL')

            # Get reference data (7-8 days ago)
            ref_query = """
                SELECT rating
                FROM ratings
                WHERE rated_at BETWEEN
                    datetime('now', '-8 days') AND datetime('now', '-1 days')
            """
            ref_df = pd.read_sql_query(ref_query, conn)

            # Process each window type
            windows = {
                'hourly': 1,
                'daily': 24,
                'weekly': 168
            }

            for window_type, hours in windows.items():
                # Get current data for this window
                curr_query = f"""
                    SELECT rating
                    FROM ratings
                    WHERE rated_at >= datetime('now', '-{hours} hours')
                """
                curr_df = pd.read_sql_query(curr_query, conn)

                if len(ref_df) >= 100 and len(curr_df) >= 30:
                    ref_data = ref_df['rating'].values
                    curr_data = curr_df['rating'].values

                    # Compute KS test
                    ks_statistic, ks_p_value = stats.ks_2samp(ref_data, curr_data)

                    # Compute Wasserstein distance
                    wd = wasserstein_distance(ref_data, curr_data)
                    ref_std = np.std(ref_data)
                    normalized_wd = wd / ref_std if ref_std > 0 else 0

                    # Determine drift severity (0=none, 1=low, 2=medium, 3=high)
                    if normalized_wd > 0.3:
                        severity_level = 3  # high
                    elif normalized_wd > 0.15:
                        severity_level = 2  # medium
                    elif normalized_wd > 0.1:
                        severity_level = 1  # low
                    else:
                        severity_level = 0  # none

                    # Update Prometheus metrics
                    drift_metrics['severity'].labels(window_type=window_type).set(severity_level)
                    drift_metrics['wasserstein'].labels(window_type=window_type).set(normalized_wd)
                    drift_metrics['ks_statistic'].labels(window_type=window_type).set(ks_statistic)
                    drift_metrics['ks_pvalue'].labels(window_type=window_type).set(ks_p_value)

                    # Update alert counter if drift detected
                    if severity_level > 0:
                        drift_metrics['alerts'].labels(alert_type='data_drift_detected').inc()

                    logger.info(f"Drift computed for {window_type}: WD={normalized_wd:.4f}, KS={ks_statistic:.4f}, p={ks_p_value:.4f}, severity={severity_level}")
                else:
                    logger.warning(f"Insufficient data for {window_type} drift: ref={len(ref_df)}, curr={len(curr_df)}")
                    # Set default values
                    drift_metrics['severity'].labels(window_type=window_type).set(0)
                    drift_metrics['wasserstein'].labels(window_type=window_type).set(0)
                    drift_metrics['ks_statistic'].labels(window_type=window_type).set(0)
                    drift_metrics['ks_pvalue'].labels(window_type=window_type).set(1.0)

            conn.close()

        except Exception as e:
            logger.error(f"Error computing drift metrics: {e}", exc_info=True)

        # Sleep for 5 minutes before next check
        time.sleep(300)


def main():
    load_dotenv()

    # Start Prometheus HTTP server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    # Define service health metrics
    TOTAL_REQUESTS = Counter('recommendation_requests_total', 'Total recommendation requests received')
    SUCCESSFUL_REQUESTS = Counter('successful_recommendation_requests_total', 'Total successful recommendation requests (status 200)')
    SERVICE_AVAILABILITY = Gauge('recommendation_service_availability_percentage', 'Service availability percentage')

    # Define data drift metrics (matching Grafana dashboard queries)
    drift_metrics = {
        'severity': Gauge(
            'recommendation_data_drift_severity',
            'Data drift severity level (0=none, 1=low, 2=medium, 3=high)',
            ['window_type']
        ),
        'wasserstein': Gauge(
            'recommendation_data_drift_wasserstein_normalized',
            'Normalized Wasserstein distance for drift detection',
            ['window_type']
        ),
        'ks_statistic': Gauge(
            'recommendation_data_drift_ks_statistic',
            'Kolmogorov-Smirnov test statistic',
            ['window_type']
        ),
        'ks_pvalue': Gauge(
            'recommendation_data_drift_ks_pvalue',
            'Kolmogorov-Smirnov test p-value',
            ['window_type']
        ),
        'alerts': Counter(
            'recommendation_alerts_total',
            'Total number of drift alerts triggered',
            ['alert_type']
        )
    }

    # Initialize alerts counter to 0 (so it shows in Grafana even when no alerts)
    drift_metrics['alerts'].labels(alert_type='data_drift_detected')

    # Start drift monitoring in background thread
    db_path = '/app/db/home_alone.db'
    drift_thread = Thread(
        target=compute_drift_metrics,
        args=(db_path, drift_metrics),
        daemon=True
    )
    drift_thread.start()
    logger.info("Drift monitoring thread started")

    # Setup Kafka SSH client
    kafka_client = KafkaSSHClient(
        username=os.getenv("TUNNEL_USER"),
        host=os.getenv("TUNNEL_IP"),
        password=os.getenv("TUNNEL_PASSWORD"),
        local_port=os.getenv("TUNNEL_PORT", "9092"),
        remote_port=os.getenv("TUNNEL_PORT", "9092")
    )

    # Ensure SSH tunnel is running
    kafka_client.ensure_tunnel()

    # Connect to Kafka with retry logic
    # NOTE: Don't subscribe to topic in constructor - we'll manually assign partitions
    consumer = None
    while consumer is None:
        try:
            logger.info("Attempting to connect to Kafka...")
            consumer = KafkaConsumer(
                bootstrap_servers=f"localhost:{kafka_client.local_port}",
                group_id='team10-monitor-group',
                auto_offset_reset='earliest',  # Changed from 'latest' to enable seeking to specific timestamp
                enable_auto_commit=False,
                consumer_timeout_ms=10000
            )
            logger.info("Successfully connected to Kafka!")
        except NoBrokersAvailable:
            logger.warning("No Kafka brokers available. Retrying in 5 seconds...")
            time.sleep(5)

    # Seek to specific timestamp: November 14, 2025
    target_date = datetime(2025, 11, 14, 0, 0, 0)  # Midnight on Nov 14, 2025
    target_timestamp_ms = int(target_date.timestamp() * 1000)  # Convert to milliseconds

    logger.info(f"Seeking to timestamp: {target_date} ({target_timestamp_ms}ms)")

    # Get partitions for the topic
    partitions = consumer.partitions_for_topic('movielog10')
    if partitions:
        # Create TopicPartition objects for all partitions
        topic_partitions = [TopicPartition('movielog10', p) for p in partitions]

        # Assign partitions to consumer (manual assignment instead of subscribe)
        consumer.assign(topic_partitions)
        logger.info(f"Assigned {len(topic_partitions)} partitions for topic 'movielog10'")

        # Get offsets for the target timestamp
        offsets = consumer.offsets_for_times(
            {tp: target_timestamp_ms for tp in topic_partitions}
        )

        # Seek to the calculated offsets
        for tp, offset_and_timestamp in offsets.items():
            if offset_and_timestamp is not None:
                consumer.seek(tp, offset_and_timestamp.offset)
                logger.info(f"Seeking partition {tp.partition} to offset {offset_and_timestamp.offset}")
            else:
                # If timestamp is beyond available data, seek to end
                logger.warning(f"No data found for partition {tp.partition} at timestamp {target_date}")
                consumer.seek_to_end([tp])
    else:
        logger.error("No partitions found for topic 'movielog10'")

    logger.info("Starting to consume recommendation requests from November 14, 2025...")

    try:
        for message in consumer:
            msg_str = message.value.decode('utf-8')
            parsed = process_message(msg_str)

            if parsed is not None:
                # Update counters
                TOTAL_REQUESTS.inc()

                # Log every recommendation request
                logger.info(f"Recommendation request - User: {parsed['userid']}, Status: {parsed['status']}, Time: {parsed['responsetime']}ms")

                # Update success counter
                if parsed['status'] == '200':
                    SUCCESSFUL_REQUESTS.inc()
                    logger.info(f"✓ SUCCESS - User: {parsed['userid']}, Recommendations: {parsed['recommendations']}")

                # Calculate and update service availability percentage
                total = TOTAL_REQUESTS._value.get()
                successful = SUCCESSFUL_REQUESTS._value.get()
                availability = (successful / total * 100) if total > 0 else 0
                SERVICE_AVAILABILITY.set(availability)

    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        consumer.close()
        logger.info("Consumer closed.")


if __name__ == "__main__":
    main()
