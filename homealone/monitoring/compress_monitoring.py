import os
from dotenv import load_dotenv
import re
import logging
import time
import psutil
import subprocess
from datetime import datetime
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import NoBrokersAvailable
from prometheus_client import Gauge, Counter, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KafkaSSHClient:
    """SSH tunnel-based Kafka client."""

    def __init__(self, username, host, password, local_port, remote_port):
        self.username = username
        self.host = host
        self.password = password
        self.local_port = local_port
        self.remote_port = remote_port
        self.ssh_proc = None

    def _is_tunnel_running(self):
        """Check if SSH tunnel is already running."""
        for proc in psutil.process_iter(attrs=["pid", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and "ssh" in cmdline[0] and "-L" in cmdline:
                    if f"{self.local_port}:localhost:{self.remote_port}" in cmdline and f"{self.username}@{self.host}" in cmdline:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _start_tunnel(self):
        """Start SSH tunnel using sshpass."""
        logger.info(f"Starting SSH tunnel: localhost:{self.local_port} -> {self.username}@{self.host}:{self.remote_port}")

        tunnel_cmd = [
            "sshpass", "-p", self.password,
            "ssh", "-L", f"{self.local_port}:localhost:{self.remote_port}",
            f"{self.username}@{self.host}", "-NT"
        ]

        self.ssh_proc = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        logger.info("SSH tunnel started. Waiting for it to establish...")
        time.sleep(3)

    def ensure_tunnel(self):
        """Ensure SSH tunnel is running, start it if not."""
        if not self._is_tunnel_running():
            logger.info("SSH tunnel not running. Starting it...")
            self._start_tunnel()
        else:
            logger.info("SSH tunnel already running.")


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


def main():
    load_dotenv()

    # Start Prometheus HTTP server
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")

    # Define metrics
    TOTAL_REQUESTS = Counter('recommendation_requests_total', 'Total recommendation requests received')
    SUCCESSFUL_REQUESTS = Counter('successful_recommendation_requests_total', 'Total successful recommendation requests (status 200)')
    SERVICE_AVAILABILITY = Gauge('recommendation_service_availability_percentage', 'Service availability percentage')

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
