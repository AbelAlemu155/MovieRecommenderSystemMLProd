"""
Kafka Log Reader for Monitoring
Reads recommendation logs from Kafka stream for monitoring purposes
"""
import os
from dotenv import load_dotenv
from kafka import KafkaConsumer
import logging

logger = logging.getLogger(__name__)


class KafkaLogReader:
    """
    Simple Kafka consumer for reading recommendation logs for monitoring.
    """

    def __init__(self, username=None, host=None, password=None, local_port=None, remote_port=None):
        load_dotenv()

        self.username = username or os.getenv("TUNNEL_USER")
        self.host = host or os.getenv("TUNNEL_IP")
        self.password = password or os.getenv("TUNNEL_PASSWORD")
        self.local_port = local_port or os.getenv("TUNNEL_PORT")
        self.remote_port = remote_port or os.getenv("TUNNEL_PORT")

    def consume_batch(self, batch_size: int = 100, topic: str = "movielog10", group_id: str = "team10-monitor-group"):
        """
        Consume a batch of messages from Kafka.
        """
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=f"localhost:{self.local_port}",
                group_id=group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=10000
            )

            messages = []
            count = 0

            for message in consumer:
                try:
                    msg_str = message.value.decode('utf-8')
                    messages.append(msg_str)
                    count += 1

                    if count >= batch_size:
                        break
                except Exception as e:
                    logger.error(f"Error decoding message: {e}")
                    continue

            consumer.close()
            return messages

        except Exception as e:
            logger.error(f"Error consuming from Kafka: {e}")
            return []

    def consume_continuous(self, topic: str = "movielog10", group_id: str = "team10-monitor-group"):
        """
        Continuously consume messages from Kafka (generator).
        """
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=f"localhost:{self.local_port}",
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True
        )

        try:
            for message in consumer:
                try:
                    yield message.value.decode('utf-8')
                except Exception as e:
                    logger.error(f"Error decoding message: {e}")
                    continue
        finally:
            consumer.close()
