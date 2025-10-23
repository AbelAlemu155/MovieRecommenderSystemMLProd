import os
from dotenv import load_dotenv
from kafka.errors import NoBrokersAvailable
import re
import pandas as pd
import time

import homealone.data_processing.data_queue as dq
from homealone.data_ingestion.kafka_pipeline import KafkaSSHClient
# import schema validation manager and schemas
import logging
from homealone.data_processing.schema_validation.validator import ValidationManager
from homealone.data_processing.schema_validation.schemas import MovieEvent, RatingEvent, RecommendationEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, kafka_client: KafkaSSHClient):
        self.kafka_client = kafka_client
        self.data_queue = dq.DataQueue()
        self.validator = ValidationManager()
        logger.info("DataProcessor initialized with ValidationManager")

    def process_data(self, data):
        processed_data = None
        # movie log line matching
        movie_pattern = r'^([^,]+),([^,]+),GET /data/m/([^/]+)/([^\.]+)\.mpg$'
        match = re.match(movie_pattern, data)
        if match:
            time, userid, movieid, minute = match.groups()
            processed_data = {
                "type": "movie",
                "time": time,
                "userid": userid,
                "movieid": movieid,
                "minute": minute
            }

        # is it a rating
        rating_pattern = r'^([^,]+),([^,]+),GET /rate/([^=]+)=([^ ]+)$'
        match = re.match(rating_pattern, data)
        if match:
            time, userid, movieid, rating = match.groups()
            processed_data = {
                "type": "rating",
                "time": time,
                "userid": userid,
                "movieid": movieid,
                "rating": rating
            }

        # is it recommendation request
        recommendation_pattern = r'^([^,]+),([^,]+),recommendation request ([^,]+), status (\d+), result: (.+), (\d+) ms$'
        match = re.match(recommendation_pattern, data)
        if match:
            time, userid, server, status, recommendations, responsetime = match.groups()
            # Clean up recommendations string
            responsetime_str = responsetime.strip().replace(' ms', '')
            processed_data = {
                "type": "recommendation_request",
                "time": time,
                "userid": userid,
                "server": server,
                "status": status,
                "recommendations": recommendations,
                "responsetime": float(responsetime_str)
            }
        print("Done processing data: ", data)
        return processed_data

    def validate_data(self, data):
        """
        Validate data against appropriate schema.
        """
        data_type = data.get("type")

        # Select schema based on data type
        if data_type == "movie":
            schema = MovieEvent
        elif data_type == "rating":
            schema = RatingEvent
        elif data_type == "recommendation_request":
            schema = RecommendationEvent
        else:
            logger.error(f"Unknown data type: {data_type}")
            return False

        is_valid, validated_data, error_msg = self.validator.validate_record(data, schema)
        if not is_valid:
            logger.error(f"Validation errors for data {data}: {error_msg}")
        else:
            logger.info(
                f"Data validated successfully: {data}, for event type: {data_type}")
        return is_valid

    def get_validation_stats(self):
        """
        Get validation statistics from the validator.
        """
        return self.validator.get_stats()

    def run(self):
        """
        Main loop consuming Kafka messages, validating, and queueing them.
        """
        logger.info("Starting DataProcessor main loop...")
        for message in self.kafka_client.consume():
            # process message in kafka
            processed_message = self.process_data(message)
            # validate processed message
            if processed_message is not None and self.validate_data(processed_message):
                # enqueue valid data
                self.data_queue.enqueue_data(processed_message)
                logger.debug(f"Message enqueued: {processed_message['type']}")
            else:
                logger.info(f"Skipping invalid message: {message}, with type: {processed_message['type'] if processed_message else 'unknown'}")


if __name__ == "__main__":
    load_dotenv()

    logger.info("Initializing DataProcessor with Kafka SSH Client...")

    client = KafkaSSHClient(
        username=os.getenv("TUNNEL_USER"),
        host=os.getenv("TUNNEL_IP"),
        password=os.getenv("TUNNEL_PASSWORD"),
        local_port=os.getenv("TUNNEL_PORT"),
        remote_port=os.getenv("TUNNEL_PORT")
    )

    # Consume from topic "movielog10"

    while True:
        try:
            data_processor = DataProcessor(kafka_client=client)
            data_processor.run()
        except NoBrokersAvailable:
            print(
                "No Kafka brokers available. Please check your connection and broker settings.")
        except KeyboardInterrupt:
            print("Stopping consumption...")
            # print the validation statistics
            if 'data_processor' in locals():
                stats = data_processor.get_validation_stats()
                logger.info(f"Final validation statistics: {stats}")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
