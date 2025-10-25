from homealone.data_processing.kafka_pipeline import KafkaSSHClient
import os
from dotenv import load_dotenv
from kafka.errors import NoBrokersAvailable
import re
import pandas as pd
import time
import homealone.data_processing.data_queue as dq

class DataProcessor:
    def __init__(self, kafka_client: KafkaSSHClient):
        self.kafka_client = kafka_client
        self.data_queue = dq.DataQueue()

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
        #is it recommendation request
        recommendation_pattern = r'^([^,]+),([^,]+),recommendation request ([^,]+), status (\d+), result: ([^,]+), ([^,]+)$'
        match = re.match(recommendation_pattern, data)
        if match:
            time, userid, server, status, recommendations, responsetime = match.groups()
            processed_data = {
                "type": "recommendation_request",
                "time": time,
                "userid": userid,
                "server": server,
                "status": status,
                "recommendations": recommendations,
                "responsetime": responsetime
            }
        print("Done processing data: ", data)
        return processed_data

    def run(self):
        
        for message in self.kafka_client.consume():
            processed_message = self.process_data(message)
            if processed_message:
                self.data_queue.enqueue_data(processed_message)
                
            
            
if __name__ == "__main__":
    load_dotenv()
    
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
            print("No Kafka brokers available. Please check your connection and broker settings.")
        except KeyboardInterrupt:
            print("Stopping consumption...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
         