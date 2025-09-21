from data_ingestion.kafka_pipeline import KafkaSSHClient
import os
from dotenv import load_dotenv
from kafka.errors import NoBrokersAvailable

class DataProcessor:
    def __init__(self, kafka_client: KafkaSSHClient):
        self.kafka_client = kafka_client

    def process_data(self, data):
        # Implement your data processing logic here
        processed_data = data.upper()  # Example processing
        return processed_data

    def run(self):
        for message in self.kafka_client.consume_messages():
            processed_message = self.process_data(message)
            print(f"Processed Message: {processed_message}")
            # Further actions can be taken with the processed message
            
            
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
            count = 0
            for message in client.consume("movielog10"):
                print(message)
                if count >= 10:
                    break
        except NoBrokersAvailable:
            print("No Kafka brokers available. Please check your connection and broker settings.")
        except KeyboardInterrupt:
            print("Stopping consumption...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
         