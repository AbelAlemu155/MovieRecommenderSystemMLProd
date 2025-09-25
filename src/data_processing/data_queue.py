import pandas as pd
import os
from dotenv import load_dotenv

class DataQueue:
    def __init__(self):
        self.movie_df = pd.DataFrame(columns=["time", "userid", "movieid", "minute"])
        self.rating_df = pd.DataFrame(columns=["time", "userid", "movieid", "rating"])
        self.recommendation_df = pd.DataFrame(columns=["time", "userid", "server", "status", "recommendations", "responsetime"])
        
        # queue_folder 
        load_dotenv()
        self.queue_folder = os.getenv("DATA_QUEUE_DIR", "src/data_processing/data_queues/")
        
        # create folder if not exists
        if not os.path.exists(self.queue_folder):
            os.makedirs(self.queue_folder, exist_ok=True)
        
    def enqueue_data(self, data):
        if data["type"] == "movie":
            self.movie_queue(data)
        elif data["type"] == "rating":
            self.rating_queue(data)
        elif data["type"] == "recommendation_request":
            self.recommendation_queue(data)
        else:
            print("Unknown data type:", data)
        
    def movie_queue(self, data):
        self.movie_df = pd.concat([self.movie_df, pd.DataFrame([data])], ignore_index=True)
        # every 10_000 movies save to csv
        if len(self.movie_df) >= 10_000:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.movie_df.to_csv(f"{self.queue_folder}movies_{timestamp}.csv", index=False)
            print(f"Saved {len(self.movie_df)} movies to movies_{timestamp}.csv")
            self.movie_df = pd.DataFrame(columns=["time", "userid", "movieid", "minute"])  # reset dataframe

    def rating_queue(self, data):
        self.rating_df = pd.concat([self.rating_df, pd.DataFrame([data])], ignore_index=True)
        # every 100 ratings save to csv
        if len(self.rating_df) >= 100:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.rating_df.to_csv(f"{self.queue_folder}ratings_{timestamp}.csv", index=False)
            print(f"Saved {len(self.rating_df)} ratings to ratings_{timestamp}.csv")
            self.rating_df = pd.DataFrame(columns=["time", "userid", "movieid", "rating"])  # reset dataframe

    def recommendation_queue(self, data):
        self.recommendation_df = pd.concat([self.recommendation_df, pd.DataFrame([data])], ignore_index=True)
        # every 10_000 recommendations save to csv
        if len(self.recommendation_df) >= 10:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            self.recommendation_df.to_csv(f"{self.queue_folder}recommendations_{timestamp}.csv", index=False)
            print(f"Saved {len(self.recommendation_df)} recommendations to recommendations_{timestamp}.csv")
            self.recommendation_df = pd.DataFrame(columns=["time", "userid", "server", "status", "recommendations", "responsetime"])  # reset dataframe