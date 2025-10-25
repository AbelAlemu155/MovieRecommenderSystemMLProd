import pandas as pd
import os
from dotenv import load_dotenv

class DataQueue:
    def __init__(self):
        self.movie_df = pd.DataFrame(columns=["time", "userid", "movieid", "minute"])
        self.rating_df = pd.DataFrame(columns=["time", "userid", "movieid", "rating"])
        self.recommendation_df = pd.DataFrame(columns=["time", "userid", "server", "status", "recommendations", "responsetime"])
        self.movie_ids = set()
        self.user_ids = set()
        
        # queue_folder 
        load_dotenv()
        # self.queue_folder = os.getenv("DATA_QUEUE_DIR", "homealone/data_processing/data_queues/")
        self.queue_folder = "/app/data_queues/"
        
        # create folder if not exists
        if not os.path.exists(self.queue_folder):
            os.makedirs(self.queue_folder, exist_ok=True)
        
    def enqueue_data(self, data):
        
        if data["type"] == "movie" or data["type"] == "rating":
            if not self.movie_id_in_cache(data['movieid']):
                self.movie_ids_queue(data)
                
            # self.user_ids_queue(data)
        
        if data["type"] == "movie":
            self.movie_queue(data)
        elif data["type"] == "rating":
            self.rating_queue(data)
        elif data["type"] == "recommendation_request":
            self.recommendation_queue(data)
        else:
            print("Unknown data type:", data)
    
    
    def movie_id_in_cache(self, movie_id):
        cache_file = os.path.join("homealone/data_processing/cache/", 'movieids.csv')
        if not os.path.exists(cache_file):
            return False
        df = pd.read_csv(cache_file)
        return movie_id in df['movie_id'].values
    
    def movie_ids_queue(self, data):
        pass
        # self.movie_ids.add(data['movieid'])
        # if len(self.movie_ids) >= 100:
        #     timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        #     with open(f"{self.queue_folder}movieids_{timestamp}.csv", "a") as f:
        #         f.write(f"movie_id\n")
        #         for movie_id in self.movie_ids:
        #             f.write(f"{movie_id}\n")
        #     print(f"Saved {len(self.movie_ids)} movie IDs to movieids_{timestamp}.csv")
        #     self.movie_ids = set()  # reset set
        
    def user_ids_queue(self, data):
        self.user_ids.add(data['userid'])
        if len(self.user_ids) >= 50:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            with open(f"{self.queue_folder}userids_{timestamp}.csv", "a") as f:
                f.write(f"user_id\n")
                for user_id in self.user_ids:
                    f.write(f"{user_id}\n")
            print(f"Saved {len(self.user_ids)} user IDs to userids_{timestamp}.csv")
            self.user_ids = set()  # reset set
      
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