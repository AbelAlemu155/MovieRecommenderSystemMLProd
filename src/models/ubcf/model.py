import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle
from dotenv import load_dotenv 
import os
from sqlalchemy import create_engine
from models.base_model import Recommender
from pathlib import Path

class UBCFRecommender(Recommender):
    def __init__(self, model_path: str = None):
        load_dotenv()
        
        

        # Assuming PROJECT_ROOT is defined as the directory containing 'src'
        relative_path = os.environ.get('UBCF_MODEL_PATH', 'src/models/ubcf/ubcf_model.pkl') # Use a default if not set
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        # Combine the paths to get the absolute path
        full_path = PROJECT_ROOT / relative_path
        
        self.model_path = full_path
        self.utility_matrix = None
        self.user_similarity_df = None

    def get_ratings_data(self) -> pd.DataFrame:
        """
        Fetches ratings data from the database.
        """
        path_to_db = os.getenv("DB_PATH", "src/db/home_alone.db")
        engine = create_engine("sqlite:///" + path_to_db)
        ratings_df = pd.read_sql_query("SELECT * FROM ratings where rating > 3", engine)
        return ratings_df

    def train(self, df: pd.DataFrame = None):
        """
        Trains the UBCF model and saves utility matrix and user similarity matrix to a file.
        """
        
        # fetch ratings data
        if df is None:
            df = self.get_ratings_data()
        
        print("--- Training UBCF Model ---")
        utility_matrix = df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        )
        print(f"Utility Matrix Shape: {utility_matrix.shape}")
        user_ratings_mean = utility_matrix.mean(axis=1)
        ratings_normalized = utility_matrix.subtract(user_ratings_mean, axis=0)
        ratings_filled = ratings_normalized.fillna(0)
        user_similarity_matrix = 1 - pairwise_distances(ratings_filled, metric='cosine')
        user_similarity_df = pd.DataFrame(
            user_similarity_matrix,
            index=utility_matrix.index,
            columns=utility_matrix.index
        )
        print("User-User Similarity Matrix computed successfully.")
        # Save model
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "utility_matrix": utility_matrix,
                "user_similarity_df": user_similarity_df
            }, f)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """
        Loads the trained model from file.
        """
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
            self.utility_matrix = model["utility_matrix"]
            self.user_similarity_df = model["user_similarity_df"]

    def recommend(self, user_id: int, k: int = 5, n_recommendations: int = 20) -> pd.DataFrame:
        """
        Generates movie recommendations for a specific user using UBCF.
        """
        if self.utility_matrix is None or self.user_similarity_df is None:
            self.load()
      
        print(f"\n--- Generating Recommendations for User {user_id} ---")
      
        utility_matrix = self.utility_matrix
        user_similarity_df = self.user_similarity_df
      
        if user_id not in utility_matrix.index:
            return pd.DataFrame({'message': [f"User ID {user_id} not found in the data."]})
        
        similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
        knn = similar_users.head(k)
        
        if knn.empty:
            return pd.DataFrame({'message': ["No similar users found."]})
        
        neighbor_ratings = utility_matrix.loc[knn.index]
        neighbor_weights = knn.values
        
        #  movies already rated by the user
        user_rated_movies = utility_matrix.loc[user_id].dropna().index
        
        # recommend only movies not yet rated by the user
        movies_to_predict = neighbor_ratings.columns.difference(user_rated_movies)
        
        
        neighbor_ratings_predict = neighbor_ratings[movies_to_predict].copy()
        weighted_ratings = neighbor_ratings_predict.fillna(0).T.dot(neighbor_weights)
        sum_of_weights = neighbor_ratings_predict.notna().T.dot(np.abs(neighbor_weights))
        predicted_ratings = weighted_ratings / sum_of_weights
        predictions_df = predicted_ratings.dropna().sort_values(ascending=False).head(n_recommendations)
        predictions_df = pd.DataFrame(predictions_df, columns=['predicted_rating'])
        predictions_df.index.name = 'movie_id'
        print(f"Generated {len(predictions_df)} recommendations.")
        return predictions_df.index

# --- Example Usage ---

if __name__ == "__main__":
    
    # Initialize and train the recommender
    recommender = UBCFRecommender()
    recommender.train()
    recommendations = recommender.recommend(user_id=1728, k=5, n_recommendations=20)
    print(recommendations)
