import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from joblib import dump, load # Optimized serialization for NumPy arrays
from dotenv import load_dotenv 
import os
from sqlalchemy import create_engine
from models.base_model import Recommender
from pathlib import Path



class UBCFRecommender(Recommender):
    def __init__(self, model_path: str = None, top_k: int = 50):
        load_dotenv()
        
        self.top_k = top_k
        self.user_to_idx = None # User ID to internal index mapping
        self.idx_to_user = None # Internal index to User ID mapping
        
        
        self.utility_matrix = None # Full utility matrix (users x items)
        self.similarity_matrix = None # Full user-user similarity matrix
        
        # Optimized Model Components
        self.user_means = None
        self.top_k_neighbors = None 
        
        relative_path = os.environ.get('UBCF_MODEL_PATH', 'src/models/ubcf/ubcf_model_optimized.joblib')
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = PROJECT_ROOT / relative_path
        
        self.path_to_csv = "src/data_processing/data_queues/ratings.csv"
    
    def get_ratings_data(self, from_csv=False) -> pd.DataFrame:
        """
        Fetches ratings data from the database.
        """
        if from_csv:
            ratings_df = pd.read_csv(self.path_to_csv)
        else:
            path_to_db = os.getenv("DB_PATH", "src/db/home_alone.db")
            engine = create_engine("sqlite:///" + path_to_db)
            # Fetching all ratings for proper evaluation later
            ratings_df = pd.read_sql_query("SELECT * FROM ratings", engine) 
        
        return ratings_df
   
    def _handle_cold_start(self, n_recommendations: int) -> pd.DataFrame:
        """
        Handles the 'cold start' problem for new users or users with no neighbors.
        Returns a DataFrame of randomly selected movies with a default high prediction score.
        """
        if self.utility_matrix is None:
            self.load() # Ensure utility matrix is loaded to get movie IDs

        # Get all unique movie IDs
        all_movie_ids = self.utility_matrix.columns
        
        # Handle case where n_recommendations is greater than total movies
        if n_recommendations >= len(all_movie_ids):
            random_movies = all_movie_ids.tolist()
        else:
            # Randomly select N movie IDs
            random_movies = np.random.choice(
                all_movie_ids, 
                size=n_recommendations, 
                replace=False
            )
        
        # Create a DataFrame with a default prediction score (e.g., 5.0)
        # This signals high confidence for the initial random recommendations.
        predictions_df = pd.DataFrame({
            'predicted_rating': [5.0] * len(random_movies)
        }, index=random_movies)
        
        predictions_df.index.name = 'movie_id'
        print(f"COLD START: Generated {len(predictions_df)} random recommendations.")
        return predictions_df
    
    # ----------------------------------------------------------------------
    # Modified get_recommendations to use the cold start handler
    # ----------------------------------------------------------------------

    def get_recommendations(self, user_id: int, k: int = 5, n_recommendations: int = 20) -> pd.DataFrame:
        """
        Generates movie recommendations for a specific user using  (Top-K Lookup).
        Includes cold-start handling.
        """
        if self.utility_matrix is None or self.top_k_neighbors is None:
            self.load()
        
        print(f"\n--- Generating Recommendations for User {user_id} ---")
        
        
        if user_id not in self.top_k_neighbors:
            print("User not in trained set. Activating Cold Start handler.")
            return self._handle_cold_start(n_recommendations)
            
        
        # knn is a NumPy array of (neighbor_id, similarity_score)
        knn_array = self.top_k_neighbors.get(user_id) 

        # Check if the user has any neighbors (knn_array is an empty array if no neighbors)
        if len(knn_array) == 0:
            print("User has no neighbors. Activating Cold Start handler.")
            return self._handle_cold_start(n_recommendations)
        
        # Select the top 'k' from the pre-computed 'top_k' neighbors
        knn_subset = knn_array[:k]
        neighbor_ids = knn_subset[:, 0].astype(int)
        neighbor_weights = knn_subset[:, 1].astype(float)
        
        
        
        #prediction calculation
        utility_matrix = self.utility_matrix
        neighbor_ratings = utility_matrix.loc[neighbor_ids]
        
        user_rated_movies = utility_matrix.loc[user_id].dropna().index
        movies_to_predict = neighbor_ratings.columns.difference(user_rated_movies)
        
        # Use vectorized operations
        neighbor_ratings_predict = neighbor_ratings[movies_to_predict].copy()
        
        # Normalize neighbor ratings by subtracting their mean (centered ratings)
        neighbor_means = self.user_means.loc[neighbor_ids]
        centered_ratings = neighbor_ratings_predict.subtract(neighbor_means, axis=0).fillna(0)
        
        # Prediction formula: sum(sim * centered_rating) / sum(abs(sim)) + user_mean
        weighted_centered_ratings = centered_ratings.T.dot(neighbor_weights)
        sum_of_abs_weights = np.dot(neighbor_ratings_predict.notna().values.T, np.abs(neighbor_weights))
        
        # Calculate predicted rating (User Mean + Weighted Sum of Centered Ratings)
        predicted_ratings_centered = np.divide(
            weighted_centered_ratings, 
            sum_of_abs_weights, 
            out=np.zeros_like(weighted_centered_ratings), 
            where=sum_of_abs_weights!=0
        )
        
        # Add back active user's mean to "de-normalize" the prediction
        user_mean = self.user_means.loc[user_id]
        predicted_ratings = predicted_ratings_centered + user_mean
        
        predictions_series = pd.Series(predicted_ratings, index=movies_to_predict)
        predictions_df = predictions_series.dropna().sort_values(ascending=False).head(n_recommendations)
        predictions_df = pd.DataFrame(predictions_df, columns=['predicted_rating'])
        predictions_df.index.name = 'movie_id'
        
        print(f"Generated {len(predictions_df)} recommendations via UBCF.")
        return predictions_df


    def train(self, df: pd.DataFrame = None):
        """
        Trains the UBCF model, pre-computes Top-K neighbors, and saves components using joblib.
        (Original implementation)
        """
        
        if df is None:
            df = self.get_ratings_data(from_csv=True)
        
        print("--- Training UBCF Model (Optimized) ---")
        utility_matrix = df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        )
        print(f"Utility Matrix Shape: {utility_matrix.shape}")
        
        # store User Mapping
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(utility_matrix.index)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        
        # compute Mean and Normalized Ratings
        self.user_means = utility_matrix.mean(axis=1)
        ratings_normalized = utility_matrix.subtract(self.user_means, axis=0)
        ratings_filled = ratings_normalized.fillna(0)
        
        # compute Similarity Matrix 
        user_similarity_matrix = 1 - pairwise_distances(ratings_filled, metric='cosine')

        # Optimization: Pre-calculate and store only Top-K Neighbors
        print(f"Pre-calculating Top-{self.top_k} neighbors...")
        self.top_k_neighbors = {}
        for user_idx, user_id in enumerate(utility_matrix.index):
            # Sort similarities, drop self-similarity (score of 1.0), and get Top-K
            similarities = user_similarity_matrix[user_idx]
            # Create pairs of (user_id, similarity)
            sim_pairs = [(self.idx_to_user[i], s) for i, s in enumerate(similarities)]
            # Convert to DataFrame for easy sort, filter out self
            sim_df = pd.DataFrame(sim_pairs, columns=['neighbor_id', 'similarity'])
            sim_df = sim_df[sim_df['neighbor_id'] != user_id]
            sim_df = sim_df.sort_values(by='similarity', ascending=False).head(self.top_k)
            
            # Store as a NumPy array for fast access
            self.top_k_neighbors[user_id] = sim_df.to_numpy()
        
        # 5. Save Model Components with Joblib
        model_components = {
            "utility_matrix_full": utility_matrix, # Kept for evaluation, normally would be smaller
            "user_means": self.user_means,
            "top_k_neighbors": self.top_k_neighbors
        }
        
        # Using joblib for efficient large NumPy array serialization
        dump(model_components, self.model_path)
        print(f"Model saved to {self.model_path} using joblib.")

    def load(self):
        """
        Loads the trained model components from file using joblib.
        (Original implementation)
        """
        # Using joblib.load for optimized deserialization
        model = load(self.model_path)
        self.utility_matrix = model["utility_matrix_full"] # Loaded for prediction and evaluation
        self.user_means = model["user_means"]
        self.top_k_neighbors = model["top_k_neighbors"]
        
    def recommend(self, user_id: int, k: int = 5, n_recommendations: int = 20) -> str:

        recommendations = self.get_recommendations(user_id=user_id, k=k, n_recommendations=n_recommendations)
        # # return comma-separated movie IDs
        if 'movie_id' in recommendations.columns:
            movie_ids = recommendations['movie_id'].astype(str).tolist()
        else:
            movie_ids = recommendations.index.astype(str).tolist()

        return movie_ids

    def evaluate_model(self, test_df: pd.DataFrame, top_n: int = 10, threshold: float = 4.0):
        """
        Evaluates the model using Precision, Recall, and Accuracy.
        A 'relevant' item is defined as a rating >= threshold.
        (Original implementation)
        """
        if self.utility_matrix is None or self.top_k_neighbors is None:
            self.load()

        all_precision, all_recall, all_accuracy = [], [], []
        
        # Convert test_df to a dictionary of {user_id: set(relevant_movie_ids)}
        relevant_items = test_df[test_df['rating'] >= threshold].groupby('user_id')['movie_id'].apply(set).to_dict()
        
        for user_id in relevant_items.keys():
            if user_id not in self.top_k_neighbors:
                continue

            # 1. Get predicted top-N recommendations
            try:
                # Call get_recommendations, which now includes the cold-start handler
                predictions_df = self.get_recommendations(user_id, k=self.top_k, n_recommendations=top_n)
                predicted_set = set(predictions_df.index.values)
            except Exception:
                continue # Skip if prediction fails

            actual_relevant_set = relevant_items[user_id]
            
            # Intersection: items that are both predicted AND relevant
            hits = predicted_set.intersection(actual_relevant_set)
            
            # --- Metrics Calculation ---
            
            # Precision: What proportion of recommended items were relevant?
            precision = len(hits) / top_n if top_n > 0 else 0
            
            # Recall: What proportion of relevant items were recommended?
            recall = len(hits) / len(actual_relevant_set) if len(actual_relevant_set) > 0 else 0
            
            # Accuracy (simple hit rate over the top-N set)
            accuracy_at_n = len(hits) / top_n if top_n > 0 else 0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_accuracy.append(accuracy_at_n)


        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)
        avg_accuracy = np.mean(all_accuracy)
        
        print("\n--- Model Evaluation ---")
        print(f"Items considered 'Relevant' if rating >= {threshold}")
        print(f"Average Precision@{top_n}: {avg_precision:.4f}")
        print(f"Average Recall@{top_n}: {avg_recall:.4f}")
        print(f"Average Hit-Rate (Accuracy)@{top_n}: {avg_accuracy:.4f}")
        
        return avg_precision, avg_recall, avg_accuracy



if __name__ == "__main__":
    
    recommender = UBCFRecommender()
    # recommender.train()
    
    # ---------------------------------
    # 1. Recommendation Test
    # ---------------------------------
    recommendations = recommender.recommend(user_id=1728, k=50, n_recommendations=20)
    print("\nRecommendations")
    print(recommendations)
    
    # ---------------------------------
    # 2. Evaluation Test
    # ---------------------------------
    # Use 20% of the data for evaluation (conceptually)
    # all_data = recommender.get_ratings_data(from_csv=True)
    # Simple split concept (not robust for time-series evaluation)
    # test_df = all_data.sample(frac=0.2, random_state=42) 
    
    # recommender.evaluate_model(test_df, top_n=10, threshold=4.0)

    
    