import os
import torch
import numpy as np
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod
from time import time
from datetime import datetime
from .utils.data_loading import (
    load_user_features_npz,
    load_movie_features_npz,
    load_engagements_npz,
)
from .utils.export_db import export_db
from .utils.compute_features import (
    build_user_features_from_interactions,
    build_movie_features_npz,
)
from .utils.compute_engagement import compute_engagement_from_watch_and_ratings
from .inference import StatelessModelServer
from .model import TwoTowerBPR, sample_neg_indices_uniform, gather_item_features
from .dataset import MovieDataset, create_dataloaders
from .train import train_one_epoch, eval_one_epoch

from models.base_model import Recommender

class DeepRecSysModel(Recommender):
    """
    Encapsulates the data preparation, training, and inference logic
    for the Two-Tower BPR recommender system.
    """
    def __init__(self, base_path: str = "src/models/deep_rec_sys"):
        self.base_path = base_path
        self.db_path = "src/db"
        self.feature_dir = os.path.join(base_path, "features")
        self.checkpoint_dir = os.path.join(base_path, "checkpoints")

        # File paths
        self.user_feat_npz = os.path.join(self.feature_dir, "user_features.npz")
        self.movie_feat_npz = os.path.join(self.feature_dir, "movie_features.npz")
        self.engagement_npz = os.path.join(self.feature_dir, "engagements.npz")
        self.best_ckpt_path = os.path.join(self.checkpoint_dir, "two_tower_bpr_best.pt")
        self.item_embeds_npz = os.path.join(self.feature_dir, "item_embeds.npz")
        self.movie_data_csv = os.path.join(base_path, "fake_data", "movie_data.csv")
        
        # Internal state
        self.server: Optional[StatelessModelServer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_feats: Optional[Dict[str, Any]] = None
        self.movie_feats: Optional[Dict[str, Any]] = None

    def export_db_to_csv(self) -> None:
        """Export SQLite DB to CSV files."""
        print("Exporting DB tables to CSV...")
        self.watch_csv = os.path.join(self.db_path, "watch_history.csv")
        self.ratings_csv = os.path.join(self.db_path, "ratings.csv")
        self.movies_csv = os.path.join(self.db_path, "movies.csv")
        os.makedirs(self.db_path, exist_ok=True)
        export_db()
        return 

    def clear_csv_from_db(self) -> None:
        os.remove(self.watch_csv)
        os.remove(self.ratings_csv)
        os.remove(self.movies_csv)
        print(f"Removed temporary csv files for computing features.")
        return

    def compute_features(self) -> None:
        """Compute and save user/movie features to NPZ."""
        print("Computing and saving features to NPZ...")
        os.makedirs(self.feature_dir, exist_ok=True)
        
        user_result = build_user_features_from_interactions(
            watch_csv=self.watch_csv,
            ratings_csv=self.ratings_csv,
            out_npz=self.user_feat_npz,
        )
        print(f"  User features saved to {user_result['out']} with shape {user_result['X'].shape}")

        movie_result = build_movie_features_npz(self.movies_csv, self.movie_feat_npz)
        print(f"  Movie features saved to {movie_result['out']} with shape {movie_result['X'].shape}")
    
    def compute_engagement(self) -> None:
        """Compute and save engagement scores to NPZ."""
        print("Computing and saving engagement scores to NPZ...")
        os.makedirs(self.feature_dir, exist_ok=True)

        user_ids, movie_ids, engagement = compute_engagement_from_watch_and_ratings(
            watch_csv=self.watch_csv,
            ratings_csv=self.ratings_csv,
            out_npz=self.engagement_npz,
            today=datetime.now(),
        )
        print(f"  Computed engagement for {len(engagement)} (user,movie) pairs.")
    
    def load_data(self) -> Dict[str, Any]:
        """
        Prepares all necessary NPZ files and loads them into memory.
        """
        self.export_db_to_csv()
        self.compute_features()
        self.compute_engagement()
        self.clear_csv_from_db()


        print("\nLoading features and engagements from NPZ...")
        self.user_feats = load_user_features_npz(self.user_feat_npz)
        self.movie_feats = load_movie_features_npz(self.movie_feat_npz)
        self.engagement = load_engagements_npz(self.engagement_npz)
        print("Data loaded successfully.")
        
        return {
            "users": self.user_feats,
            "movies": self.movie_feats,
            "engagement": self.engagement,
        }

    def train(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Trains the TwoTowerBPR model.
        Args: data is ignored but kept for compatibility with Recommender ABC.
        """
        if self.user_feats is None or self.movie_feats is None or self.engagement is None:
            print("Feature data not found. Running load_data()...")
            self.load_data()
            
        print("\nStarting model training (TwoTowerBPR)...")
        
        # Data Setup
        dataset = MovieDataset(self.user_feats, self.movie_feats, self.engagement)
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=kwargs.get("batch_size", 256),
            val_ratio=kwargs.get("val_ratio", 0.1),
            test_ratio=kwargs.get("test_ratio", 0.1),
            seed=kwargs.get("seed", 42),
        )

        # Model/Optimizer Setup
        users_dim = self.user_feats["X"].shape[1]
        items_dim = self.movie_feats["X"].shape[1]
        d_embed = kwargs.get("d_embed", 128)
        
        model = TwoTowerBPR(du=users_dim, di=items_dim, d_embed=d_embed).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)
        all_item_feats = torch.from_numpy(self.movie_feats["X"]).to(self.device)
        K = kwargs.get("K", 20)
        EPOCHS = kwargs.get("epochs", 30)

        # Helper function from train.py
        def precompute_negatives(loader, num_items, K, device):
            neg_cache = []
            for batch in loader:
                *rest, pos_item_idx = batch
                pos_item_idx = pos_item_idx.to(device)
                neg_idx = sample_neg_indices_uniform(num_items, pos_item_idx, K, device)
                neg_cache.append(neg_idx)
            return neg_cache
            
        VAL_NEG = precompute_negatives(val_loader, all_item_feats.shape[0], K=K, device=self.device)
        
        # Training Loop
        best_val = float("inf")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            t0 = time()
            train_loss = train_one_epoch(model, train_loader, opt, all_item_feats, self.device, K=K)
            val_loss = eval_one_epoch(model, val_loader, all_item_feats, self.device, K=K, cached_negs=VAL_NEG)
            dt = time() - t0

            print(f"[Epoch {epoch}/{EPOCHS}] train_bpr={train_loss:.4f}  val_bpr={val_loss:.4f}  ({dt:.1f}s)")

            sch.step()

            if val_loss < best_val:
                best_val = val_loss
                print(f"  New best val bpr, saving model to {self.best_ckpt_path}")
                torch.save(model.state_dict(), self.best_ckpt_path)

        print(f"Training finished. Best Val BPR: {best_val:.4f}")

    def _setup_inference_server(self) -> None:
        """Initializes and configures the StatelessModelServer."""
        if self.server is not None:
            return

        print("\nSetting up inference server...")
        
        # Initialize server with feature paths
        self.server = StatelessModelServer(
            device=self.device,
            user_features_npz=self.user_feat_npz,
            movie_features_npz=self.movie_feat_npz,
        )

        # Load the trained model checkpoint
        self.server.load_model(self.best_ckpt_path)

        # Precompute Item Embeddings (and save them)
        self.server.precompute_item_embeddings(save_npz=self.item_embeds_npz)
        
        # Load the precomputed Item Embeddings for fast serving
        self.server.load_item_embeddings(self.item_embeds_npz)
        print("Inference server ready.")


    def recommend(self, user_id: Any, top_k: int=20) -> List[int]:
        """
        Calls the inference server to get recommendations.
        
        Returns: A list of recommended movie IDs (integers/Any).
        """
        print(f"\nGetting recommendations for user_id={user_id} (top_k={top_k})...")
        
        if self.server is None:
            self._setup_inference_server()
        
        t0 = time()
        recs = self.server.get_recommendations(user_id=user_id, top_k=top_k)
        dt = time() - t0
        print(f"Time taken: {dt*1000} ms")
        return recs["item_ids"].tolist()


if __name__ == "__main__":
    recommender = DeepRecSysModel(base_path="src/models/deep_rec_sys")

    # Load Data
    recommender.watch_csv = "src/db/watch_history.csv"
    recommender.ratings_csv = "src/db/ratings.csv"
    recommender.movies_csv = "src/db/movies.csv"
    recommender.load_data()

    # Train Model
    recommender.train(epochs=10, batch_size=512, K=40)

    # Recommend
    recs = recommender.recommend(user_id=21585, top_k=20)
    print(recs)
    
    # Unknown user example
    rec2 = recommender.recommend(user_id="unknown_user", top_k=20)
    print(rec2)