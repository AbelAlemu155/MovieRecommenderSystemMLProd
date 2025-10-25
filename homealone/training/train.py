import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, NMF  # Import NMF
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import dump      # Import the dump module for saving
import os
import pickle
import time
import  logging
from logging.handlers import RotatingFileHandler    

logging.basicConfig(level=logging.INFO)

# --- 1. Load Data from SQLite ---
db_path = 'db/home_alone.db'  
table_name = 'ratings'
trainset = None
testset = None        

class Trainning:
    def __init__(self):
        self.db_path = 'db/home_alone.db'  
        self.table_name = 'ratings'
        self.trainset, self.testset = None, None
        self.conn = sqlite3.connect(self.db_path)
        self.model_path = '/app/models/'

        self.load_data()

    def load_data(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            query = f"SELECT user_id, movie_id, rating FROM {self.table_name}"
            df = pd.read_sql_query(query, self.conn)
        finally:
            if self.conn:
                self.conn.close()

        reader = Reader(rating_scale=(1, 5)) 
        data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

        # We build the full trainset one time for training all models
        self.trainset = data.build_full_trainset()
        # We also build a testset for evaluation
        self.testset = self.trainset.build_anti_testset() # trainset, testset = train_test_split(data, test_size=0.25)
        


    def save_trainset(self):
        file_path = os.path.join(self.model_path, 'trainset.dump')
        print(f"Saving trainset to {file_path} file type {type(self.trainset)}")
        with open(file_path, 'wb') as f:
            pickle.dump(self.trainset, f)
        print("Trainset saved.")


    def train_svd(self):
        # SVD (Matrix Factorization) ---
        print("\nTraining SVD model...")
        model_svd = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
        model_svd.fit(self.trainset)
        print("SVD model trained.")
        return model_svd

    def train_knn(self):
        # KNNBasic (User-Based) ---
        print("\nTraining KNNBasic model...")
        model_knn = KNNBasic(sim_options={'user_based': True}) 
        model_knn.fit(self.trainset)
        print("KNNBasic model trained.")
        return model_knn

    def train_nmf(self):
        # NMF (Matrix Factorization) ---
        print("\nTraining NMF model...")
        model_nmf = NMF(n_factors=20, n_epochs=20, random_state=42) # NMF is another MF technique
        model_nmf.fit(self.trainset)
        print("NMF model trained.")
        return model_nmf


    def save_model(self, model, file_path):
        print(f"Saving model to {file_path}...")
        dump.dump(file_path, algo=model)
        print("Model saved.")

    def evaluate_model(self, model, testset):
        print("Evaluating model...")
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        return rmse



if __name__ == "__main__":
    
    trainer = Trainning()
    
    while True:
        # add logging here
        logging.info("Starting training cycle...")
        print("Starting training cycle...")            
        model_svd = trainer.train_svd()
        model_knn = trainer.train_knn()
        model_nmf = trainer.train_nmf()

        svd_file_path = '/app/models/svd_model.dump'
        knn_file_path = '/app/models/knn_model.dump'
        nmf_file_path = '/app/models/nmf_model.dump'
        trainset_file_path = '/app/models/trainset.dump'

        trainer.save_model(model_svd, svd_file_path)
        trainer.save_model(model_knn, knn_file_path)
        trainer.save_model(model_nmf, nmf_file_path)
        trainer.save_trainset()
        print("All models and trainset saved.")
        # wait for some time before retraining
        time.sleep(60*60*12)  # Train every 12 hours
        trainer.load_data()
        
