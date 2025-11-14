import sqlite3
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise import accuracy
from surprise import dump
import os
import pickle
import time
import logging
from datetime import datetime, timedelta

import sys
sys.path.insert(0, '/app')
from homealone.provenance.model_provenance import ModelProvenance
from homealone.provenance.data_provenance import DataProvenance



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Training:
    def __init__(self):
        self.db_path = 'db/home_alone.db'
        self.table_name = 'ratings'
        self.trainset = None
        self.conn = None
        self.model_path = '/app/models/'
        
        # Evaluation configuration - percentage-based
        self.test_percentage = 0.30
        self.retrain_threshold = 0.05
        self.training_cutoff = None

        # setup provenance tracking
        self.model_prov = ModelProvenance()
        self.data_prov = DataProvenance()

        self.model_prov.initialize()
        self.data_prov.initialize()
        
        self._connect()
        self._create_metadata_table()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
    
    def _create_metadata_table(self):
        """Create table to track training runs"""
        cursor = self.conn.cursor()
        
    
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_timestamp DATETIME,
                training_cutoff DATETIME,
                n_train_samples INTEGER,
                n_test_samples INTEGER,
                total_samples INTEGER,
                train_date_range_start TEXT,
                train_date_range_end TEXT,
                test_date_range_start TEXT,
                test_date_range_end TEXT,
                svd_rmse REAL,
                knn_rmse REAL,
                nmf_rmse REAL,
                best_model TEXT
            )
        ''')
        self.conn.commit()
        logger.info("Training metadata table created")

    def get_last_training_sample_count(self):
        """Get the number of samples used in last training run"""
        query = """
            SELECT total_samples
            FROM training_metadata
            ORDER BY training_timestamp DESC
            LIMIT 1
        """
        result = pd.read_sql_query(query, self.conn)
        
        if len(result) == 0:
            return 0  # First training run
        
        return result.iloc[0]['total_samples']
    
    def should_retrain(self):
        """
        Determine if retraining is necessary based on new data volume.
        Retrain only if we have at least 10% more data than last training.
        
        Returns:
            should_train (bool): Whether to retrain
            current_count (int): Current total data count
            new_data_count (int): Number of new samples since last training
        """
        # Get current total data count
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        current_count = pd.read_sql_query(query, self.conn).iloc[0]['count']
        
        # Get last training count
        last_count = self.get_last_training_sample_count()
        
        if last_count == 0:
            logger.info("First training run - will train")
            return True, current_count, 0
        
        # Calculate new data percentage
        new_data_count = current_count - last_count
        new_data_percentage = (new_data_count / last_count) * 100 if last_count > 0 else 100
        
        logger.info(f"Data Status:")
        logger.info(f"  Current: {current_count} ratings")
        logger.info(f"  Last training: {last_count} ratings")
        logger.info(f"  New data: {new_data_count} ratings ({new_data_percentage:.1f}%)")
        
        # Retrain if we have at least 10% more data
        threshold_percentage = self.retrain_threshold * 100
        if new_data_percentage >= threshold_percentage:
            logger.info(f"New data exceeds threshold ({new_data_percentage:.1f}% >= {threshold_percentage}%)")
            return True, current_count, new_data_count
        else:
            logger.info(f"Insufficient new data ({new_data_percentage:.1f}% < {threshold_percentage}%)")
            logger.info(f"Need {int(last_count * self.retrain_threshold - new_data_count)} more ratings")
            return False, current_count, new_data_count
    
    def export_data_to_parquet(self):
        """Export current training data to Parquet for versioning"""
        logger.info("Exporting training data to Parquet...")
        
        query = f"""
            SELECT user_id, movie_id, rating, rated_at
            FROM {self.table_name}
            ORDER BY rated_at ASC
        """
        df = pd.read_sql_query(query, self.conn)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_dir = 'homealone/data/raw'
        os.makedirs(data_dir, exist_ok=True)
        
        parquet_file = f'{data_dir}/ratings_{timestamp}.parquet'
        
        # Save as Parquet
        df.to_parquet(
            parquet_file,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Create symlink to latest
        latest_link = f'{data_dir}/ratings_latest.parquet'
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.basename(parquet_file), latest_link)
        
        logger.info(f"Data exported to: {parquet_file}")
        logger.info(f"Size: {os.path.getsize(parquet_file) / 1024:.1f} KB")
        
        return parquet_file, timestamp

    def get_percentage_based_split(self):
        """
        Split data by percentage (temporal ordering).
        - Train on: First 90% of data (oldest)
        - Test on: Last 10% of data (most recent)
        
        Returns:
            train_df, test_df, cutoff_timestamp
        """
        logger.info(f"Loading data with {self.test_percentage*100:.0f}% test split...")
        
        # Get all ratings ordered by time
        query = f"""
            SELECT user_id, movie_id, rating, rated_at
            FROM {self.table_name}
            ORDER BY rated_at ASC
        """
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) == 0:
            logger.error("No data found in ratings table")
            return None, None, None
        
        original_count = len(df)
        logger.info(f"Loaded {original_count} ratings from database")
        
        # Parse timestamps and handle errors
        df['rated_at'] = pd.to_datetime(df['rated_at'], format='ISO8601', errors='coerce')
        
        # Handle invalid timestamps
        invalid_count = df['rated_at'].isna().sum()
        if invalid_count > 0:
            logger.warning(f"Skipping {invalid_count} invalid timestamps ({invalid_count/original_count*100:.2f}%)")
        
        df = df.dropna(subset=['rated_at']).copy()
        
        if len(df) < 100:
            logger.error("Insufficient valid data after cleaning")
            return None, None, None
        
        # Calculate split index
        split_idx = int(len(df) * (1 - self.test_percentage))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Get cutoff timestamp
        cutoff_timestamp = train_df['rated_at'].max()
        
        # Filter test set for warm-start
        train_users = set(train_df['user_id'].unique())
        train_movies = set(train_df['movie_id'].unique())
        
        test_df_original_size = len(test_df)
        test_df = test_df[
            test_df['user_id'].isin(train_users) &
            test_df['movie_id'].isin(train_movies)
        ].copy()
        
        filtered_count = test_df_original_size - len(test_df)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} cold-start test samples")
        
        logger.info("=" * 60)
        logger.info(f"Training set: {len(train_df)} ratings ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {train_df['rated_at'].min()} to {train_df['rated_at'].max()}")
        logger.info(f"Test set: {len(test_df)} ratings ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {test_df['rated_at'].min()} to {test_df['rated_at'].max()}")
        logger.info(f"Training cutoff: {cutoff_timestamp}")
        logger.info(f"Unique users: {len(train_users)}, Unique movies: {len(train_movies)}")
        logger.info("=" * 60)
        
        return train_df, test_df, cutoff_timestamp
    
    def load_data_for_training(self):
        """
        Load data and create trainset for model training.
        Uses percentage-based split: 90% train, 10% test.
        """
        train_df, test_df, self.training_cutoff = self.get_percentage_based_split()
        
        if train_df is None or len(train_df) < 100:
            logger.error("Insufficient training data")
            return None, None
        
        # Convert to Surprise format for training
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(
            train_df[['user_id', 'movie_id', 'rating']], 
            reader
        )
        
        self.trainset = train_data.build_full_trainset()
        logger.info("Training data loaded and trainset built")
        
        return train_df, test_df
    
    def save_trainset(self):
        """Save trainset for inference service"""
        file_path = os.path.join(self.model_path, 'trainset.dump')
        logger.info(f"Saving trainset to {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(self.trainset, f)
        logger.info("Trainset saved")
    
    def train_svd(self):
        """Train SVD model"""
        logger.info("Training SVD model...")
        model_svd = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
        model_svd.fit(self.trainset)
        logger.info("SVD model trained")
        return model_svd
    
    def train_knn(self):
        """Train KNN model"""
        logger.info("Training KNN model...")
        model_knn = KNNBasic(sim_options={'user_based': True})
        model_knn.fit(self.trainset)
        logger.info("KNN model trained")
        return model_knn
    
    def train_nmf(self):
        """Train NMF model"""
        logger.info("Training NMF model...")
        model_nmf = NMF(n_factors=20, n_epochs=20, random_state=42)
        model_nmf.fit(self.trainset)
        logger.info("NMF model trained")
        return model_nmf
    
    def save_model(self, model, file_path):
        """Save trained model"""
        logger.info(f"Saving model to {file_path}")
        dump.dump(file_path, algo=model)
        logger.info("Model saved")
    
    def evaluate_model(self, model, test_df, model_name):
        """Evaluate model on test set"""
        if test_df is None or len(test_df) < 10:
            logger.warning(f"Insufficient test data for {model_name}")
            return None
        
        logger.info(f"Evaluating {model_name} on {len(test_df)} test ratings...") 
        
        testset = [
            (row['user_id'], row['movie_id'], row['rating'])
            for _, row in test_df.iterrows()
        ]
        
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        logger.info(f"{model_name} RMSE: {rmse:.4f}")
        
        return rmse
    
    def log_training_metadata(self, train_df, test_df, evaluation_results):
        """Log training run metadata with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                cursor = self.conn.cursor()
                
                total_samples = len(train_df) + (len(test_df) if test_df is not None else 0)
                
                best_model_name = None
                if evaluation_results:
                    valid_results = {k: v for k, v in evaluation_results.items() if v is not None}
                    if valid_results:
                        best_model_name = min(valid_results.items(), key=lambda x: x[1])[0]
                
                cursor.execute('''
                    INSERT INTO training_metadata (
                        training_timestamp, training_cutoff,
                        n_train_samples, n_test_samples, total_samples,
                        train_date_range_start, train_date_range_end,
                        test_date_range_start, test_date_range_end,
                        svd_rmse, knn_rmse, nmf_rmse, best_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    self.training_cutoff.isoformat() if self.training_cutoff else None,
                    len(train_df),
                    len(test_df) if test_df is not None else 0,
                    total_samples,
                    str(train_df['rated_at'].min()),
                    str(train_df['rated_at'].max()),
                    str(test_df['rated_at'].min()) if test_df is not None and len(test_df) > 0 else None,
                    str(test_df['rated_at'].max()) if test_df is not None and len(test_df) > 0 else None,
                    evaluation_results.get('svd_rmse'),
                    evaluation_results.get('knn_rmse'),
                    evaluation_results.get('nmf_rmse'),
                    best_model_name
                ))
                
                self.conn.commit()
                logger.info("Training metadata logged")

                # log to mlflow
                if hasattr(self, 'current_run_id') and self.current_run_id:
                    self.model_prov.log_metrics({
                        k:v for k,v in evaluation_results.items() if v is not None
                    })
                return
                
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Database locked, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to log metadata: {e}")
                    return

if __name__ == "__main__":
    trainer = Training()
    
    while True:
        # check training status
        logger.info("=" * 60)
        logger.info(f"Checking training status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        should_train, current_count, new_data = trainer.should_retrain()
        
        # If not enough new data, wait and check again
        if not should_train:
            logger.info("Skipping training - insufficient new data")
            
            # Check again in 3 hours
            wait_time = 60 * 60 * 1
            logger.info(f"Next check in 1 hour ({(datetime.now() + timedelta(hours=1)).strftime('%H:%M:%S')})")
            logger.info("=" * 60)
            time.sleep(wait_time)
            continue
        
        logger.info("Starting training cycle...")

        # start experiment run
        run_id = trainer.model_prov.start_run(
            run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={'trigger': 'auto', 'new_data_count': new_data}
        )
        trainer.current_run_id = run_id

        # data versioning
        try:
            logger.info("=" * 60)
            logger.info("DATA VERSIONING")
            logger.info("=" * 60)
            
            # Export data to Parquet
            parquet_file, timestamp = trainer.export_data_to_parquet()
            
            # Version with DVC
            dvc_file = trainer.data_prov.version_data(parquet_file, push=True)
            data_hash = trainer.data_prov.get_data_hash(parquet_file)
            
            logger.info(f"Data versioned: {data_hash}")
            
            # Log data version to MLflow
            trainer.model_prov.log_data_version(data_hash or timestamp, {
                'num_ratings': current_count,
                'new_data': new_data,
                'format': 'parquet',
                'compression': 'snappy'
            })
            
        except Exception as e:
            logger.error(f"Failed to version data: {e}")
            trainer.model_prov.end_run(status='FAILED')
            time.sleep(3600)
            continue
        
        # Load data
        train_df, test_df = trainer.load_data_for_training()
        
        if train_df is None:
            logger.error("Failed to load training data. Retrying in 1 hour...")
            time.sleep(3600)
            continue

        # log dataset splits to mlflow
        trainer.model_prov.log_params({
            'train_size': len(train_df),
            'test_size': len(test_df) if test_df is not None else 0,
            'test_percentage': trainer.test_percentage,
            'training_cutoff': trainer.training_cutoff.isoformat() if trainer.training_cutoff else None
        })
        
        # Train models
        model_svd = trainer.train_svd()
        # model_knn = trainer.train_knn()
        model_nmf = trainer.train_nmf()

        # log training hyperparameters (need to make training functions return their hyperparameters as a dictionary)
        trainer.model_prov.log_params({
            'svd_n_factors': 20,
            'svd_n_epochs': 20,
            'svd_lr_all': 0.01,
            'svd_reg_all': 0.01,
            'knn_user_based': True,
            'nmf_n_factors': 20,
            'nmf_n_epochs': 20
        })
        
        # Save models
        trainer.save_model(model_svd, '/app/models/svd_model.dump')
        # trainer.save_model(model_knn, '/app/models/knn_model.dump')
        trainer.save_model(model_nmf, '/app/models/nmf_model.dump')
        trainer.save_trainset()
        
        logger.info("All models trained and deployed")

        logger.info("=" * 60)
        logger.info("MODEL VERSIONING")
        logger.info("=" * 60)
        
        try:
            # Log to MLflow (for experiment tracking)
            trainer.model_prov.log_surprise_model(
                model_svd, 
                trainer.trainset, 
                artifact_path="svd_model",
                registered_model_name="movie-recommender-svd"
            )
            trainer.model_prov.log_surprise_model(
                model_knn,
                trainer.trainset,
                artifact_path="knn_model"
            )
            trainer.model_prov.log_surprise_model(
                model_nmf,
                trainer.trainset,
                artifact_path="nmf_model"
            )
            
            # Also log the actual .dump files as artifacts
            trainer.model_prov.log_artifact(svd_path)
            # trainer.model_prov.log_artifact(knn_path)
            trainer.model_prov.log_artifact(nmf_path)
            trainer.model_prov.log_artifact(trainset_path)
            
            logger.info("✓ Models logged to MLflow")
            
            # Version with DVC (for deployment)
            trainer.data_prov.version_data('models/svd_model.dump', push=True)
            trainer.data_prov.version_data('models/knn_model.dump', push=True)
            trainer.data_prov.version_data('models/nmf_model.dump', push=True)
            trainer.data_prov.version_data('models/trainset.dump', push=True)
            
            logger.info("✓ Models versioned with DVC")
            
        except Exception as e:
            logger.error(f"Failed to version models: {e}")

        
        
        evaluation_results = {}
        if test_df is not None and len(test_df) >= 10:
            logger.info("=" * 60)
            logger.info("Evaluating models on most recent data...")
            logger.info("=" * 60)
            
            evaluation_results['svd_rmse'] = trainer.evaluate_model(model_svd, test_df, 'SVD')
            # evaluation_results['knn_rmse'] = trainer.evaluate_model(model_knn, test_df, 'KNN')
            evaluation_results['nmf_rmse'] = trainer.evaluate_model(model_nmf, test_df, 'NMF')
            
            valid_results = {k: v for k, v in evaluation_results.items() if v is not None}
            if valid_results:
                best = min(valid_results.items(), key=lambda x: x[1])
                logger.info(f"Best model: {best[0].upper()} (RMSE: {best[1]:.4f})")
        else:
            logger.warning("Insufficient test data for evaluation")
        
        # Log metadata and end run
        trainer.log_training_metadata(train_df, test_df, evaluation_results)
        trainer.model_prov.end_run(status='FINISHED')
        
        logger.info("=" * 60)
        logger.info("Training cycle complete")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info("View experiments at: http://localhost:5000")
        logger.info("Will check for retraining in 1 hour")
        logger.info("=" * 60)
        
        # Check every 1 hour
        # time.sleep(60 * 60 * 1)
        time.sleep(120)
