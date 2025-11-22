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
import hashlib
import json
import requests
import time


from data_provenance import DataProvenance
from model_provenance import ModelProvenance


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trigger_rolling_model_reload():
    """
    Trigger rolling model reload across all inference replicas.
    Ensures zero-downtime deployment by reloading one server at a time.
    """
    servers = [
        'http://inference-1:5000',
        'http://inference-2:5000',
        'http://inference-3:5000'
    ]
    
    logger.info("=" * 60)
    logger.info("Starting rolling model reload (zero-downtime deployment)")
    logger.info("=" * 60)
    
    # Reload one server at a time
    for i, server in enumerate(servers, 1):
        logger.info(f"[{i}/{len(servers)}] Reloading {server}...")
        
        try:
            # Trigger reload
            resp = requests.post(f"{server}/reload_models", timeout=60)
            
            if resp.status_code == 200:
                logger.info(f"{server} reload initiated")
                
                # Wait for server to become healthy again
                if wait_for_healthy(server, timeout=90):
                    logger.info(f"{server} is healthy with new models")
                else:
                    logger.error(f"{server} failed health check after reload")
                    return False
            else:
                logger.error(f"{server} reload failed: {resp.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"{server} unreachable: {e}")
            return False
        
        # Brief pause between servers (for safety)
        if i < len(servers):
            logger.info(f"Waiting 5 seconds before next server...")
            time.sleep(5)
    
    logger.info("=" * 60)
    logger.info("All servers reloaded successfully - Zero downtime deployment complete!")
    logger.info("=" * 60)
    return True


def wait_for_healthy(server, timeout=90, interval=2):
    """
    Wait for server to respond to health checks with 'ready' status.
    """
    logger.info(f"Waiting for {server} to become healthy...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{server}/health", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'ready':
                    return True
        except:
            pass
        
        time.sleep(interval)
    
    return False


class Training:
    def __init__(self):
        self.db_path = 'db/home_alone.db'
        self.table_name = 'ratings'
        self.trainset = None
        self.conn = None
        self.model_path = '/app/models/'
        self.data_path = '/app/db/dvc_data/'
        
        # Evaluation configuration - percentage-based
        self.test_percentage = 0.30
        self.training_cutoff = None
        self.periodic_training_interval = 60 * 60 * 6  # secs * mins * N hours in seconds
        self.last_training_time = None


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


        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_deployment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_timestamp DATETIME,
                model_name TEXT,
                model_version_hash TEXT,
                data_version_hash TEXT,
                git_commit_hash TEXT,
                mlflow_run_id TEXT,
                training_id INTEGER,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (training_id) REFERENCES training_metadata(id)
            )
        ''')
        self.conn.commit()
        logger.info("Training metadata tables created")

    def get_git_commit(self):
        """Get current Git commit hash (pipeline version)"""
        commit = os.getenv('GIT_COMMIT', 'unknown')
        logger.info(f"Git commit: {commit[:8] if commit != 'unknown' else commit}")
        return commit


    def calculate_model_hash(self, model_file):
        """Calculate hash of model file for versioning"""
        import hashlib
        with open(model_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    
    def export_data_to_parquet(self, train_df, test_df):
        """Export training data to Parquet with FULL dataset + splits"""
        logger.info("Exporting training data to Parquet...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directories
        raw_dir = f'{self.data_path}/raw'
        splits_dir = f'{self.data_path}/splits'
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(splits_dir, exist_ok=True)
        
        # Export full dataset
        query = f"""
            SELECT user_id, movie_id, rating, rated_at
            FROM {self.table_name}
            ORDER BY rated_at ASC
        """
        full_df = pd.read_sql_query(query, self.conn)
        
        full_file = f'{raw_dir}/ratings_full_{timestamp}.parquet'
        full_df.to_parquet(full_file, engine='pyarrow', compression='snappy', index=False)
        
        # Export train/test splits
        train_file = f'{splits_dir}/train_{timestamp}.parquet'
        test_file = f'{splits_dir}/test_{timestamp}.parquet'
        
        train_df.to_parquet(train_file, engine='pyarrow', compression='snappy', index=False)
        test_df.to_parquet(test_file, engine='pyarrow', compression='snappy', index=False)
        
        # Create symlinks
        for src, link_name in [
            (full_file, f'{raw_dir}/ratings_latest.parquet'),
            (train_file, f'{splits_dir}/train_latest.parquet'),
            (test_file, f'{splits_dir}/test_latest.parquet')
        ]:
            if os.path.exists(link_name):
                os.remove(link_name)
            os.symlink(os.path.basename(src), link_name)
        
        # Calculate data hash
        data_hash = hashlib.md5(
            full_df.head(1000).to_csv(index=False).encode()
        ).hexdigest()[:8]
        
        logger.info(f"Full dataset: {len(full_df)} rows")
        logger.info(f"Train split: {len(train_df)} rows")
        logger.info(f"Test split: {len(test_df)} rows")
        logger.info(f"Data hash: {data_hash}")
        
        return {
            'full_file': full_file,
            'train_file': train_file,
            'test_file': test_file,
            'data_hash': data_hash,
            'timestamp': timestamp
        }



    def get_percentage_based_split(self):
        """
        Split data by percentage (temporal ordering).
        
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
        
        logger.info(f"Training set: {len(train_df)} ratings ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {train_df['rated_at'].min()} to {train_df['rated_at'].max()}")
        logger.info(f"Test set: {len(test_df)} ratings ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"  Date range: {test_df['rated_at'].min()} to {test_df['rated_at'].max()}")
        logger.info(f"Training cutoff: {cutoff_timestamp}")
        logger.info(f"Unique users: {len(train_users)}, Unique movies: {len(train_movies)}")
        
        return train_df, test_df, cutoff_timestamp
    
    def load_data_for_training(self):
        """
        Load data and create trainset for model training.
        Uses percentage-based split: 70% train, 30% test.
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
        return file_path
    
    def train_svd(self):
        """Train SVD model"""
        logger.info("Training SVD model...")
        model_svd = SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
        model_svd.fit(self.trainset)
        logger.info("SVD model trained")
        return model_svd
    
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
        return file_path
    
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
    
    def save_provenance_manifest(self, provenance_data):
        """
        This allows anyone to trace back from model > data > code.
        """
        manifest_path = os.path.join(self.model_path, 'provenance_manifest.json')
        
        with open(manifest_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)
        
        logger.info(f"Provenance manifest saved: {manifest_path}")
        return manifest_path
    
    def log_training_metadata(self, train_df, test_df, evaluation_results,
                              mlflow_run_id, data_version, git_commit, model_version):
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
                        svd_rmse, nmf_rmse, best_model,
                        mlflow_run_id, data_version_hash, git_commit_hash, model_version_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    evaluation_results.get('nmf_rmse'),
                    best_model_name,
                    mlflow_run_id,
                    data_version,
                    git_commit,
                    model_version
                ))


                training_id = cursor.lastrowid


                cursor.execute('UPDATE model_deployment SET is_active = 0')


                cursor.execute('''
                    INSERT INTO model_deployment (
                        deployment_timestamp, model_name,
                        model_version_hash, data_version_hash, git_commit_hash,
                        mlflow_run_id, training_id, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ''', (
                    datetime.now().isoformat(),
                    best_model_name or 'svd',
                    model_version,
                    data_version,
                    git_commit,
                    mlflow_run_id,
                    training_id
                ))
                
                self.conn.commit()
                logger.info("Training metadata and deployment logged")


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
        logger.info(f"Checking training status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if enough time has passed since last training
        should_train = False
        if trainer.last_training_time is None:
            should_train = True
            logger.info("First training run - will train")
        else:
            time_since_last = (datetime.now() - trainer.last_training_time).total_seconds()
            should_train = time_since_last >= trainer.periodic_training_interval
            if should_train:
                logger.info(f"Periodic training triggered ({time_since_last/3600:.1f} hours since last training)")
            else:
                hours_remaining = (trainer.periodic_training_interval - time_since_last) / 3600
                logger.info(f"Not yet time for training ({hours_remaining:.1f} hours remaining)")
        
        # If not time for training, wait and check again
        if not should_train:
            wait_time = trainer.periodic_training_interval / 4
            logger.info(f"Waiting for {wait_time} seconds...")
            logger.info(f"Next check at {(datetime.now() + timedelta(seconds=wait_time)).strftime('%H:%M:%S')}")
            time.sleep(wait_time)
            continue
        
        logger.info("Starting training cycle...")

        git_commit = trainer.get_git_commit()
        
        # Get current data count for logging
        query = f"SELECT COUNT(*) as count FROM {trainer.table_name}"
        current_count = pd.read_sql_query(query, trainer.conn).iloc[0]['count']
        logger.info(f"Current data: {current_count} ratings")
        
        run_id = trainer.model_prov.start_run(
            run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={'trigger': 'periodic', 'total_data_count': str(current_count), 'git_commit': git_commit[:8]}
        )
        trainer.current_run_id = run_id

        train_df, test_df = trainer.load_data_for_training()

        if train_df is None:
            trainer.model_prov.end_run(status='FAILED')
            time.sleep(60)
            continue


        # data versioning
        try:
            logger.info("Data versioning...")
            
            # Export data to Parquet
            data_info = trainer.export_data_to_parquet(train_df, test_df)


            trainer.data_prov.version_data(data_info['full_file'], push=True)
            trainer.data_prov.version_data(data_info['train_file'], push=True)
            trainer.data_prov.version_data(data_info['test_file'], push=True)


            data_version = data_info['data_hash']


            trainer.model_prov.log_data_version(data_version, {
                'full_size': len(pd.read_parquet(data_info['full_file'])),
                'train_size': len(train_df),
                'test_size': len(test_df),
                'format': 'parquet',
                'timestamp': data_info['timestamp'],
                'compression': 'snappy'
            })
            
        except Exception as e:
            logger.error(f"Failed to version data: {e}")
            trainer.model_prov.end_run(status='FAILED')
            continue


        # log config
        trainer.model_prov.log_params({
            'git_commit': git_commit,
            'data_version': data_version,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'svd_n_factors': 20,
            'svd_lr': 0.01,
            'nmf_n_factors': 20,
            'training_cutoff': trainer.training_cutoff.isoformat() if trainer.training_cutoff else None
        })
        
        # Train models
        model_svd = trainer.train_svd()
        model_nmf = trainer.train_nmf()
        
        # Save models
        svd_path = trainer.save_model(model_svd, '/app/models/svd_model.dump')
        nmf_path = trainer.save_model(model_nmf, '/app/models/nmf_model.dump')
        trainset_path = trainer.save_trainset()
        
        model_version = trainer.calculate_model_hash(svd_path)
        logger.info(f"Model version hash: {model_version}")

        
        logger.info("Model versioning...")
        try:
            # Log to MLflow
            trainer.model_prov.log_artifact(svd_path, artifact_path="models")
            trainer.model_prov.log_artifact(nmf_path, artifact_path="models")
            trainer.model_prov.log_artifact(trainset_path, artifact_path="models")
            logger.info("Models logged to mlflow")
            
            # Version with DVC
            trainer.data_prov.version_data('models/svd_model.dump', push=True)
            trainer.data_prov.version_data('models/nmf_model.dump', push=True)
            trainer.data_prov.version_data('models/trainset.dump', push=True)
            logger.info("Models versioned with DVC")


            provenance_data = {
                'model_version': model_version,
                'data_version': data_version,
                'git_commit': git_commit,
                'mlflow_run_id': run_id,
                'training_timestamp': datetime.now().isoformat(),
                'model_files': {
                    'svd': 'models/svd_model.dump',
                    'nmf': 'models/nmf_model.dump',
                    'trainset': 'models/trainset.dump'
                },
                'data_files': {
                    'full': data_info['full_file'],
                    'train': data_info['train_file'],
                    'test': data_info['test_file']
                },
                'hyperparameters': {
                    'svd_n_factors': 20,
                    'svd_lr': 0.01,
                    'svd_reg': 0.01,
                    'svd_epochs': 20
                }
            }


            manifest_path = trainer.save_provenance_manifest(provenance_data)
            trainer.model_prov.log_artifact(manifest_path)


            logger.info("Models and provenance versioned")
            
        except Exception as e:
            logger.error(f"Failed to version models: {e}")

        logger.info("Triggering model reload on inference servers...")
        reload_success = trigger_rolling_model_reload()

        if reload_success:
            logger.info("Models deployed to all replicas successfully")
        else:
            logger.error("Model deployment failed on some replicas")
        
        
        evaluation_results = {}
        if test_df is not None and len(test_df) >= 10:
            logger.info("Evaluating models on most recent data...")
            
            evaluation_results['svd_rmse'] = trainer.evaluate_model(model_svd, test_df, 'SVD')
            evaluation_results['nmf_rmse'] = trainer.evaluate_model(model_nmf, test_df, 'NMF')
            valid_results = {k: v for k, v in evaluation_results.items() if v is not None}
            if valid_results:
                best = min(valid_results.items(), key=lambda x: x[1])
                logger.info(f"Best model: {best[0].upper()} (RMSE: {best[1]:.4f})")
        else:
            logger.warning("Insufficient test data for evaluation")
            reason = ""
            if test_df is None:
                reason = "test_df is None"
                logger.warning("  Reason: test_df is None")
            else:
                reason = f"Only {len(test_df)} test samples"
                logger.warning(f"  Reason: Only {len(test_df)} test samples (need >= 10)")
            logger.warning("Model will be logged WITHOUT evaluation metrics")


            trainer.model_prov.log_params({
                'evaluation_skipped': True,
                'evaluation_skip_reason': reason,
            })


        # Log metadata
        trainer.log_training_metadata(
            train_df, test_df, evaluation_results,
            mlflow_run_id=run_id,
            data_version=data_version,
            git_commit=git_commit,
            model_version=model_version
        )
        
        if not evaluation_results:
            trainer.model_prov.end_run(status='FAILED')
            logger.error("Training marked as FAILED due to missing evaluation")
        else:
            trainer.model_prov.end_run(status='FINISHED')
            logger.info("Training completed successfully")
        
        # Update last training time
        trainer.last_training_time = datetime.now()
        logger.info(f"Training completed at {trainer.last_training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Next training in {trainer.periodic_training_interval/3600:.0f} hours")
            
        logger.info("Training Completed")
        logger.info(f"Model Version: {model_version}")
        logger.info(f"Data Version: {data_version}")
        logger.info(f"Git Commit: {git_commit[:8]}")
        logger.info(f"MLflow Run: {run_id}")
        logger.info("View experiments at: http://localhost:5000")
        if evaluation_results:
            logger.info(f"Metrics: {evaluation_results}")
        else:
            logger.info("Metrics: NONE (insufficient test data)")
        
        # Check every 15 minutes
        time.sleep(60 * 15)
