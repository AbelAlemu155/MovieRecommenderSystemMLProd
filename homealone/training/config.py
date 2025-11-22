import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProvenanceConfig:
    """configurating for data and model provenance tracking"""

    dvc_remote_name: str = "mylocal"
    dvc_remote_url: str = '/dvc-storage'
    dvc_cache_dir: str = '.dvc/cache'

    mlflow_tracking_uri: str = 'http://mlflow:5000'
    mlflow_experiment_name: str = 'homealone-movie-recommendations'
    mlflow_artifact_location: Optional[str] = None

    data_dir: str = '/app/homealone/data/raw'
    model_dir: str = '/app/homealone/models'
    repo_root: str = '/app'

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            dvc_remote_name=os.getenv('DVC_REMOTE_NAME', 'mylocal'),
            dvc_remote_url=os.getenv('DVC_REMOTE_URL', '/dvc-storage'),
            dvc_cache_dir=os.getenv('DVC_CACHE_DIR', '.dvc/cache'),
            mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
            mlflow_experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'homealone-movie-recommendations'),
            mlflow_artifact_location=os.getenv('MLFLOW_ARTIFACT_LOCATION'),
            data_dir=os.getenv('DATA_DIR', '/app/homealone/data/raw'),
            model_dir=os.getenv('MODEL_DIR', '/app/homealone/models'),
            repo_root=os.getenv('REPO_ROOT', '/app')
        )

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'dvc_remote_name': self.dvc_remote_name,
            'dvc_remote_url': self.dvc_remote_url,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'repo_root': self.repo_root
        }
