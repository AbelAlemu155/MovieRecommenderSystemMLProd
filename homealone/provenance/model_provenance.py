import mlflow
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime
from homealone.provenance.config import ProvenanceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvenance:
    """
    Manages model provenance using MLflow.
    
    Functions:
    - Track experiments and runs
    - Log parameters, metrics, artifacts
    - Register models
    - Track model lineage
    """

    def __init__(self, config: Optional[ProvenanceConfig] = None):
        self.config = config or ProvenanceConfig.from_env()
        self.current_run = None
    
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

    def initialize(self) -> bool:
        try:
            experiment = mlflow.set_experiment(self.config.mlflow_experiment_name)
            logger.info(f"MLflow experiment: {experiment.name} (ID: {experiment.experiment_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            return False

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> str:
        """
        Start a new MLflow run.
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to run
        Returns:
            Run ID
        """
        if self.current_run:
            logger.warning("A run is already active. Ending it first.")
            self.end_run()
        
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        run_id = self.current_run.info.run_id
        
        logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
        return run_id

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.current_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]):
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        if not self.current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_artifact(filepath, artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {filepath}")

    def log_data_version(self, data_hash: str, data_info: Dict):
        self.log_params({
            'data_version': data_hash,
            **{f'data_{k}': v for k, v in data_info.items()}
        })
        logger.info(f"Logged data version: {data_hash}")

    def get_run_info(self, run_id: Optional[str] = None) -> Dict:
        if run_id is None:
            if not self.current_run:
                raise RuntimeError("No active run and no run_id provided")
            run_id = self.current_run.info.run_id
        
        run = mlflow.get_run(run_id)
        
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }

    def search_runs(self, filter_string: Optional[str] = None, 
                    max_results: int = 100) -> List[Dict]:
        experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
        
        return runs.to_dict('records')

    def get_best_run(self, metric: str = 'test_rmse', ascending: bool = True) -> Optional[Dict]:
        runs = self.search_runs()
        
        if not runs:
            return None
        
        # Sort by metric
        sorted_runs = sorted(
            runs,
            key=lambda r: r.get(f'metrics.{metric}', float('inf')),
            reverse=not ascending
        )
        
        return sorted_runs[0] if sorted_runs else None
