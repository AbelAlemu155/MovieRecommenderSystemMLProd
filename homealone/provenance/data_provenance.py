import os
import subprocess
from typing import Dict, Optional, List
from pathlib import Path
import logging

from homealone.provenance.config import ProvenanceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvenance:
    """
    Manages data provenance using DVC.

    Functions:
    - initialize DVC repository
    - version data snapshots
    - track data lineage
    - push/pull data to/from remote storage
    """

    def __init__(self, config: Optional[ProvenanceConfig] = None):
        self.config = config or ProvenanceConfig.from_env()
        self.repo_root = Path(self.config.repo_root)
        self.data_dir = Path(self.config.data_dir)

        logger.info(f"Initializing DataProvenance with repo root: {self.repo_root}")

    def initialize(self) -> bool:
        """Initialize DVC repository if not already initialized."""
        dvc_dir = self.repo_root / '.dvc'
        if dvc_dir.exists():
            logger.info("DVC repository already initialized.")
            return True
        
        try:
            self._run_command("dvc init --no-scm")

            # Configure remote
            logger.info(f"Configuring DVC remote: {self.config.dvc_remote_url}")
            self._run_command(
                f"dvc remote add -d {self.config.dvc_remote_name} {self.config.dvc_remote_url}"
            )

            logger.info("DVC initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DVC: {e}")
            return False

    def version_data(self, filepath: str, push: bool = True) -> Optional[str]:
        """
        Version a data file with DVC.
        
        Args:
            filepath: Path to data file (relative to repo root)
            push: Whether to push to remote storage
            
        Returns:
            Path to .dvc file if successful, None otherwise
        """
        try:
            logger.info(f"Versioning data file: {filepath}")
            
            full_path = self.repo_root / filepath
            if not full_path.exists():
                raise FileNotFoundError(f"Data file not found: {full_path}")
            
            self._run_command(f"dvc add {filepath}")
            dvc_file = f"{filepath}.dvc"
            
            logger.info(f"Created DVC file: {dvc_file}")
            
            if push:
                self.push(filepath)
            
            return dvc_file
            
        except Exception as e:
            logger.error(f"Failed to version data: {e}")
            return None

    def push(self, filepath: Optional[str] = None) -> bool:
        """
        Push data to DVC remote storage.
        
        Args:
            filepath: Specific file to push, or None to push all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath:
                logger.info(f"Pushing {filepath} to DVC remote...")
                self._run_command(f"dvc push {filepath}.dvc")
            else:
                logger.info("Pushing all data to DVC remote...")
                self._run_command("dvc push")
            
            logger.info("Push successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push to DVC remote: {e}")
            return False

    def pull(self, filepath: Optional[str] = None) -> bool:
        """
        Pull data from DVC remote storage.
        
        Args:
            filepath: Specific file to pull, or None to pull all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filepath:
                logger.info(f"Pulling {filepath} from DVC remote...")
                self._run_command(f"dvc pull {filepath}.dvc")
            else:
                logger.info("Pulling all data from DVC remote...")
                self._run_command("dvc pull")
            
            logger.info("Pull successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull from DVC remote: {e}")
            return False

    def get_data_hash(self, filepath: str) -> Optional[str]:
        """
        Get MD5 hash of data file from DVC.
        
        Args:
            filepath: Path to data file
            
        Returns:
            MD5 hash string or None if file not tracked
        """
        try:
            import yaml
            
            dvc_file = self.repo_root / f"{filepath}.dvc"
            if not dvc_file.exists():
                return None
            
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
            
            return dvc_data['outs']['md5']
            
        except Exception as e:
            logger.error(f"Failed to get data hash: {e}")
            return None

    def get_data_info(self, filepath: str) -> Dict:
        """
        Get information about a tracked data file.
        
        Args:
            filepath: Path to data file
            
        Returns:
            Dictionary with file information
        """
        info = {
            'filepath': filepath,
            'exists': (self.repo_root / filepath).exists(),
            'tracked': (self.repo_root / f"{filepath}.dvc").exists(),
            'hash': None,
            'size': None
        }
        
        if info['tracked']:
            info['hash'] = self.get_data_hash(filepath)
        
        if info['exists']:
            full_path = self.repo_root / filepath
            info['size'] = full_path.stat().st_size
        
        return info

    def _run_command(self, cmd: str, capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a shell command."""
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, cwd=self.repo_root)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
        return result
