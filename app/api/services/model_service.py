# app/api/services/model_service.py
"""
Model management service for credit scoring.
Handles model loading, deployment, and versioning.
"""
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import mlflow.sklearn
from mlflow.tracking import MlflowClient

from app.config.settings import get_config
from app.utils.logging_config import setup_logging

logger = setup_logging('model')


class ModelService:
    """Service for managing ML models (loading, deployment, rollback)."""
    
    def __init__(self):
        self.config = get_config()
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self._model_cache = None
        self._threshold_cache = None
    
    def get_production_model(self) -> Tuple[Any, float]:
        """
        Get the production model with fallback strategy:
        1. First try /prod_models/model.pkl (git-committed, preferred)
        2. Then try MLflow registry with 'Production' stage
        
        Returns:
            Tuple of (model, threshold)
        """
        # Try file-based model first
        model, threshold = self._load_from_file()
        if model is not None:
            return model, threshold
        
        # Fallback to MLflow registry
        return self._load_from_mlflow()
    
    def _load_from_file(self) -> Tuple[Any, float]:
        """Load model from prod_models/ directory."""
        default_threshold = self.config.DEFAULT_THRESHOLD
        
        try:
            if os.path.exists(self.config.PROD_MODEL_PATH):
                with open(self.config.PROD_MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"✅ Model loaded from {self.config.PROD_MODEL_PATH}")
            else:
                logger.warning(f"⚠️ Model file not found at {self.config.PROD_MODEL_PATH}")
                return None, default_threshold
            
            # Load threshold
            if os.path.exists(self.config.PROD_THRESHOLD_PATH):
                with open(self.config.PROD_THRESHOLD_PATH, 'r') as f:
                    threshold_data = json.load(f)
                    threshold = float(threshold_data.get("optimal_threshold", default_threshold))
                logger.info(f"✅ Threshold loaded: {threshold}")
            else:
                threshold = default_threshold
                logger.warning(f"⚠️ Using default threshold: {threshold}")
            
            return model, threshold
        
        except Exception as e:
            logger.error(f"❌ Error loading model from file: {e}")
            return None, default_threshold
    
    def _load_from_mlflow(self) -> Tuple[Any, float]:
        """Load model from MLflow registry."""
        try:
            model_uri = f"models:/{self.config.MODEL_NAME}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get threshold from run params
            versions = self.client.get_latest_versions(self.config.MODEL_NAME, stages=["Production"])
            if versions:
                run_id = versions[0].run_id
                run = self.client.get_run(run_id)
                threshold = float(run.data.params.get("business_optimal_threshold", self.config.DEFAULT_THRESHOLD))
            else:
                threshold = self.config.DEFAULT_THRESHOLD
            
            logger.info(f"✅ Model loaded from MLflow: {model_uri}")
            return model, threshold
        except Exception as e:
            logger.error(f"Error loading from MLflow: {e}")
            return None, self.config.DEFAULT_THRESHOLD
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current production model."""
        try:
            model, threshold = self.get_production_model()
            model_source = "file" if os.path.exists(self.config.PROD_MODEL_PATH) else "mlflow"
            
            metadata = self._load_metadata()
            
            return {
                "model_loaded": model is not None,
                "threshold": threshold,
                "source": model_source,
                "metadata": metadata
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from file."""
        if os.path.exists(self.config.PROD_METADATA_PATH):
            with open(self.config.PROD_METADATA_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models."""
        models = []
        prod_dir = Path(self.config.PROD_MODEL_PATH).parent
        
        # File-based models
        if prod_dir.exists():
            for model_file in prod_dir.glob("*.pkl"):
                metadata = {}
                metadata_path = prod_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                models.append({
                    "source": "file",
                    "path": str(model_file),
                    "name": model_file.stem,
                    "active": str(model_file) == self.config.PROD_MODEL_PATH,
                    "metadata": metadata
                })
        
        # MLflow models
        try:
            for rm in self.client.search_registered_models():
                for version in self.client.search_model_versions(f"name='{rm.name}'"):
                    models.append({
                        "source": "mlflow",
                        "name": rm.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "active": version.current_stage == "Production"
                    })
        except Exception as e:
            logger.warning(f"MLflow listing error: {e}")
        
        return {"models": models}
    
    def get_current_model(self) -> Dict[str, Any]:
        """Get current model details."""
        model, threshold = self.get_production_model()
        model_source = "file" if os.path.exists(self.config.PROD_MODEL_PATH) else "mlflow"
        
        return {
            "model_loaded": model is not None,
            "source": model_source,
            "threshold": threshold,
            "path": self.config.PROD_MODEL_PATH if model_source == "file" else None,
            "metadata": self._load_metadata()
        }
    
    def deploy_model(
        self,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deploy a model from MLflow to production."""
        try:
            model_name = model_name or self.config.MODEL_NAME
            
            if not run_id and not version:
                return {"error": "Provide either run_id or version"}
            
            # Get run_id from version if needed
            if version and not run_id:
                mv = self.client.get_model_version(model_name, version)
                run_id = mv.run_id
            
            # Load model
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get run info
            run = self.client.get_run(run_id)
            threshold = float(run.data.params.get("business_optimal_threshold", self.config.DEFAULT_THRESHOLD))
            
            # Backup current model
            prod_dir = Path(self.config.PROD_MODEL_PATH).parent
            prod_dir.mkdir(parents=True, exist_ok=True)
            
            if os.path.exists(self.config.PROD_MODEL_PATH):
                backup_path = self.config.PROD_MODEL_PATH.replace(
                    ".pkl", 
                    f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                )
                os.rename(self.config.PROD_MODEL_PATH, backup_path)
            
            # Save new model
            with open(self.config.PROD_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                "run_id": run_id,
                "model_name": model_name,
                "deployed_at": datetime.now().isoformat(),
                "optimal_threshold": threshold,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            with open(self.config.PROD_METADATA_PATH, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save threshold
            with open(self.config.PROD_THRESHOLD_PATH, 'w') as f:
                json.dump({"optimal_threshold": threshold}, f, indent=2)
            
            logger.info(f"✅ Model {run_id} deployed to production")
            
            return {
                "success": True,
                "message": f"Model {run_id} deployed to production",
                "metadata": metadata
            }
        
        except Exception as e:
            logger.error(f"Deploy error: {e}")
            return {"error": str(e)}
    
    def rollback_model(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """Rollback to a previous model version."""
        prod_dir = Path(self.config.PROD_MODEL_PATH).parent
        backups = list(prod_dir.glob("model_backup_*.pkl"))
        
        if not backup_name:
            return {
                "backups": [
                    {"name": b.name, "created": b.stat().st_mtime}
                    for b in backups
                ]
            }
        
        backup_path = prod_dir / backup_name
        if not backup_path.exists():
            return {"error": f"Backup not found: {backup_name}"}
        
        # Swap
        current_backup = self.config.PROD_MODEL_PATH.replace(
            ".pkl",
            f"_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        if os.path.exists(self.config.PROD_MODEL_PATH):
            os.rename(self.config.PROD_MODEL_PATH, current_backup)
        os.rename(str(backup_path), self.config.PROD_MODEL_PATH)
        
        logger.info(f"✅ Rolled back to {backup_name}")
        
        return {
            "success": True,
            "message": f"Rolled back to {backup_name}",
            "previous_saved_as": current_backup
        }
