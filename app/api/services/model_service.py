# app/api/services/model_service.py
"""
Model management service for credit scoring.
Handles model loading, deployment, and versioning.
Supports both local pickle files and MLflow Serving with SHAP.
"""
import os
import json
import pickle
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

from app.config.settings import get_config
from app.utils.logging_config import setup_logging

logger = setup_logging('model')


class ModelService:
    """Service for managing ML models (loading, deployment, rollback)."""
    
    # Class-level cache for singleton-like behavior
    _model_cache = None
    _threshold_cache = None
    _model_source = None
    _cached_version = None  # Track which MLflow version is cached
    
    def __init__(self):
        self.config = get_config()
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
    
    def is_mlflow_serving_enabled(self) -> bool:
        """Check if MLflow Serving mode is enabled."""
        return self.config.USE_MLFLOW_SERVING
    
    def predict_with_mlflow_serving(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction via MLflow Serving endpoint with SHAP explanations.
        
        Args:
            X: DataFrame with features for prediction
            
        Returns:
            Dict with probability, decision, threshold, and SHAP explanation
        """
        if not self.is_mlflow_serving_enabled():
            raise ValueError("MLflow Serving is not enabled. Set USE_MLFLOW_SERVING=true")
        
        try:
            # Prepare request for MLflow Serving
            # MLflow expects 'dataframe_split' format
            payload = {
                "dataframe_split": {
                    "columns": X.columns.tolist(),
                    "data": X.values.tolist()
                }
            }
            
            # Call MLflow Serving endpoint
            url = f"{self.config.MLFLOW_SERVING_URI}/invocations"
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"MLflow Serving error: {response.status_code} - {response.text}")
                return {"error": f"MLflow Serving returned {response.status_code}"}
            
            result = response.json()
            
            # Parse response from custom PyFunc model
            # Expected format: {"predictions": [{"probability": 0.25, "decision": "ACCEPTED", ...}]}
            if "predictions" in result:
                pred = result["predictions"][0] if isinstance(result["predictions"], list) else result["predictions"]
                return {
                    "probability": pred.get("probability"),
                    "decision": pred.get("decision"),
                    "threshold": pred.get("threshold"),
                    "shap_explanation": json.loads(pred.get("shap_explanation", "{}"))
                }
            else:
                # Handle raw probability array (fallback for non-custom models)
                proba = result[0] if isinstance(result, list) else result
                threshold = self._get_threshold()
                return {
                    "probability": float(proba),
                    "decision": "REJECTED" if proba >= threshold else "ACCEPTED",
                    "threshold": threshold,
                    "shap_explanation": None
                }
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to MLflow Serving at {self.config.MLFLOW_SERVING_URI}")
            return {"error": "MLflow Serving is not available"}
        except Exception as e:
            logger.error(f"MLflow Serving prediction error: {e}")
            return {"error": str(e)}
    
    def _get_threshold(self) -> float:
        """Get threshold from cache or config files. Raises error if not configured."""
        if ModelService._threshold_cache is not None:
            return ModelService._threshold_cache
        
        # Try threshold.json first
        if os.path.exists(self.config.PROD_THRESHOLD_PATH):
            with open(self.config.PROD_THRESHOLD_PATH, 'r') as f:
                threshold_data = json.load(f)
                if "optimal_threshold" in threshold_data:
                    threshold = float(threshold_data["optimal_threshold"])
                    logger.info(f"Threshold loaded from threshold.json: {threshold}")
                    return threshold
        
        # Try metadata.json as fallback source
        if os.path.exists(self.config.PROD_METADATA_PATH):
            with open(self.config.PROD_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                if "optimal_threshold" in metadata:
                    threshold = float(metadata["optimal_threshold"])
                    logger.info(f"Threshold loaded from metadata.json: {threshold}")
                    return threshold
        
        # NO FALLBACK - fail clearly
        raise ValueError(
            "CRITICAL: Threshold not configured! "
            "Ensure prod_models/threshold.json or prod_models/metadata.json exists "
            "with 'optimal_threshold' field. Application cannot run without threshold configuration."
        )

    def get_current_model_id(self) -> Optional[str]:
        """
        Get the current model ID (run_id) for the active champion model.
        This is used to link predictions to the model that made them.
        
        Returns:
            Model ID string or None if not available
        """
        # First check metadata file
        metadata = self._load_metadata()
        if metadata.get("run_id"):
            return metadata.get("run_id")
        
        # Fallback: get from MLflow champion alias (MLflow 2.9+)
        try:
            version_info = self.client.get_model_version_by_alias(self.config.MODEL_NAME, "champion")
            return version_info.run_id
        except Exception as e:
            logger.warning(f"Could not get model ID from MLflow: {e}")
        
        return None

    def get_current_model_version(self) -> Optional[str]:
        """
        Get the current model version string.
        
        Returns:
            Model version string or None if not available
        """
        # First check metadata file
        metadata = self._load_metadata()
        if metadata.get("model_version"):
            return metadata.get("model_version")
        
        # Fallback: get from MLflow champion alias (MLflow 2.9+)
        try:
            version_info = self.client.get_model_version_by_alias(self.config.MODEL_NAME, "champion")
            return version_info.version
        except Exception as e:
            logger.warning(f"Could not get model version from MLflow: {e}")
        
        return None
    
    def get_production_model(self) -> Tuple[Any, float]:
        """
        Get the production model with caching and fallback strategy:
        1. Check if cached model version matches MLflow Production version
        2. If not, reload from MLflow registry
        3. Fallback to /prod_models/model.pkl (for offline/testing)
        
        Returns:
            Tuple of (model, threshold)
        """
        # Check if cache is still valid (same version as MLflow Production)
        if ModelService._model_cache is not None:
            try:
                current_prod_version = self._get_mlflow_production_version()
                if current_prod_version and current_prod_version != ModelService._cached_version:
                    logger.info(f"MLflow Production changed from v{ModelService._cached_version} to v{current_prod_version}, reloading...")
                    ModelService.clear_cache()
                else:
                    logger.debug(f"Using cached model v{ModelService._cached_version} (source: {ModelService._model_source})")
                    return ModelService._model_cache, ModelService._threshold_cache
            except Exception as e:
                logger.warning(f"Version check failed, using cache: {e}")
                return ModelService._model_cache, ModelService._threshold_cache
        
        # Try MLflow registry first (primary source of truth)
        model, threshold = self._load_from_mlflow()
        if model is not None:
            ModelService._model_cache = model
            ModelService._threshold_cache = threshold
            ModelService._model_source = 'mlflow'
            ModelService._cached_version = self._get_mlflow_production_version()
            return model, threshold
        
        # Fallback to file-based model (for offline/testing)
        model, threshold = self._load_from_file()
        if model is not None:
            ModelService._model_cache = model
            ModelService._threshold_cache = threshold
            ModelService._model_source = 'file'
            ModelService._cached_version = 'file'
        return model, threshold
    
    def _get_mlflow_production_version(self) -> Optional[str]:
        """Get the current champion version from MLflow registry (using aliases)."""
        try:
            version_info = self.client.get_model_version_by_alias(self.config.MODEL_NAME, "champion")
            return version_info.version
        except:
            pass
        return None
    
    @classmethod
    def clear_cache(cls):
        """Clear model cache (useful for testing or model reload)."""
        cls._model_cache = None
        cls._threshold_cache = None
        cls._model_source = None
        cls._cached_version = None
        logger.info("Model cache cleared")
    
    @classmethod
    def is_model_loaded(cls) -> bool:
        """Check if model is cached."""
        return cls._model_cache is not None
    
    def _load_from_file(self) -> Tuple[Any, float]:
        """Load model from prod_models/ directory. Raises error if threshold not configured."""
        try:
            if os.path.exists(self.config.PROD_MODEL_PATH):
                with open(self.config.PROD_MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"✅ Model loaded from {self.config.PROD_MODEL_PATH}")
            else:
                raise FileNotFoundError(f"Model file not found at {self.config.PROD_MODEL_PATH}")
            
            # Load threshold - will raise error if not configured
            threshold = self._get_threshold()
            logger.info(f"✅ Threshold loaded: {threshold}")
            
            return model, threshold
        
        except Exception as e:
            logger.error(f"❌ Error loading model from file: {e}")
            raise
    
    def _load_from_mlflow(self) -> Tuple[Any, float]:
        """Load model from MLflow registry using alias (MLflow 2.9+ compatible)."""
        try:
            # Use @champion alias (new MLflow 2.9+ approach)
            model_uri = f"models:/{self.config.MODEL_NAME}@champion"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get threshold from run params using alias
            try:
                version_info = self.client.get_model_version_by_alias(self.config.MODEL_NAME, "champion")
                run_id = version_info.run_id
                run = self.client.get_run(run_id)
                if "business_optimal_threshold" in run.data.params:
                    threshold = float(run.data.params["business_optimal_threshold"])
                    logger.info(f"Threshold loaded from MLflow run params: {threshold}")
                else:
                    # Fallback to file-based threshold
                    threshold = self._get_threshold()
            except Exception as mlflow_err:
                logger.warning(f"Could not get threshold from MLflow: {mlflow_err}, using file-based threshold")
                threshold = self._get_threshold()
            
            logger.info(f"✅ Model loaded from MLflow: {model_uri}")
            return model, threshold
        except Exception as e:
            logger.error(f"Error loading from MLflow: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current production model."""
        try:
            model, threshold = self.get_production_model()
            
            # Use the actual source from where the model was loaded
            model_source = ModelService._model_source or "unknown"
            
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
        
        # MLflow models - using aliases (MLflow 2.9+)
        try:
            # Get current champion version for comparison
            champion_version = None
            try:
                champion_info = self.client.get_model_version_by_alias(self.config.MODEL_NAME, "champion")
                champion_version = champion_info.version
            except:
                pass
            
            for rm in self.client.search_registered_models():
                for version in self.client.search_model_versions(f"name='{rm.name}'"):
                    is_champion = version.version == champion_version
                    # Convert aliases to list (it's a RepeatedScalarContainer)
                    aliases = list(version.aliases) if hasattr(version, 'aliases') and version.aliases else []
                    models.append({
                        "source": "mlflow",
                        "name": rm.name,
                        "version": version.version,
                        "aliases": aliases,
                        "run_id": version.run_id,
                        "active": is_champion
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

    def promote_version(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Promote a model version using MLflow Model Aliases (MLflow 2.9+).
        Sets the 'champion' alias on the specified version.
        This is the primary deployment mechanism - clears cache so next
        prediction uses the newly promoted model.
        """
        try:
            # Get the run_id for the version being promoted
            version_info = self.client.get_model_version(model_name, version)
            run_id = version_info.run_id
            
            # Set the 'champion' alias on this version (replaces old alias automatically)
            # This is the MLflow 2.9+ way - no need to manually remove from previous version
            self.client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=version
            )
            logger.info(f"Set 'champion' alias on {model_name} v{version}")
            
            # Clear the model cache so next prediction loads the new model
            ModelService.clear_cache()
            
            # Update metadata file for audit trail
            try:
                run = self.client.get_run(run_id)
                metadata = {
                    "run_id": run_id,
                    "model_name": model_name,
                    "model_version": version,
                    "deployed_at": datetime.now().isoformat(),
                    "source": "mlflow_registry",
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params)
                }
                with open(self.config.PROD_METADATA_PATH, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Update threshold if available
                threshold = float(run.data.params.get("business_optimal_threshold", self.config.DEFAULT_THRESHOLD))
                with open(self.config.PROD_THRESHOLD_PATH, 'w') as f:
                    json.dump({"optimal_threshold": threshold}, f, indent=2)
                    
                logger.info(f"Updated metadata for v{version}")
            except Exception as e:
                logger.warning(f"Could not update metadata: {e}")
            
            logger.info(f"✅ Set champion alias on {model_name} v{version}")
            
            return {
                "success": True,
                "message": f"Deployed {model_name} version {version} as champion",
                "model_name": model_name,
                "version": version,
                "run_id": run_id,
                "alias": "champion"
            }
        
        except Exception as e:
            logger.error(f"Promote error: {e}")
            return {"error": str(e)}