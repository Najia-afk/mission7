# app/config/settings.py
"""
Configuration settings for Mission7 Credit Scoring API.
Supports both development (SQLite) and production (PostgreSQL) environments.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    
    # Flask
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    # Database
    USE_POSTGRES: bool = os.getenv("USE_POSTGRES", "false").lower() == "true"
    DB_URI: str = os.getenv(
        "DB_URI", 
        "postgresql://mission7:mission7pass@postgres:5432/credit_scoring"
    )
    DB_PATH: str = os.getenv("DB_PATH", "/app/dataset/home_credit.db")
    
    # Database Pool Settings
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-dev:5005")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "CreditScoring_BestModel")
    
    # MLflow Serving (optional - for custom PyFunc with SHAP)
    USE_MLFLOW_SERVING: bool = os.getenv("USE_MLFLOW_SERVING", "false").lower() == "true"
    MLFLOW_SERVING_URI: str = os.getenv("MLFLOW_SERVING_URI", "http://mlflow-serve:5003")
    
    # Production Model Paths
    PROD_MODEL_PATH: str = os.getenv("PROD_MODEL_PATH", "/app/prod_models/model.pkl")
    PROD_THRESHOLD_PATH: str = os.getenv("PROD_THRESHOLD_PATH", "/app/prod_models/threshold.json")
    PROD_METADATA_PATH: str = os.getenv("PROD_METADATA_PATH", "/app/prod_models/metadata.json")
    PROD_FEATURES_PATH: str = os.getenv("PROD_FEATURES_PATH", "/app/prod_models/feature_names.txt")
    
    # Business Rules
    DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", "0.45"))
    FN_FP_COST_RATIO: int = int(os.getenv("FN_FP_COST_RATIO", "10"))
    
    # API Settings
    API_PREFIX: str = "/api"
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10MB


# Singleton config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the configuration singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def get_flask_config() -> dict:
    """Get configuration as dict for Flask app.config.update()."""
    config = get_config()
    return {
        'SECRET_KEY': config.SECRET_KEY,
        'DEBUG': config.DEBUG,
        'MAX_CONTENT_LENGTH': config.MAX_CONTENT_LENGTH,
        'USE_POSTGRES': config.USE_POSTGRES,
        'MLFLOW_TRACKING_URI': config.MLFLOW_TRACKING_URI,
    }
