#!/usr/bin/env python3
"""
Custom MLflow PyFunc Model Wrapper with SHAP Explanations

This module provides a custom MLflow model wrapper that:
1. Wraps a LightGBM credit scoring model
2. Computes SHAP explanations at inference time
3. Returns human-readable feature names
4. Can be served via MLflow Model Serving

Usage:
    python scripts/register_shap_model.py --register
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import shap

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class CreditScoringModelWithSHAP(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc model that wraps LightGBM with SHAP explanations.
    
    This model returns predictions along with SHAP values for interpretability,
    using human-readable feature names.
    """
    
    def __init__(self, model=None, threshold: float = 0.45, feature_mappings: Dict = None):
        """
        Initialize the wrapper.
        
        Args:
            model: The underlying sklearn pipeline (with LightGBM)
            threshold: Business-optimized decision threshold
            feature_mappings: Dict mapping technical to human-readable names
        """
        self.model = model
        self.threshold = threshold
        self.feature_mappings = feature_mappings or {}
        self.explainer = None
    
    def load_context(self, context):
        """
        Load model artifacts when the model is loaded for serving.
        
        This method is called by MLflow when loading the model.
        """
        # Load the pickled model
        model_path = context.artifacts.get("model_pickle")
        if model_path:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        
        # Load threshold
        threshold_path = context.artifacts.get("threshold_json")
        if threshold_path:
            with open(threshold_path, "r") as f:
                threshold_data = json.load(f)
                self.threshold = float(threshold_data.get("optimal_threshold", 0.45))
        
        # Load feature mappings
        mappings_path = context.artifacts.get("feature_mappings")
        if mappings_path:
            with open(mappings_path, "r") as f:
                self.feature_mappings = json.load(f)
        
        # Initialize SHAP explainer (lazily, on first prediction)
        self.explainer = None
    
    def _init_explainer(self, X_sample: pd.DataFrame):
        """Initialize SHAP TreeExplainer with the model."""
        if self.explainer is not None:
            return
        
        # Extract the actual model from pipeline
        model_obj = self.model
        if hasattr(model_obj, 'best_estimator_'):
            model_obj = model_obj.best_estimator_
        if hasattr(model_obj, 'named_steps'):
            step_name = 'model' if 'model' in model_obj.named_steps else list(model_obj.named_steps.keys())[-1]
            model_obj = model_obj.named_steps[step_name]
        
        self.explainer = shap.TreeExplainer(model_obj)
    
    def _get_human_readable_name(self, technical_name: str) -> str:
        """Convert technical feature name to human-readable format."""
        # Remove ColumnTransformer prefixes
        clean_name = technical_name
        for prefix in ['num__', 'cat__', 'ind__']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        # Remove _origin suffix from indicator columns
        if clean_name.endswith('_origin'):
            clean_name = clean_name[:-7]
        
        # Look up in mappings
        feature_names = self.feature_mappings.get('feature_names', {})
        return feature_names.get(clean_name, clean_name)
    
    def _transform_value_for_display(self, feature_name: str, value: float) -> float:
        """Transform value for human-readable display (e.g., DAYS to Years)."""
        days_columns = self.feature_mappings.get('days_to_years_columns', [])
        
        # Check if this is a DAYS column (negative days to positive years)
        clean_name = feature_name
        for prefix in ['num__', 'cat__', 'ind__']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        if clean_name in days_columns:
            # DAYS_BIRTH = -15000 means ~41 years old
            # Convert: abs(days) / 365.25
            return round(abs(value) / 365.25, 1)
        
        return value
    
    def _compute_shap_values(self, X: pd.DataFrame, X_processed: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """
        Compute SHAP values for predictions.
        
        Returns dict with:
            - shap_values: Raw SHAP values per feature
            - top_features: Top 15 features with human-readable names
            - expected_value: Base value from explainer
        """
        try:
            self._init_explainer(X)
            
            # Convert to DataFrame for explainer
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            X_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X_df)
            
            # Handle list format (binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            
            # Get single sample values (take first row)
            sample_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            sample_features = X_df.iloc[0].values
            
            # Sort by absolute SHAP value and get top 15
            indices = np.argsort(np.abs(sample_shap))[::-1][:15]
            
            top_features = []
            for idx in indices:
                tech_name = feature_names[idx]
                human_name = self._get_human_readable_name(tech_name)
                raw_value = float(sample_features[idx])
                display_value = self._transform_value_for_display(tech_name, raw_value)
                shap_value = float(sample_shap[idx])
                
                top_features.append({
                    "feature": human_name,
                    "technical_name": tech_name,
                    "value": display_value,
                    "raw_value": raw_value,
                    "shap_value": round(shap_value, 6),
                    "impact": "increases_risk" if shap_value > 0 else "decreases_risk"
                })
            
            return {
                "shap_values": {feature_names[i]: float(sample_shap[i]) for i in range(len(feature_names))},
                "top_features": top_features,
                "expected_value": float(expected_value)
            }
            
        except Exception as e:
            return {"error": f"SHAP computation failed: {str(e)}", "top_features": []}
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with SHAP explanations.
        
        Args:
            context: MLflow context (unused but required by interface)
            model_input: DataFrame with features
            
        Returns:
            DataFrame with columns:
                - probability: Default probability
                - decision: ACCEPTED or REJECTED
                - threshold: Decision threshold used
                - shap_explanation: JSON string with SHAP details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_context first.")
        
        # Drop SK_ID_CURR if present (not a feature)
        X = model_input.drop(columns=['SK_ID_CURR'], errors='ignore')
        
        # Get probability predictions
        y_proba = self.model.predict_proba(X)[:, 1]
        decisions = ["REJECTED" if p >= self.threshold else "ACCEPTED" for p in y_proba]
        
        # Compute SHAP for each sample
        # Get processed features from pipeline
        model_obj = self.model
        if hasattr(model_obj, 'best_estimator_'):
            model_obj = model_obj.best_estimator_
        
        shap_explanations = []
        if hasattr(model_obj, 'named_steps') and 'preprocessor' in model_obj.named_steps:
            preprocessor = model_obj.named_steps['preprocessor']
            X_processed = preprocessor.transform(X)
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except:
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
            # Compute SHAP for first sample (batch support can be added later)
            shap_result = self._compute_shap_values(X, X_processed, feature_names)
            shap_explanations.append(json.dumps(shap_result))
        else:
            shap_explanations.append(json.dumps({"error": "Pipeline structure not recognized"}))
        
        # Extend shap_explanations if batch
        while len(shap_explanations) < len(y_proba):
            shap_explanations.append(shap_explanations[0])  # Reuse for batch
        
        return pd.DataFrame({
            "probability": y_proba,
            "decision": decisions,
            "threshold": [self.threshold] * len(y_proba),
            "shap_explanation": shap_explanations
        })


def register_model_to_mlflow(
    model_path: str = "prod_models/model.pkl",
    threshold_path: str = "prod_models/threshold.json",
    mappings_path: str = "app/config/feature_mappings.json",
    model_name: str = "credit-scoring-shap",
    tracking_uri: str = "http://localhost:5002"
):
    """
    Register the custom PyFunc model to MLflow Model Registry.
    
    Args:
        model_path: Path to the pickled model
        threshold_path: Path to threshold JSON
        mappings_path: Path to feature mappings JSON
        model_name: Name in MLflow registry
        tracking_uri: MLflow tracking server URI
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load artifacts to validate
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
        threshold = float(threshold_data.get("optimal_threshold", 0.45))
    
    with open(mappings_path, "r") as f:
        feature_mappings = json.load(f)
    
    print(f"✅ Loaded model from {model_path}")
    print(f"✅ Threshold: {threshold}")
    print(f"✅ Feature mappings: {len(feature_mappings.get('feature_names', {}))} features")
    
    # Create wrapper instance
    wrapper = CreditScoringModelWithSHAP(
        model=model,
        threshold=threshold,
        feature_mappings=feature_mappings
    )
    
    # Create sample input for signature
    # Use a subset of features for signature inference
    sample_columns = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", 
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
    ]
    sample_input = pd.DataFrame({col: [0.0] for col in sample_columns})
    
    # Define artifacts to include
    artifacts = {
        "model_pickle": model_path,
        "threshold_json": threshold_path,
        "feature_mappings": mappings_path
    }
    
    # Create conda environment
    conda_env = {
        "name": "credit_scoring_shap",
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "mlflow>=2.10.0",
                    "shap>=0.45.0",
                    "lightgbm>=4.0.0",
                    "scikit-learn>=1.3.0",
                    "pandas>=2.0.0",
                    "numpy>=1.24.0"
                ]
            }
        ]
    }
    
    # Log and register model
    with mlflow.start_run(run_name="register_shap_model") as run:
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("model_source", model_path)
        mlflow.log_param("features_mapped", len(feature_mappings.get('feature_names', {})))
        
        # Log the custom pyfunc model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            artifacts=artifacts,
            conda_env=conda_env,
            registered_model_name=model_name,
            code_path=[__file__]  # Include this script for reproducibility
        )
        
        print(f"\n🎉 Model registered as '{model_name}'")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Model URI: models:/{model_name}/latest")
    
    return run.info.run_id


def test_model_locally():
    """Test the model wrapper locally before registration."""
    print("🧪 Testing CreditScoringModelWithSHAP locally...\n")
    
    # Load components
    model_path = PROJECT_ROOT / "prod_models" / "model.pkl"
    threshold_path = PROJECT_ROOT / "prod_models" / "threshold.json"
    mappings_path = PROJECT_ROOT / "app" / "config" / "feature_mappings.json"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
        threshold = float(threshold_data.get("optimal_threshold", 0.45))
    
    with open(mappings_path, "r") as f:
        feature_mappings = json.load(f)
    
    # Create wrapper
    wrapper = CreditScoringModelWithSHAP(
        model=model,
        threshold=threshold,
        feature_mappings=feature_mappings
    )
    
    # Create test input (minimal features for testing)
    print("📊 Creating test input with sample data...")
    
    # Load a real sample from database or create synthetic
    try:
        # Try to load from CSV
        train_path = PROJECT_ROOT / "dataset" / "application_train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path, nrows=1)
            df = df.drop(columns=['TARGET'], errors='ignore')
            print(f"   Loaded sample from {train_path}")
        else:
            raise FileNotFoundError("No test data available")
    except Exception as e:
        print(f"   Warning: {e}")
        print("   Using synthetic test data")
        # Create synthetic data with required columns
        df = pd.DataFrame({
            "AMT_INCOME_TOTAL": [150000.0],
            "AMT_CREDIT": [500000.0],
            "AMT_ANNUITY": [25000.0],
            "DAYS_BIRTH": [-15000],
            "DAYS_EMPLOYED": [-2000],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.4]
        })
    
    # Make prediction
    print("\n🔮 Making prediction...")
    result = wrapper.predict(context=None, model_input=df)
    
    print(f"\n📋 Results:")
    print(f"   Probability: {result['probability'].iloc[0]:.4f}")
    print(f"   Decision: {result['decision'].iloc[0]}")
    print(f"   Threshold: {result['threshold'].iloc[0]}")
    
    # Parse SHAP explanation
    shap_data = json.loads(result['shap_explanation'].iloc[0])
    if 'error' not in shap_data:
        print(f"\n🎯 Top SHAP Features:")
        for feat in shap_data.get('top_features', [])[:5]:
            impact = "↑" if feat['impact'] == 'increases_risk' else "↓"
            print(f"   {impact} {feat['feature']}: {feat['value']} (SHAP: {feat['shap_value']:.4f})")
    else:
        print(f"   ⚠️ SHAP error: {shap_data['error']}")
    
    print("\n✅ Local test complete!")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register Credit Scoring Model with SHAP to MLflow")
    parser.add_argument("--test", action="store_true", help="Test model locally before registration")
    parser.add_argument("--register", action="store_true", help="Register model to MLflow")
    parser.add_argument("--tracking-uri", default="http://localhost:5002", help="MLflow tracking URI")
    parser.add_argument("--model-name", default="credit-scoring-shap", help="Model name in registry")
    
    args = parser.parse_args()
    
    if args.test:
        test_model_locally()
    
    if args.register:
        register_model_to_mlflow(
            tracking_uri=args.tracking_uri,
            model_name=args.model_name
        )
    
    if not args.test and not args.register:
        print("Usage: python register_shap_model.py [--test] [--register]")
        print("  --test      Test model locally")
        print("  --register  Register to MLflow Model Registry")
