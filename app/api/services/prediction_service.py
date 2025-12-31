# app/api/services/prediction_service.py
"""
Prediction service for credit scoring.
Handles model inference, SHAP explanations, and prediction logging.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import traceback

from app.api.services.model_service import ModelService
from app.api.services.client_service import ClientService
from app.utils.database import log_prediction_to_postgres
from app.utils.logging_config import setup_logging

# Import from exploration classes (reuse existing code)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from classes.feature_engineering import FeatureEngineering
from classes.model_visualizer import ModelVisualizer

logger = setup_logging('prediction')


class PredictionService:
    """Service for making credit risk predictions with SHAP explanations."""
    
    def __init__(self):
        self.model_service = ModelService()
        self.client_service = ClientService()
        self.feature_engineering = FeatureEngineering()
        self.visualizer = ModelVisualizer()
    
    def predict(
        self,
        client_id: Optional[int] = None,
        manual_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a prediction for a client.
        
        Args:
            client_id: Client ID to look up in database
            manual_features: Manual features for what-if analysis
            
        Returns:
            Dict with prediction results or error
        """
        try:
            # 1. Load/Prepare Data
            if manual_features:
                df_client = self._prepare_manual_features(manual_features)
            else:
                df_client = self.client_service.get_client_dataframe(client_id)
                if df_client.empty:
                    return {
                        "error": f"Client {client_id} not found in database.",
                        "status_code": 404
                    }
            
            # 2. Feature Engineering
            df_processed = self.feature_engineering.simple_feature_engineering(df_client)
            
            # 3. Load Model
            model, threshold = self.model_service.get_production_model()
            if model is None:
                return {
                    "error": "Production model not found.",
                    "status_code": 500
                }
            
            # 4. Predict
            X = df_processed.drop(columns=['TARGET'], errors='ignore')
            X_features = X.drop(columns=['SK_ID_CURR'], errors='ignore')
            
            y_proba = model.predict_proba(X_features)[:, 1][0]
            decision = "REJECTED" if y_proba >= threshold else "ACCEPTED"
            
            # 5. SHAP Explanation (get both HTML and values for logging)
            shap_html, shap_dict = self._compute_shap_with_values(model, X_features)
            
            # 6. Log prediction with SHAP values for drift monitoring
            log_prediction_to_postgres(
                client_id=client_id if client_id else 0,
                probability=float(y_proba),
                threshold=threshold,
                decision=decision,
                shap_values=shap_dict  # Store top SHAP values as JSON
            )
            
            logger.info(f"Prediction for {client_id or 'manual'}: {decision} ({y_proba:.4f})")
            
            return {
                "client_id": client_id or "Manual Input",
                "probability": float(y_proba),
                "threshold": threshold,
                "decision": decision,
                "shap_html": shap_html
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "status_code": 500
            }
    
    def _prepare_manual_features(self, manual_features: Dict[str, Any]) -> pd.DataFrame:
        """Prepare manual features into a DataFrame with all required columns."""
        all_cols = self.client_service.get_all_columns()
        
        # Create template with all columns as NaN
        df_template = pd.DataFrame(columns=all_cols)
        df_template.loc[0] = [np.nan] * len(all_cols)
        
        # Update with manual features
        for key, value in manual_features.items():
            if key in df_template.columns:
                df_template.at[0, key] = value
        
        return df_template
    
    def _compute_shap(self, model, X_features: pd.DataFrame) -> str:
        """Compute SHAP explanation and return as HTML."""
        try:
            shap_data = self.visualizer.compute_shap_values(model, X_features, sample_size=1)
            if shap_data:
                fig_local = self.visualizer.plot_shap_local(shap_data, sample_idx=0)
                return fig_local.to_html(full_html=False, include_plotlyjs=False)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
        
        return "<p>SHAP explanation not available for this model type.</p>"
    
    def _compute_shap_with_values(self, model, X_features: pd.DataFrame) -> tuple:
        """
        Compute SHAP explanation and return both HTML and dict of top values.
        
        Returns:
            Tuple of (shap_html: str, shap_dict: dict with top 15 feature:value pairs)
        """
        shap_dict = None
        shap_html = "<p>SHAP explanation not available for this model type.</p>"
        
        try:
            shap_data = self.visualizer.compute_shap_values(model, X_features, sample_size=1)
            if shap_data:
                # Generate HTML plot
                fig_local = self.visualizer.plot_shap_local(shap_data, sample_idx=0)
                shap_html = fig_local.to_html(full_html=False, include_plotlyjs=False)
                
                # Extract top SHAP values as dict for logging
                shap_values = shap_data['shap_values']
                feature_names = shap_data['feature_names']
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    vals = shap_values[0] if len(shap_values) > 0 else shap_values
                else:
                    vals = shap_values
                
                if hasattr(vals, 'values'):
                    vals = vals.values
                
                # Get single sample values
                if len(vals.shape) > 1:
                    vals = vals[0]
                
                # Sort by absolute value and take top 15
                indices = np.argsort(np.abs(vals))[::-1][:15]
                shap_dict = {
                    feature_names[i]: round(float(vals[i]), 6) 
                    for i in indices
                }
                
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
        
        return shap_html, shap_dict
