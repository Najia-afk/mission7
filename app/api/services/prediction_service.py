# app/api/services/prediction_service.py
"""
Prediction service for credit scoring.
Handles model inference, SHAP explanations, and prediction logging.
Supports both local model and MLflow Serving modes.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import traceback
import json

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
        
        Supports two modes:
        1. Local mode (default): Load pickle, compute SHAP locally
        2. MLflow Serving mode: Call HTTP endpoint with custom PyFunc that includes SHAP
        
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
            
            # 3. Prepare features
            X = df_processed.drop(columns=['TARGET'], errors='ignore')
            X_features = X.drop(columns=['SK_ID_CURR'], errors='ignore')
            
            # 4. Check if MLflow Serving is enabled
            if self.model_service.is_mlflow_serving_enabled():
                return self._predict_via_mlflow_serving(client_id, X_features)
            else:
                return self._predict_locally(client_id, X_features)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "status_code": 500
            }
    
    def _predict_via_mlflow_serving(
        self, 
        client_id: Optional[int], 
        X_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Make prediction via MLflow Serving endpoint."""
        logger.info(f"Using MLflow Serving for prediction (client: {client_id or 'manual'})")
        
        result = self.model_service.predict_with_mlflow_serving(X_features)
        
        if "error" in result:
            # Fallback to local prediction if MLflow Serving fails
            logger.warning(f"MLflow Serving failed, falling back to local: {result['error']}")
            return self._predict_locally(client_id, X_features)
        
        y_proba = result["probability"]
        threshold = result["threshold"]
        decision = result["decision"]
        shap_explanation = result.get("shap_explanation")
        
        # Generate SHAP HTML from the structured explanation
        shap_html = self._shap_explanation_to_html(shap_explanation) if shap_explanation else None
        
        # Fallback to local SHAP if not returned by MLflow Serving
        if not shap_html:
            model, _ = self.model_service.get_production_model()
            if model:
                shap_html, _ = self._compute_shap_with_values(model, X_features)
        
        # Extract top features dict for logging
        shap_dict = None
        if shap_explanation and "top_features" in shap_explanation:
            shap_dict = {
                feat["technical_name"]: feat["shap_value"] 
                for feat in shap_explanation["top_features"]
            }
        
        # Log prediction
        log_prediction_to_postgres(
            client_id=client_id if client_id else 0,
            probability=float(y_proba),
            threshold=threshold,
            decision=decision,
            shap_values=shap_dict
        )
        
        logger.info(f"MLflow Serving prediction for {client_id or 'manual'}: {decision} ({y_proba:.4f})")
        
        return {
            "client_id": client_id or "Manual Input",
            "probability": float(y_proba),
            "threshold": threshold,
            "decision": decision,
            "shap_html": shap_html,
            "shap_features": shap_explanation.get("top_features") if shap_explanation else None,
            "source": "mlflow_serving"
        }
    
    def _predict_locally(
        self, 
        client_id: Optional[int], 
        X_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Make prediction using locally loaded model."""
        # Load Model
        model, threshold = self.model_service.get_production_model()
        if model is None:
            return {
                "error": "Production model not found.",
                "status_code": 500
            }
        
        # Predict
        y_proba = model.predict_proba(X_features)[:, 1][0]
        decision = "REJECTED" if y_proba >= threshold else "ACCEPTED"
        
        # SHAP Explanation (get both HTML and values for logging)
        shap_html, shap_dict = self._compute_shap_with_values(model, X_features)
        
        # Log prediction with SHAP values for drift monitoring
        log_prediction_to_postgres(
            client_id=client_id if client_id else 0,
            probability=float(y_proba),
            threshold=threshold,
            decision=decision,
            shap_values=shap_dict
        )
        
        logger.info(f"Local prediction for {client_id or 'manual'}: {decision} ({y_proba:.4f})")
        
        return {
            "client_id": client_id or "Manual Input",
            "probability": float(y_proba),
            "threshold": threshold,
            "decision": decision,
            "shap_html": shap_html,
            "source": "local"
        }
    
    def _shap_explanation_to_html(self, shap_explanation: Dict) -> str:
        """Convert structured SHAP explanation from MLflow Serving to HTML visualization."""
        if not shap_explanation or "error" in shap_explanation:
            return "<p>SHAP explanation not available.</p>"
        
        top_features = shap_explanation.get("top_features", [])
        if not top_features:
            return "<p>No feature importance data available.</p>"
        
        # Build HTML bar chart similar to local SHAP visualization
        expected_value = shap_explanation.get("expected_value", 0)
        
        html_parts = [
            '<div class="shap-explanation" style="font-family: Arial, sans-serif;">',
            f'<h4>Feature Impact on Risk Score</h4>',
            f'<p style="font-size: 12px; color: #666;">Base value: {expected_value:.3f}</p>',
            '<div style="display: flex; flex-direction: column; gap: 8px;">'
        ]
        
        # Find max absolute SHAP for scaling
        max_shap = max(abs(f["shap_value"]) for f in top_features) if top_features else 1
        
        for feat in reversed(top_features[:15]):  # Show top 15, reversed for bottom-up
            shap_val = feat["shap_value"]
            width_pct = min(abs(shap_val) / max_shap * 100, 100)
            color = "#d62728" if shap_val > 0 else "#1f77b4"  # Red = increases risk, Blue = decreases
            direction = "→" if shap_val > 0 else "←"
            
            html_parts.append(f'''
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 200px; text-align: right; font-size: 12px;">
                        <b>{feat["value"]}</b> | {feat["feature"]}
                    </div>
                    <div style="width: 150px; height: 20px; background: #f0f0f0; position: relative;">
                        <div style="position: absolute; {'right' if shap_val < 0 else 'left'}: 50%; 
                                    width: {width_pct/2}%; height: 100%; background: {color};"></div>
                    </div>
                    <div style="width: 60px; font-size: 11px; color: {color};">
                        {direction} {abs(shap_val):.4f}
                    </div>
                </div>
            ''')
        
        html_parts.append('</div>')
        html_parts.append('<p style="font-size: 11px; color: #888; margin-top: 10px;">')
        html_parts.append('<span style="color: #d62728;">■</span> Increases default risk | ')
        html_parts.append('<span style="color: #1f77b4;">■</span> Decreases default risk</p>')
        html_parts.append('</div>')
        
        return ''.join(html_parts)
    
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
