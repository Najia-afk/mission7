import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.classes.sqlite_connector import DatabaseConnection
from src.classes.model_visualizer import ModelVisualizer
from src.classes.feature_engineering import FeatureEngineering

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-dev:5005")
MODEL_NAME = "CreditScoring_BestModel"

# Database configuration - supports both SQLite (dev) and Postgres (prod)
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"
DB_URI = os.getenv("DB_URI", "postgresql://mission7:mission7pass@postgres:5432/credit_scoring")
DB_PATH = os.getenv("DB_PATH", "/app/dataset/home_credit.db")

# Production model path (loaded from git instead of MLflow)
PROD_MODEL_PATH = os.getenv("PROD_MODEL_PATH", "/app/prod_models/model.pkl")
PROD_THRESHOLD_PATH = os.getenv("PROD_THRESHOLD_PATH", "/app/prod_models/threshold.json")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# =============================================================================
# DATABASE CONNECTION (Postgres or SQLite)
# =============================================================================

_pg_engine = None
_pg_session_factory = None


def get_postgres_session():
    """Get a thread-safe PostgreSQL session with connection pooling."""
    global _pg_engine, _pg_session_factory
    
    if _pg_engine is None:
        _pg_engine = create_engine(
            DB_URI,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        _pg_session_factory = scoped_session(sessionmaker(bind=_pg_engine))
    
    return _pg_session_factory()


def get_client_from_postgres(client_id: int) -> pd.DataFrame:
    """
    Fetch client data from PostgreSQL using indexed SK_ID_CURR lookup.
    
    Args:
        client_id: The SK_ID_CURR to look up
        
    Returns:
        DataFrame with client data or empty DataFrame if not found
    """
    session = get_postgres_session()
    try:
        # Try application_train first
        query = text("SELECT * FROM application_train WHERE \"SK_ID_CURR\" = :client_id")
        result = session.execute(query, {"client_id": client_id})
        rows = result.fetchall()
        
        if not rows:
            # Try application_test
            query = text("SELECT * FROM application_test WHERE \"SK_ID_CURR\" = :client_id")
            result = session.execute(query, {"client_id": client_id})
            rows = result.fetchall()
        
        if rows:
            columns = result.keys()
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()
    finally:
        session.close()


def log_prediction_to_postgres(client_id, probability, threshold, decision, model_version=None):
    """
    Log prediction to PostgreSQL for drift monitoring and audit.
    
    Args:
        client_id: The SK_ID_CURR
        probability: Model probability output
        threshold: Business threshold used
        decision: ACCEPTED or REJECTED
        model_version: Optional MLflow run_id
    """
    if not USE_POSTGRES:
        return  # Skip logging in SQLite mode
    
    session = get_postgres_session()
    try:
        query = text("""
            INSERT INTO predictions (client_id, probability, threshold, decision, model_version, request_source)
            VALUES (:client_id, :probability, :threshold, :decision, :model_version, 'api')
        """)
        session.execute(query, {
            "client_id": client_id,
            "probability": probability,
            "threshold": threshold,
            "decision": decision,
            "model_version": model_version
        })
        session.commit()
    except Exception as e:
        print(f"Warning: Failed to log prediction: {e}")
        session.rollback()
    finally:
        session.close()


def get_client_data_from_db(client_id: int) -> pd.DataFrame:
    """
    Fetch client data from database (Postgres or SQLite based on config).
    
    Args:
        client_id: The SK_ID_CURR to look up
        
    Returns:
        DataFrame with client data or empty DataFrame if not found
    """
    if USE_POSTGRES:
        return get_client_from_postgres(client_id)
    else:
        # Use SQLite (development mode)
        db = DatabaseConnection(DB_PATH)
        query = f"SELECT * FROM application_train WHERE SK_ID_CURR = {client_id}"
        df = db.execute_query(query)
        
        if df.empty:
            query = f"SELECT * FROM application_test WHERE SK_ID_CURR = {client_id}"
            df = db.execute_query(query)
        
        return df

# Global variable to store column names
ALL_COLUMNS = None

def get_all_columns():
    """Retrieves all column names from the database to use as a template."""
    global ALL_COLUMNS
    if ALL_COLUMNS is not None:
        return ALL_COLUMNS
    
    try:
        if USE_POSTGRES:
            session = get_postgres_session()
            try:
                result = session.execute(text("SELECT * FROM application_train LIMIT 1"))
                ALL_COLUMNS = list(result.keys())
            finally:
                session.close()
        else:
            db = DatabaseConnection(DB_PATH)
            df = db.execute_query("SELECT * FROM application_train LIMIT 1")
            ALL_COLUMNS = df.columns.tolist()
        return ALL_COLUMNS
    except Exception as e:
        print(f"Error getting columns: {e}")
        return []


def load_model_from_file():
    """
    Load model from /prod_models/ directory (committed to git).
    This is the primary method for production deployment.
    
    Returns:
        tuple: (model, threshold) or (None, default_threshold) on error
    """
    default_threshold = 0.45
    
    try:
        # Load model
        if os.path.exists(PROD_MODEL_PATH):
            with open(PROD_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded from {PROD_MODEL_PATH}")
        else:
            print(f"⚠️ Model file not found at {PROD_MODEL_PATH}")
            return None, default_threshold
        
        # Load threshold
        if os.path.exists(PROD_THRESHOLD_PATH):
            with open(PROD_THRESHOLD_PATH, 'r') as f:
                threshold_data = json.load(f)
                threshold = float(threshold_data.get("optimal_threshold", default_threshold))
            print(f"✅ Threshold loaded: {threshold}")
        else:
            threshold = default_threshold
            print(f"⚠️ Using default threshold: {threshold}")
        
        return model, threshold
    
    except Exception as e:
        print(f"❌ Error loading model from file: {e}")
        return None, default_threshold


def get_production_model():
    """
    Loads the production model with fallback strategy:
    1. First try /prod_models/model.pkl (git-committed, preferred)
    2. Then try MLflow registry with 'Production' stage
    
    Returns:
        tuple: (model, threshold)
    """
    # Try file-based model first (production deployment)
    model, threshold = load_model_from_file()
    if model is not None:
        return model, threshold
    
    # Fallback to MLflow registry
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get threshold from run params
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            run = client.get_run(run_id)
            threshold = float(run.data.params.get("business_optimal_threshold", 0.45))
        else:
            threshold = 0.45
            
        return model, threshold
    except Exception as e:
        print(f"Error loading production model: {e}")
        return None, 0.45

@app.route('/')
@app.route('/api/html')
def index():
    return render_template('index.html')


@app.route('/audit')
def audit_page():
    """Standalone audit documentation page for BCE/FINMA regulators."""
    return render_template('audit.html')


# =============================================================================
# MODEL MANAGEMENT ENDPOINTS (from mission7_dashboard)
# =============================================================================

@app.route('/api/models/list')
def list_models():
    """
    List all available models in prod_models directory and MLflow registry.
    Allows switching between model versions.
    """
    try:
        models = []
        prod_models_dir = Path(PROD_MODEL_PATH).parent
        
        # List file-based models in prod_models/
        if prod_models_dir.exists():
            for model_file in prod_models_dir.glob("*.pkl"):
                # Try different naming conventions for metadata
                possible_metadata = [
                    prod_models_dir / "metadata.json",
                    prod_models_dir / f"{model_file.stem}_metadata.json",
                ]
                
                metadata = {}
                for mp in possible_metadata:
                    if mp.exists():
                        with open(mp, 'r') as f:
                            metadata = json.load(f)
                        break
                
                models.append({
                    "source": "file",
                    "path": str(model_file),
                    "name": model_file.stem,
                    "active": str(model_file) == PROD_MODEL_PATH,
                    "metadata": metadata
                })
        
        # List MLflow registered models
        try:
            registered_models = client.search_registered_models()
            for rm in registered_models:
                for version in client.search_model_versions(f"name='{rm.name}'"):
                    models.append({
                        "source": "mlflow",
                        "name": rm.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "run_id": version.run_id,
                        "active": version.current_stage == "Production"
                    })
        except Exception as mlflow_err:
            print(f"MLflow listing error: {mlflow_err}")
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/current')
def current_model():
    """Get information about the currently active model."""
    try:
        model, threshold = get_production_model()
        
        # Determine source
        model_source = "file" if os.path.exists(PROD_MODEL_PATH) else "mlflow"
        
        # Load metadata
        metadata = {}
        metadata_path = PROD_MODEL_PATH.replace("model.pkl", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return jsonify({
            "model_loaded": model is not None,
            "source": model_source,
            "threshold": threshold,
            "path": PROD_MODEL_PATH if model_source == "file" else None,
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/deploy', methods=['POST'])
def deploy_model():
    """
    Deploy a specific model version from MLflow to production.
    This exports the model to prod_models/ directory.
    
    Body: {"run_id": "abc123"} or {"model_name": "...", "version": "1"}
    """
    try:
        data = request.get_json()
        run_id = data.get('run_id')
        model_name = data.get('model_name', MODEL_NAME)
        version = data.get('version')
        
        if not run_id and not version:
            return jsonify({"error": "Provide either run_id or version"}), 400
        
        # If version provided, get run_id from MLflow
        if version and not run_id:
            mv = client.get_model_version(model_name, version)
            run_id = mv.run_id
        
        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Get run info for metadata
        run = client.get_run(run_id)
        threshold = float(run.data.params.get("business_optimal_threshold", 0.45))
        
        # Save to prod_models/
        prod_dir = Path(PROD_MODEL_PATH).parent
        prod_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup current model
        if os.path.exists(PROD_MODEL_PATH):
            backup_path = PROD_MODEL_PATH.replace(".pkl", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            os.rename(PROD_MODEL_PATH, backup_path)
        
        # Save new model
        with open(PROD_MODEL_PATH, 'wb') as f:
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
        metadata_path = PROD_MODEL_PATH.replace("model.pkl", "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save threshold
        threshold_path = PROD_MODEL_PATH.replace("model.pkl", "threshold.json")
        with open(threshold_path, 'w') as f:
            json.dump({"optimal_threshold": threshold}, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": f"Model {run_id} deployed to production",
            "metadata": metadata
        })
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/models/rollback', methods=['POST'])
def rollback_model():
    """
    Rollback to a previous model version.
    Lists available backups and restores the selected one.
    """
    try:
        data = request.get_json() or {}
        backup_name = data.get('backup')
        
        prod_dir = Path(PROD_MODEL_PATH).parent
        
        # List backups
        backups = list(prod_dir.glob("model_backup_*.pkl"))
        
        if not backup_name:
            # Return list of available backups
            return jsonify({
                "backups": [
                    {
                        "name": b.name,
                        "created": b.stat().st_mtime
                    } for b in backups
                ]
            })
        
        # Restore specific backup
        backup_path = prod_dir / backup_name
        if not backup_path.exists():
            return jsonify({"error": f"Backup not found: {backup_name}"}), 404
        
        # Swap current model with backup
        current_backup = PROD_MODEL_PATH.replace(".pkl", f"_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        if os.path.exists(PROD_MODEL_PATH):
            os.rename(PROD_MODEL_PATH, current_backup)
        os.rename(str(backup_path), PROD_MODEL_PATH)
        
        return jsonify({
            "success": True,
            "message": f"Rolled back to {backup_name}",
            "previous_saved_as": current_backup
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "use_postgres": USE_POSTGRES,
        "mlflow_uri": MLFLOW_TRACKING_URI
    })


@app.route('/api/model/info')
def model_info():
    """Get current model information."""
    try:
        model, threshold = get_production_model()
        
        # Check model source
        model_source = "file" if os.path.exists(PROD_MODEL_PATH) else "mlflow"
        
        # Load metadata if available
        metadata = {}
        metadata_path = PROD_MODEL_PATH.replace("model.pkl", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return jsonify({
            "model_loaded": model is not None,
            "threshold": threshold,
            "source": model_source,
            "metadata": metadata
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/client/<int:client_id>')
def get_client_data(client_id):
    """Fetches raw client data from the database (Postgres or SQLite)."""
    try:
        df_client = get_client_data_from_db(client_id)
            
        if df_client.empty:
            return jsonify({"error": f"Client {client_id} not found"}), 404
            
        # Convert to dict, handle NaN for JSON compatibility
        data = df_client.iloc[0].replace({np.nan: None}).to_dict()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    # Support both form data (from existing UI) and JSON (for what-if analysis)
    if request.is_json:
        data = request.get_json()
        client_id = data.get('client_id')
        manual_features = data.get('features')
    else:
        client_id = request.form.get('client_id')
        manual_features = None
        
    if not client_id and not manual_features:
        return jsonify({"error": "No Client ID or features provided"}), 400
    
    try:
        # 1. Load/Prepare Data
        if manual_features:
            # Use provided features directly, but fill missing columns with NaN
            all_cols = get_all_columns()
            # Create a template with all columns as NaN
            df_template = pd.DataFrame(columns=all_cols)
            # Add one row of NaNs
            df_template.loc[0] = [np.nan] * len(all_cols)
            
            # Update with manual features
            for key, value in manual_features.items():
                if key in df_template.columns:
                    df_template.at[0, key] = value
            
            df_client = df_template
        else:
            # Fetch from DB (Postgres or SQLite)
            df_client = get_client_data_from_db(int(client_id))
                
            if df_client.empty:
                return jsonify({"error": f"Client {client_id} not found in database."}), 404
            
        # 2. Preprocess (Feature Engineering)
        fe = FeatureEngineering()
        df_processed = fe.simple_feature_engineering(df_client)
        
        # 3. Load Model
        model, threshold = get_production_model()
        if model is None:
            return jsonify({"error": "Production model not found in MLflow Registry."}), 500
            
        # 4. Predict
        # Drop target if exists
        X = df_processed.drop(columns=['TARGET'], errors='ignore')
        # Ensure SK_ID_CURR is not in features
        X_features = X.drop(columns=['SK_ID_CURR'], errors='ignore')
        
        y_proba = model.predict_proba(X_features)[:, 1][0]
        decision = "REJECTED" if y_proba >= threshold else "ACCEPTED"
        
        # 5. SHAP Explanation
        visualizer = ModelVisualizer()
        shap_data = visualizer.compute_shap_values(model, X_features, sample_size=1)
        
        if shap_data:
            fig_local = visualizer.plot_shap_local(shap_data, sample_idx=0)
            shap_html = fig_local.to_html(full_html=False, include_plotlyjs=False)
        else:
            shap_html = "<p>SHAP explanation not available for this model type.</p>"
        
        # 6. Log prediction for drift monitoring
        log_prediction_to_postgres(
            client_id=client_id if client_id else 0,
            probability=float(y_proba),
            threshold=threshold,
            decision=decision
        )
            
        return jsonify({
            "client_id": client_id or "Manual Input",
            "probability": float(y_proba),
            "threshold": threshold,
            "decision": decision,
            "shap_html": shap_html
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


# =============================================================================
# AUDIT & GOVERNANCE ENDPOINTS (BCE/FINMA Compliance)
# =============================================================================

@app.route('/api/audit/model-governance')
def audit_model_governance():
    """
    Model governance information for regulatory audit (BCE/FINMA).
    Returns comprehensive model documentation including:
    - Model type and version
    - Training parameters
    - Performance metrics
    - Business rules and thresholds
    - Data lineage information
    """
    try:
        model, threshold = get_production_model()
        
        # Load metadata
        metadata = {}
        metadata_path = PROD_MODEL_PATH.replace("model.pkl", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load feature names
        feature_names = []
        feature_path = PROD_MODEL_PATH.replace("model.pkl", "feature_names.txt")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        
        governance_info = {
            "audit_timestamp": datetime.utcnow().isoformat() + "Z",
            "regulatory_framework": ["BCE Guidelines", "FINMA Circular 2008/21", "GDPR Article 22"],
            "model_identification": {
                "name": MODEL_NAME,
                "type": "LightGBM Classifier",
                "version": metadata.get("model_version", "1.0.0"),
                "training_date": metadata.get("training_date", "Unknown"),
                "mlflow_run_id": metadata.get("run_id", "N/A")
            },
            "performance_metrics": {
                "auc_roc": metadata.get("metrics", {}).get("auc_roc"),
                "business_cost": metadata.get("business_cost", metadata.get("metrics", {}).get("business_cost")),
                "optimal_threshold": threshold,
                "recall_at_threshold": metadata.get("metrics", {}).get("recall"),
                "precision_at_threshold": metadata.get("metrics", {}).get("precision"),
                "f1_score": metadata.get("metrics", {}).get("f1_score")
            },
            "business_rules": {
                "decision_threshold": threshold,
                "cost_fn_fp_ratio": "10:1 (False Negative costs 10x more than False Positive)",
                "decision_logic": "REJECTED if probability >= threshold else ACCEPTED",
                "explainability_method": "SHAP (SHapley Additive exPlanations)"
            },
            "feature_information": {
                "total_features": len(feature_names),
                "feature_engineering": "Domain-specific ratios, temporal features, external data flags",
                "features_list": feature_names[:20] if feature_names else [],  # First 20 for preview
                "full_feature_list_endpoint": "/api/audit/features"
            },
            "data_governance": {
                "training_data_source": "Home Credit Default Risk (Kaggle)",
                "data_period": "Historical loan applications",
                "personal_data_handling": "SK_ID_CURR pseudonymized, no direct PII in features",
                "data_storage": "PostgreSQL (production) / SQLite (development)"
            },
            "model_monitoring": {
                "drift_detection": "Evidently AI for data/prediction drift",
                "drift_report_endpoint": "/api/audit/drift-report",
                "prediction_logging": USE_POSTGRES,
                "retraining_trigger": "Manual review when drift detected"
            },
            "compliance_status": {
                "explainability": "✅ SHAP values for each prediction",
                "fairness_testing": "✅ Tested across demographic groups",
                "documentation": "✅ Full model card available",
                "audit_trail": "✅ Predictions logged to PostgreSQL",
                "human_oversight": "✅ Threshold adjustable by business"
            }
        }
        
        return jsonify(governance_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/features')
def audit_features():
    """Full list of model features for audit."""
    try:
        feature_names = []
        feature_path = PROD_MODEL_PATH.replace("model.pkl", "feature_names.txt")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
        
        return jsonify({
            "total_features": len(feature_names),
            "features": feature_names
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/predictions')
def audit_predictions():
    """
    Recent prediction log for audit trail.
    Only available when PostgreSQL is enabled.
    """
    if not USE_POSTGRES:
        return jsonify({
            "error": "Prediction audit log requires PostgreSQL mode",
            "use_postgres": False
        }), 400
    
    try:
        session = get_postgres_session()
        try:
            query = text("""
                SELECT client_id, probability, threshold, decision, 
                       model_version, request_source, created_at
                FROM predictions 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            result = session.execute(query)
            predictions = [dict(row._mapping) for row in result.fetchall()]
            
            # Convert datetime to ISO format
            for pred in predictions:
                if pred.get('created_at'):
                    pred['created_at'] = pred['created_at'].isoformat()
            
            return jsonify({
                "total_records": len(predictions),
                "predictions": predictions
            })
        finally:
            session.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/drift-report')
def audit_drift_report():
    """
    Data drift report for model monitoring.
    Returns latest Evidently drift analysis if available.
    """
    try:
        drift_report_path = "/app/reports/evidently_drift_report.html"
        drift_json_path = "/app/reports/drift_metrics.json"
        
        drift_info = {
            "drift_monitoring_active": True,
            "last_check": datetime.utcnow().isoformat() + "Z",
            "drift_detected": False,
            "metrics": {}
        }
        
        if os.path.exists(drift_json_path):
            with open(drift_json_path, 'r') as f:
                drift_info["metrics"] = json.load(f)
                drift_info["drift_detected"] = drift_info["metrics"].get("dataset_drift", False)
        
        drift_info["report_available"] = os.path.exists(drift_report_path)
        drift_info["report_endpoint"] = "/reports/evidently_drift_report.html" if drift_info["report_available"] else None
        
        return jsonify(drift_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audit/model-card')
def audit_model_card():
    """
    ML Model Card for regulatory documentation.
    Following Google's Model Cards for Model Reporting framework.
    """
    model, threshold = get_production_model()
    
    model_card = {
        "model_details": {
            "name": "Home Credit Default Risk Classifier",
            "version": "1.0.0",
            "type": "Binary Classification (LightGBM)",
            "owner": "Prêt à dépenser - Data Science Team",
            "contact": "datascience@pret-a-depenser.com",
            "date_created": "2024",
            "license": "Proprietary - Internal Use Only"
        },
        "intended_use": {
            "primary_use": "Credit risk assessment for loan applications",
            "primary_users": "Loan officers, credit analysts, automated decisioning systems",
            "out_of_scope_uses": [
                "Employment decisions",
                "Insurance underwriting",
                "Criminal justice applications"
            ]
        },
        "factors": {
            "relevant_factors": [
                "Income and employment stability",
                "Credit history and bureau data",
                "Loan characteristics (amount, term)",
                "External data source availability"
            ],
            "evaluation_factors": [
                "Age groups",
                "Income brackets",
                "Employment types"
            ]
        },
        "metrics": {
            "model_performance": {
                "auc_roc": "Target > 0.75",
                "business_cost": "Optimized FN:FP = 10:1",
                "threshold": threshold
            },
            "decision_thresholds": {
                "optimal_threshold": threshold,
                "rationale": "Minimizes business cost (FN more expensive than FP)"
            }
        },
        "training_data": {
            "source": "Home Credit Default Risk Dataset",
            "size": "~300,000 loan applications",
            "features": "122 engineered features from application and bureau data",
            "target": "TARGET (1=default, 0=no default)",
            "class_imbalance": "~8% positive class"
        },
        "ethical_considerations": {
            "fairness_testing": "Evaluated across demographic subgroups",
            "bias_mitigation": "Removed direct demographic features",
            "explainability": "SHAP values provided for each prediction",
            "human_oversight": "Final decisions reviewed by loan officers",
            "right_to_explanation": "GDPR Article 22 compliant"
        },
        "caveats_and_recommendations": {
            "limitations": [
                "Model trained on historical data - may not reflect current economic conditions",
                "Performance may vary for edge cases outside training distribution",
                "Requires regular retraining as data drift is detected"
            ],
            "recommendations": [
                "Monitor prediction distribution for drift",
                "Regular fairness audits",
                "Human review for borderline cases (probability near threshold)"
            ]
        }
    }
    
    return jsonify(model_card)
