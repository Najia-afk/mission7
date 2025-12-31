import os
import sys
import json
import pickle
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
