# app/api/routes.py
"""
Main API routes for Mission7 Credit Scoring.
Handles predictions, client lookups, and model management.
"""
from flask import Blueprint, request, jsonify, render_template
import numpy as np

from app.api.services.prediction_service import PredictionService
from app.api.services.model_service import ModelService
from app.api.services.client_service import ClientService
from app.config.settings import get_config

api_bp = Blueprint('api', __name__)

# Service instances
prediction_service = PredictionService()
model_service = ModelService()
client_service = ClientService()


# =============================================================================
# PAGES
# =============================================================================

@api_bp.route('/')
def index():
    """Render home page."""
    return render_template('index.html')


@api_bp.route('/predict')
@api_bp.route('/api/html')
def predict_page():
    """Render client prediction page (database lookup)."""
    return render_template('predict.html')


@api_bp.route('/predict-test')
def predict_test_page():
    """Render what-if analysis page (feature editor)."""
    return render_template('predict_test.html')


@api_bp.route('/simulator')
def simulator_page():
    """Render end-user simulator page."""
    return render_template('simulator.html')


@api_bp.route('/audit')
def audit_page():
    """Render audit documentation page for BCE/FINMA regulators."""
    return render_template('audit.html')


@api_bp.route('/runs')
def runs_page():
    """Render model runs and deployment page."""
    return render_template('runs.html')


@api_bp.route('/dashboard')
def dashboard_page():
    """Render system monitoring dashboard."""
    return render_template('dashboard.html')


# =============================================================================
# HEALTH & STATUS
# =============================================================================

@api_bp.route('/api/health')
def health():
    """
    Health check endpoint
    ---
    tags:
      - Health
    responses:
      200:
        description: Service is healthy
        schema:
          $ref: '#/definitions/HealthResponse'
    """
    config = get_config()
    return jsonify({
        "status": "healthy",
        "use_postgres": config.USE_POSTGRES,
        "mlflow_uri": config.MLFLOW_TRACKING_URI
    })


@api_bp.route('/api/model/info')
def model_info():
    """
    Get current model information
    ---
    tags:
      - Models
    responses:
      200:
        description: Model metadata and performance metrics
        schema:
          type: object
          properties:
            model_name:
              type: string
              example: "LightGBM"
            version:
              type: string
              example: "2"
            auc_roc:
              type: number
              example: 0.751
            threshold:
              type: number
              example: 0.45
    """
    return jsonify(model_service.get_model_info())


# =============================================================================
# CLIENT DATA
# =============================================================================

@api_bp.route('/api/client/<int:client_id>')
def get_client_data(client_id: int):
    """
    Fetch client data from database
    ---
    tags:
      - Clients
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Client ID
        example: 100002
    responses:
      200:
        description: Client data retrieved successfully
        schema:
          type: object
          properties:
            client_id:
              type: integer
            features:
              type: object
      404:
        description: Client not found
        schema:
          $ref: '#/definitions/Error'
    """
    result = client_service.get_client(client_id)
    if result is None:
        return jsonify({"error": f"Client {client_id} not found"}), 404
    return jsonify(result)


# =============================================================================
# PREDICTIONS
# =============================================================================

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Make a credit risk prediction
    ---
    tags:
      - Predictions
    consumes:
      - application/json
      - application/x-www-form-urlencoded
    parameters:
      - name: body
        in: body
        required: false
        schema:
          $ref: '#/definitions/PredictionRequest'
      - name: client_id
        in: formData
        type: integer
        required: false
        description: Client ID (form submission)
    responses:
      200:
        description: Prediction successful
        schema:
          $ref: '#/definitions/PredictionResponse'
      400:
        description: Invalid request
        schema:
          $ref: '#/definitions/Error'
      404:
        description: Client not found
        schema:
          $ref: '#/definitions/Error'
    """
    # Parse input
    if request.is_json:
        data = request.get_json()
        client_id = data.get('client_id')
        manual_features = data.get('features')
    else:
        client_id = request.form.get('client_id')
        manual_features = None
    
    if not client_id and not manual_features:
        return jsonify({"error": "No Client ID or features provided"}), 400
    
    # Make prediction
    result = prediction_service.predict(
        client_id=int(client_id) if client_id and str(client_id).isdigit() else None,
        manual_features=manual_features
    )
    
    if result.get('error'):
        return jsonify(result), result.get('status_code', 500)
    
    return jsonify(result)


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

@api_bp.route('/api/models/list')
def list_models():
    """
    List all available models
    ---
    tags:
      - Models
    responses:
      200:
        description: List of models from file and MLflow registry
        schema:
          type: object
          properties:
            file_model:
              type: object
            mlflow_models:
              type: array
              items:
                type: object
    """
    return jsonify(model_service.list_models())


@api_bp.route('/api/models/current')
def current_model():
    """
    Get current active model details
    ---
    tags:
      - Models
    responses:
      200:
        description: Current model information
        schema:
          type: object
          properties:
            source:
              type: string
              example: "file"
            model_name:
              type: string
            version:
              type: string
            metrics:
              type: object
    """
    return jsonify(model_service.get_current_model())


@api_bp.route('/api/models/deploy', methods=['POST'])
def deploy_model():
    """
    Deploy a model version to production
    ---
    tags:
      - Models
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            run_id:
              type: string
              description: MLflow run ID
            model_name:
              type: string
              description: Model name in registry
            version:
              type: string
              description: Model version to deploy
    responses:
      200:
        description: Model deployed successfully
      400:
        description: Deployment failed
        schema:
          $ref: '#/definitions/Error'
    """
    data = request.get_json()
    result = model_service.deploy_model(
        run_id=data.get('run_id'),
        model_name=data.get('model_name'),
        version=data.get('version')
    )
    
    if result.get('error'):
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route('/api/models/rollback', methods=['POST'])
def rollback_model():
    """
    Rollback to previous model version
    ---
    tags:
      - Models
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            backup:
              type: string
              description: Backup name to restore
    responses:
      200:
        description: Rollback successful
      404:
        description: Backup not found
        schema:
          $ref: '#/definitions/Error'
    """
    data = request.get_json() or {}
    result = model_service.rollback_model(backup_name=data.get('backup'))
    
    if result.get('error'):
        return jsonify(result), 404
    return jsonify(result)
