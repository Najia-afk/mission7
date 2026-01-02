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


@api_bp.route('/history')
def history_page():
    """Render prediction history page."""
    return render_template('history.html')


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


@api_bp.route('/api/client/<int:client_id>/similar')
def get_similar_clients(client_id: int):
    """
    Find similar clients for comparison analysis
    ---
    tags:
      - Clients
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Reference client ID
      - name: n_neighbors
        in: query
        type: integer
        default: 10
        description: Number of similar clients to return
    responses:
      200:
        description: Similar clients found
        schema:
          type: object
          properties:
            client_id:
              type: integer
            n_similar:
              type: integer
            comparison_stats:
              type: object
            similar_default_rate:
              type: number
            population_default_rate:
              type: number
      404:
        description: Client not found
    """
    n_neighbors = request.args.get('n_neighbors', 10, type=int)
    result = client_service.get_similar_clients(client_id, n_neighbors)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@api_bp.route('/api/analysis/bivariate')
def get_bivariate_data():
    """
    Get data for bi-variate scatter plot analysis
    ---
    tags:
      - Analysis
    parameters:
      - name: feature_x
        in: query
        type: string
        required: true
        description: Feature for X axis
      - name: feature_y
        in: query
        type: string
        required: true
        description: Feature for Y axis
      - name: client_id
        in: query
        type: integer
        required: false
        description: Client ID to highlight
      - name: sample_size
        in: query
        type: integer
        default: 1000
        description: Number of data points to return
    responses:
      200:
        description: Bi-variate data for scatter plot
        schema:
          type: object
          properties:
            feature_x:
              type: string
            feature_y:
              type: string
            accepted:
              type: object
            rejected:
              type: object
            highlight:
              type: object
      400:
        description: Invalid feature selection
    """
    feature_x = request.args.get('feature_x')
    feature_y = request.args.get('feature_y')
    client_id = request.args.get('client_id', type=int)
    sample_size = request.args.get('sample_size', 1000, type=int)
    
    if not feature_x or not feature_y:
        return jsonify({"error": "feature_x and feature_y are required"}), 400
    
    result = client_service.get_bivariate_data(feature_x, feature_y, client_id, sample_size)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@api_bp.route('/api/analysis/features')
def get_available_features():
    """
    Get list of available features for analysis
    ---
    tags:
      - Analysis
    responses:
      200:
        description: List of available feature names
        schema:
          type: array
          items:
            type: string
    """
    return jsonify(client_service.get_available_features())


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


# =============================================================================
# PREDICTION HISTORY
# =============================================================================

@api_bp.route('/api/predictions/search')
def search_predictions_api():
    """
    Search prediction history with filters
    ---
    tags:
      - Predictions
    parameters:
      - name: client_id
        in: query
        type: integer
        required: false
        description: Filter by client ID
      - name: decision
        in: query
        type: string
        enum: [ACCEPTED, REJECTED]
        required: false
        description: Filter by decision
      - name: min_score
        in: query
        type: number
        required: false
        description: Minimum probability score (0-1)
      - name: max_score
        in: query
        type: number
        required: false
        description: Maximum probability score (0-1)
      - name: start_date
        in: query
        type: string
        required: false
        description: Start date (ISO format)
      - name: end_date
        in: query
        type: string
        required: false
        description: End date (ISO format)
      - name: limit
        in: query
        type: integer
        default: 50
        description: Max results per page
      - name: offset
        in: query
        type: integer
        default: 0
        description: Pagination offset
    responses:
      200:
        description: Search results
        schema:
          type: object
          properties:
            predictions:
              type: array
            total:
              type: integer
            limit:
              type: integer
            offset:
              type: integer
            has_more:
              type: boolean
    """
    from app.utils.database import search_predictions
    
    client_id = request.args.get('client_id', type=int)
    decision = request.args.get('decision')
    min_score = request.args.get('min_score', type=float)
    max_score = request.args.get('max_score', type=float)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    result = search_predictions(
        client_id=client_id,
        decision=decision,
        min_score=min_score,
        max_score=max_score,
        start_date=start_date,
        end_date=end_date,
        limit=min(limit, 500),  # Cap at 500
        offset=offset
    )
    
    return jsonify(result)


@api_bp.route('/api/predictions/stats')
def prediction_stats_api():
    """
    Get prediction statistics
    ---
    tags:
      - Predictions
    responses:
      200:
        description: Prediction statistics
        schema:
          type: object
          properties:
            total:
              type: integer
            accepted:
              type: integer
            rejected:
              type: integer
            approval_rate:
              type: number
            avg_score:
              type: number
    """
    from app.utils.database import get_prediction_stats
    return jsonify(get_prediction_stats())
