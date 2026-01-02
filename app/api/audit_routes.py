# app/api/audit_routes.py
"""
Audit and Governance API routes for BCE/FINMA compliance.
Supports both file-based and database-backed artifact retrieval.
"""
import os
import json
from flask import Blueprint, jsonify, send_file, abort, Response, request
from datetime import datetime, timezone

from app.api.services.audit_service import AuditService
from app.config.settings import get_config

audit_bp = Blueprint('audit', __name__, url_prefix='/api/audit')

# Service instance
audit_service = AuditService()


def get_artifact_from_db(model_id, artifact_type):
    """
    Retrieve artifact from PostgreSQL database.
    
    Args:
        model_id: Model's run_id or 'latest' for most recent
        artifact_type: Type of artifact (metadata, drift_report_html, etc.)
    
    Returns:
        dict with artifact data or None
    """
    try:
        # Import here to avoid circular imports
        from app.scripts.sync_artifacts_to_db import get_artifact, get_db_uri
        config = get_config()
        db_uri = config.DB_URI if config.USE_POSTGRES else None
        
        if not db_uri:
            return None
        
        return get_artifact(model_id, artifact_type, db_uri)
    except Exception as e:
        print(f"Error fetching artifact from DB: {e}")
        return None


@audit_bp.route('/model-governance')
def model_governance():
    """
    Model governance documentation
    ---
    tags:
      - Audit
    responses:
      200:
        description: Comprehensive model governance information
        schema:
          type: object
          properties:
            model_inventory:
              type: object
            risk_classification:
              type: object
            validation_framework:
              type: object
            monitoring_framework:
              type: object
    """
    return jsonify(audit_service.get_model_governance())


@audit_bp.route('/model-card')
def model_card():
    """
    ML Model Card
    ---
    tags:
      - Audit
    description: Following Google's Model Cards for Model Reporting framework
    responses:
      200:
        description: Model card with performance metrics and intended use
        schema:
          type: object
          properties:
            model_details:
              type: object
            intended_use:
              type: object
            performance_metrics:
              type: object
            limitations:
              type: object
    """
    return jsonify(audit_service.get_model_card())


@audit_bp.route('/features')
def features():
    """
    List all model features
    ---
    tags:
      - Audit
    responses:
      200:
        description: Complete feature documentation (125 features)
        schema:
          type: object
          properties:
            total_features:
              type: integer
              example: 125
            feature_categories:
              type: object
            features:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  type:
                    type: string
                  category:
                    type: string
                  description:
                    type: string
    """
    return jsonify(audit_service.get_features())


@audit_bp.route('/predictions')
def predictions():
    """
    Prediction audit log
    ---
    tags:
      - Audit
    responses:
      200:
        description: Recent predictions with SHAP explanations
        schema:
          type: object
          properties:
            total_predictions:
              type: integer
            predictions:
              type: array
      400:
        description: PostgreSQL mode required
        schema:
          $ref: '#/definitions/Error'
    """
    config = get_config()
    if not config.USE_POSTGRES:
        return jsonify({
            "error": "Prediction audit log requires PostgreSQL mode",
            "use_postgres": False
        }), 400
    
    return jsonify(audit_service.get_prediction_audit_log())


@audit_bp.route('/drift-report')
def drift_report():
    """
    Data drift monitoring report
    ---
    tags:
      - Audit
    responses:
      200:
        description: Drift analysis summary
        schema:
          type: object
          properties:
            drift_report:
              type: object
              properties:
                status:
                  type: string
                  example: "No drift detected"
                features_monitored:
                  type: integer
                  example: 125
                features_drifted:
                  type: integer
                  example: 0
                html_report_available:
                  type: boolean
                html_report_path:
                  type: string
    """
    return jsonify(audit_service.get_drift_report())


@audit_bp.route('/drift-report-html')
def drift_report_html():
    """
    Evidently HTML drift report (from database)
    ---
    tags:
      - Audit
    produces:
      - text/html
    parameters:
      - name: model_id
        in: query
        type: string
        required: false
        description: Model run_id (defaults to 'latest')
    responses:
      200:
        description: Interactive Evidently drift report
      404:
        description: Report not found
    """
    model_id = request.args.get('model_id', 'latest')
    config = get_config()
    
    # Try database first (preferred for audit compliance)
    if config.USE_POSTGRES:
        artifact = get_artifact_from_db(model_id, 'drift_report_html')
        if artifact and artifact.get('artifact_data'):
            return Response(
                artifact['artifact_data'],
                mimetype='text/html',
                headers={
                    'X-Model-ID': artifact.get('model_id', 'unknown'),
                    'X-Artifact-Name': artifact.get('artifact_name', 'drift_report.html'),
                    'X-Source': 'database'
                }
            )
    
    # Fallback to file system
    prod_models_dir = os.path.dirname(config.PROD_MODEL_PATH)
    html_report_path = os.path.join(prod_models_dir, "evidently_data_drift_report.html")
    if not os.path.exists(html_report_path):
        html_report_path = os.path.join(prod_models_dir, "drift_report.html")
    
    if os.path.exists(html_report_path):
        return send_file(
            html_report_path,
            mimetype='text/html',
            as_attachment=False
        )
    
    abort(404, description="Drift report HTML not found. Run drift analysis first.")


@audit_bp.route('/artifact/<artifact_type>')
@audit_bp.route('/artifact/<artifact_type>/<model_id>')
def get_artifact_endpoint(artifact_type, model_id='latest'):
    """
    Get artifact from database
    ---
    tags:
      - Audit
    parameters:
      - name: artifact_type
        in: path
        type: string
        required: true
        enum: [metadata, threshold, drift_report_html, drift_report_json, feature_names]
        description: Type of artifact to retrieve
      - name: model_id
        in: path
        type: string
        required: false
        description: Model run_id (defaults to 'latest')
    responses:
      200:
        description: Artifact data
      404:
        description: Artifact not found
      400:
        description: PostgreSQL mode required
    """
    config = get_config()
    
    if not config.USE_POSTGRES:
        return jsonify({
            "error": "Database artifact retrieval requires PostgreSQL mode",
            "use_postgres": False
        }), 400
    
    valid_types = ['metadata', 'threshold', 'drift_report_html', 'drift_report_json', 'feature_names']
    if artifact_type not in valid_types:
        return jsonify({
            "error": f"Invalid artifact type. Valid types: {valid_types}"
        }), 400
    
    artifact = get_artifact_from_db(model_id, artifact_type)
    
    if not artifact:
        abort(404, description=f"Artifact '{artifact_type}' not found for model '{model_id}'")
    
    # For HTML artifacts, return as HTML response
    if artifact_type == 'drift_report_html' and artifact.get('artifact_data'):
        return Response(
            artifact['artifact_data'],
            mimetype='text/html',
            headers={
                'Content-Disposition': f'inline; filename="{artifact.get("artifact_name", "report.html")}"',
                'X-Model-ID': artifact.get('model_id', 'unknown'),
                'X-Source': 'database'
            }
        )
    
    # For JSON artifacts, return parsed JSON or raw data
    if artifact.get('artifact_json'):
        return jsonify({
            "model_id": artifact.get('model_id'),
            "artifact_type": artifact_type,
            "artifact_name": artifact.get('artifact_name'),
            "created_at": artifact.get('created_at'),
            "source": "database",
            "data": artifact['artifact_json']
        })
    
    # Return raw data as JSON wrapper
    return jsonify({
        "model_id": artifact.get('model_id'),
        "artifact_type": artifact_type,
        "artifact_name": artifact.get('artifact_name'),
        "created_at": artifact.get('created_at'),
        "source": "database",
        "data": artifact.get('artifact_data')
    })


@audit_bp.route('/download/<artifact_type>')
@audit_bp.route('/download/<artifact_type>/<model_id>')
def download_artifact(artifact_type, model_id='latest'):
    """
    Download artifact as file
    ---
    tags:
      - Audit
    parameters:
      - name: artifact_type
        in: path
        type: string
        required: true
        enum: [drift_report_html, drift_report_json, metadata]
        description: Type of artifact to download
      - name: model_id
        in: path
        type: string
        required: false
        description: Model run_id (defaults to 'latest')
    produces:
      - text/html
      - application/json
    responses:
      200:
        description: File download
      404:
        description: Artifact not found
    """
    config = get_config()
    
    if not config.USE_POSTGRES:
        return jsonify({"error": "PostgreSQL mode required"}), 400
    
    artifact = get_artifact_from_db(model_id, artifact_type)
    
    if not artifact or not artifact.get('artifact_data'):
        abort(404, description=f"Artifact '{artifact_type}' not found")
    
    # Determine content type and filename
    if artifact_type == 'drift_report_html':
        mimetype = 'text/html'
        filename = f"evidently_drift_report_{artifact.get('model_id', 'unknown')}.html"
    else:
        mimetype = 'application/json'
        filename = f"{artifact_type}_{artifact.get('model_id', 'unknown')}.json"
    
    return Response(
        artifact['artifact_data'],
        mimetype=mimetype,
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"',
            'X-Model-ID': artifact.get('model_id', 'unknown'),
            'X-Source': 'database'
        }
    )


@audit_bp.route('/model/<model_id>')
def get_model_info(model_id):
    """
    Get full model information by model_id
    ---
    tags:
      - Audit
    description: Retrieve all artifacts and metadata for a specific model version.
                 Use this to audit historical predictions.
    parameters:
      - name: model_id
        in: path
        type: string
        required: true
        description: Model run_id (from predictions table)
    responses:
      200:
        description: Complete model information
        schema:
          type: object
          properties:
            model_id:
              type: string
            metadata:
              type: object
            threshold:
              type: object
            has_drift_report:
              type: boolean
            has_feature_names:
              type: boolean
      404:
        description: Model not found
    """
    config = get_config()
    
    if not config.USE_POSTGRES:
        return jsonify({"error": "PostgreSQL mode required"}), 400
    
    # Get all artifacts for this model
    artifacts = {}
    artifact_types = ['metadata', 'threshold', 'drift_report_json', 'feature_names']
    
    for artifact_type in artifact_types:
        artifact = get_artifact_from_db(model_id, artifact_type)
        if artifact and artifact.get('artifact_json'):
            artifacts[artifact_type] = artifact['artifact_json']
        elif artifact and artifact.get('artifact_data'):
            try:
                artifacts[artifact_type] = json.loads(artifact['artifact_data'])
            except:
                artifacts[artifact_type] = artifact['artifact_data']
    
    if not artifacts:
        abort(404, description=f"No artifacts found for model_id: {model_id}")
    
    return jsonify({
        "model_id": model_id,
        "metadata": artifacts.get('metadata', {}),
        "threshold": artifacts.get('threshold', {}),
        "has_drift_report": 'drift_report_json' in artifacts,
        "has_feature_names": 'feature_names' in artifacts,
        "drift_report_url": f"/api/audit/download/drift_report_html?model_id={model_id}",
        "download_metadata_url": f"/api/audit/download/metadata?model_id={model_id}"
    })