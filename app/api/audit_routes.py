# app/api/audit_routes.py
"""
Audit and Governance API routes for BCE/FINMA compliance.
"""
import os
from flask import Blueprint, jsonify, send_file, abort
from datetime import datetime, timezone

from app.api.services.audit_service import AuditService
from app.config.settings import get_config

audit_bp = Blueprint('audit', __name__, url_prefix='/api/audit')

# Service instance
audit_service = AuditService()


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
    Evidently HTML drift report
    ---
    tags:
      - Audit
    produces:
      - text/html
    responses:
      200:
        description: Interactive Evidently drift report
      404:
        description: Report not found
    """
    config = get_config()
    prod_models_dir = os.path.dirname(config.PROD_MODEL_PATH)
    # Try both possible filenames for drift report
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
