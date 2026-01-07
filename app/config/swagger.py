# app/config/swagger.py
"""
Swagger/OpenAPI configuration for Mission7 Credit Scoring API.
"""

SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/api/apispec.json',
            "rule_filter": lambda rule: (
                # Include all routes except HTML pages
                not any(rule.rule.startswith(prefix) for prefix in ['/static', '/flasgger_static']) and
                # Include the /predict endpoint and all /api/ endpoints
                (rule.rule.startswith('/api') or rule.rule == '/predict')
            ),
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs"
}

SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "Mission7 Credit Scoring API",
        "description": """
## Credit Risk Assessment API

Production-grade API for credit scoring predictions with full audit trail.

### Features
- **Real-time Predictions**: Get credit risk scores with SHAP explanations
- **Human-readable SHAP**: Feature values displayed intuitively (e.g., "918.5K$", "42.6 yrs")
- **Model Governance**: Full audit trail for regulatory compliance (BCE/FINMA)
- **Feature Documentation**: 125 features documented with categories
- **Drift Monitoring**: Evidently-based data drift reports
- **Download Artifacts**: Export reports for offline audit

### Audit Endpoints
- `/api/audit/model-governance` - Full governance documentation
- `/api/audit/predictions` - Prediction audit log from PostgreSQL
- `/api/audit/download/<type>` - Download artifacts (drift_report_html, drift_report_json, metadata)

### Authentication
Currently open API. Production should add JWT authentication.

### Rate Limits
- 100 requests/minute per IP
- 1000 requests/hour per IP
        """,
        "version": "2.1.0",
        "contact": {
            "name": "Mission7 MLOps Team",
            "email": "mlops@mission7.com"
        },
        "license": {
            "name": "Proprietary",
            "url": "https://mission7.com/license"
        }
    },
    "host": "",  # Will be auto-detected
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "Health",
            "description": "Service health and status endpoints"
        },
        {
            "name": "Predictions",
            "description": "Credit risk prediction endpoints"
        },
        {
            "name": "Clients",
            "description": "Client data retrieval"
        },
        {
            "name": "Models",
            "description": "Model management and deployment"
        },
        {
            "name": "Audit",
            "description": "Regulatory audit and governance endpoints"
        }
    ],
    "definitions": {
        "PredictionRequest": {
            "type": "object",
            "properties": {
                "client_id": {
                    "type": "integer",
                    "description": "Client ID from database",
                    "example": 100002
                },
                "features": {
                    "type": "object",
                    "description": "Manual feature input for what-if analysis",
                    "example": {
                        "EXT_SOURCE_1": 0.5,
                        "EXT_SOURCE_2": 0.6,
                        "EXT_SOURCE_3": 0.7
                    }
                }
            }
        },
        "PredictionResponse": {
            "type": "object",
            "properties": {
                "client_id": {
                    "type": "integer",
                    "example": 100002
                },
                "probability": {
                    "type": "number",
                    "format": "float",
                    "example": 0.234
                },
                "decision": {
                    "type": "string",
                    "enum": ["APPROVED", "REJECTED"],
                    "example": "APPROVED"
                },
                "threshold": {
                    "type": "number",
                    "format": "float",
                    "example": 0.45
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH"],
                    "example": "LOW"
                },
                "shap_values": {
                    "type": "object",
                    "description": "Top feature contributions"
                }
            }
        },
        "HealthResponse": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "example": "healthy"
                },
                "use_postgres": {
                    "type": "boolean",
                    "example": True
                },
                "mlflow_uri": {
                    "type": "string",
                    "example": "http://mlflow-prod:5005"
                }
            }
        },
        "Error": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "example": "Client not found"
                }
            }
        }
    }
}
