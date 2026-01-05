# app/main.py
"""
Main Flask application factory.
Entry point for production deployment.
"""
import os
from flask import Flask, g
from flask_cors import CORS
from flasgger import Swagger

from app.config.settings import get_config
from app.config.swagger import SWAGGER_CONFIG, SWAGGER_TEMPLATE
from app.api import api_bp, audit_bp
from app.utils.database import get_postgres_session, close_postgres_session
from app.utils.logging_config import setup_logging

logger = setup_logging('app')


def _validate_threshold_configuration() -> float:
    """
    Validate threshold is properly configured at startup.
    Fails fast with clear error if threshold is missing.
    
    Returns:
        The configured threshold value
    """
    from app.api.services.model_service import ModelService
    
    try:
        service = ModelService()
        threshold = service._get_threshold()
        logger.info(f"✅ Threshold configuration validated: {threshold}")
        return threshold
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: Threshold not configured: {e}")
        raise SystemExit(
            f"\n{'='*60}\n"
            f"CRITICAL STARTUP ERROR: Threshold Not Configured\n"
            f"{'='*60}\n"
            f"{e}\n\n"
            f"Please ensure prod_models/metadata.json or prod_models/threshold.json\n"
            f"contains the 'optimal_threshold' field before starting the application.\n"
            f"{'='*60}\n"
        )


def create_app(config_override: dict = None) -> Flask:
    """
    Application factory for Flask app.
    
    Args:
        config_override: Optional config overrides for testing
            - Set 'SKIP_THRESHOLD_VALIDATION': True to skip threshold check (for tests)
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    # Load configuration
    config = get_config()
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = config.DEBUG
    
    # Apply overrides for testing
    if config_override:
        for key, value in config_override.items():
            app.config[key] = value
    
    # Validate threshold configuration at startup (fail fast if missing)
    # Skip validation in test environments or when explicitly disabled
    skip_validation = (
        os.environ.get('SKIP_THRESHOLD_VALIDATION', 'false').lower() == 'true' or
        os.environ.get('TESTING', 'false').lower() == 'true' or
        (config_override and config_override.get('SKIP_THRESHOLD_VALIDATION', False))
    )
    
    if skip_validation:
        logger.warning("⚠️ Threshold validation skipped (test mode)")
        app.config['MODEL_THRESHOLD'] = None
    else:
        threshold = _validate_threshold_configuration()
        app.config['MODEL_THRESHOLD'] = threshold
    
    # Enable CORS
    CORS(app)
    
    # Initialize Swagger
    Swagger(app, config=SWAGGER_CONFIG, template=SWAGGER_TEMPLATE)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(audit_bp)
    
    # Request lifecycle hooks
    @app.before_request
    def before_request():
        """Initialize database session before each request."""
        g.db_session = get_postgres_session()
    
    @app.teardown_request
    def teardown_request(exception=None):
        """Clean up database session after each request."""
        db_session = g.pop('db_session', None)
        if db_session:
            close_postgres_session(db_session)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Resource not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return {"error": "Internal server error"}, 500
    
    logger.info(f"✅ Flask app initialized - Debug: {config.DEBUG}")
    
    return app


# Create app instance for WSGI (skip validation if TESTING env is set)
# This allows tests to import this module without failing
if os.environ.get('TESTING', 'false').lower() != 'true':
    app = create_app()
else:
    app = None  # Tests will create their own app instance


if __name__ == '__main__':
    config = get_config()
    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=config.DEBUG
    )
