# app/api/__init__.py
from app.api.routes import api_bp
from app.api.audit_routes import audit_bp

__all__ = ['api_bp', 'audit_bp']
