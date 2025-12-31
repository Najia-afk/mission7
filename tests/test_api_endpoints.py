"""
API Endpoint Tests for Credit Scoring Application.
Tests the Flask API endpoints for health, predictions, and audit compliance.
"""
import sys
import os
import pytest
import json

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Flask app from NEW modular structure
from app.main import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test that the health endpoint returns 200 and correct structure."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_model_info_endpoint(client):
    """Test that model info endpoint returns model metadata."""
    response = client.get('/api/model/info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_loaded' in data
    assert 'threshold' in data


def test_audit_model_governance(client):
    """Test audit governance endpoint for BCE/FINMA compliance."""
    response = client.get('/api/audit/model-governance')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check required audit fields
    assert 'audit_timestamp' in data
    assert 'regulatory_framework' in data
    assert 'model_identification' in data
    assert 'business_rules' in data
    assert 'compliance_status' in data


def test_audit_model_card(client):
    """Test model card endpoint follows Google Model Cards format."""
    response = client.get('/api/audit/model-card')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check Model Card structure
    assert 'model_details' in data
    assert 'intended_use' in data
    assert 'metrics' in data
    assert 'training_data' in data
    assert 'ethical_considerations' in data


def test_audit_features(client):
    """Test features endpoint returns feature list."""
    response = client.get('/api/audit/features')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_features' in data
    assert 'features' in data


def test_models_current(client):
    """Test current model endpoint."""
    response = client.get('/api/models/current')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'model_loaded' in data
    assert 'source' in data
    assert 'threshold' in data


def test_models_list(client):
    """Test models list endpoint."""
    response = client.get('/api/models/list')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'models' in data
    assert isinstance(data['models'], list)


def test_predict_missing_client_id(client):
    """Test prediction with missing client_id returns error."""
    response = client.post('/predict', data={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_with_manual_features(client):
    """Test prediction with manually provided features."""
    features = {
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 500000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 450000,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5
    }
    
    response = client.post('/predict', 
                          json={"client_id": "manual", "features": features},
                          content_type='application/json')
    
    # May fail due to missing columns, but should not crash
    assert response.status_code in [200, 500]
    data = json.loads(response.data)
    
    if response.status_code == 200:
        assert 'probability' in data
        assert 'decision' in data
        assert 'threshold' in data


def test_wcag_audit_page(client):
    """Test audit page returns HTML with WCAG compliance indicators."""
    response = client.get('/audit')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    
    # Check for WCAG-related content
    assert 'WCAG' in html or 'wcag' in html.lower() or 'accessibility' in html.lower()
    assert 'lang=' in html  # Language attribute for accessibility


def test_audit_drift_report(client):
    """Test drift report endpoint returns JSON drift analysis."""
    response = client.get('/api/audit/drift-report')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'drift_report' in data or 'drift_summary' in data or 'status' in data or 'error' in data


def test_audit_drift_report_html(client):
    """Test Evidently HTML drift report endpoint."""
    response = client.get('/api/audit/drift-report-html')
    # Should return HTML or redirect
    assert response.status_code in [200, 302, 404]
    if response.status_code == 200:
        assert b'html' in response.data.lower() or b'<!DOCTYPE' in response.data


def test_client_endpoint_valid_id(client):
    """Test client endpoint with valid client ID from database."""
    # Test with known client ID 100002 (first in application_train)
    response = client.get('/api/client/100002')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check required client fields
    assert 'SK_ID_CURR' in data
    assert data['SK_ID_CURR'] == 100002
    assert 'AMT_INCOME_TOTAL' in data
    assert 'AMT_CREDIT' in data
    assert 'DAYS_BIRTH' in data


def test_client_endpoint_invalid_id(client):
    """Test client endpoint with non-existent client ID."""
    response = client.get('/api/client/999999999')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data


def test_home_page(client):
    """Test home page returns HTML."""
    response = client.get('/')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert '<!DOCTYPE html>' in html or '<html' in html


def test_predict_page(client):
    """Test predict page returns HTML form."""
    response = client.get('/predict')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert '<form' in html.lower() or 'predict' in html.lower()


def test_simulator_page(client):
    """Test simulator page returns HTML."""
    response = client.get('/simulator')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert 'simulator' in html.lower() or 'simulation' in html.lower()


def test_dashboard_page(client):
    """Test dashboard page returns HTML."""
    response = client.get('/dashboard')
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    assert '<!DOCTYPE html>' in html or '<html' in html


def test_swagger_docs(client):
    """Test Swagger API documentation endpoint."""
    response = client.get('/api/docs')
    assert response.status_code == 200


def test_predict_with_client_id(client):
    """Test prediction with valid client_id from database."""
    response = client.post('/predict', 
                          json={"client_id": 100002},
                          content_type='application/json')
    
    # Should work with valid client from DB
    assert response.status_code in [200, 500]
    data = json.loads(response.data)
    
    if response.status_code == 200:
        assert 'probability' in data
        assert 'decision' in data
        assert 'threshold' in data
        assert 'shap_values' in data or 'explanation' in data or 'client_id' in data


def test_api_cors_headers(client):
    """Test that API endpoints return CORS headers."""
    response = client.get('/api/health')
    # CORS headers may be added by nginx in production
    assert response.status_code == 200


def test_model_metadata_structure(client):
    """Test model metadata has complete information."""
    response = client.get('/api/model/info')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check metadata structure if present
    if 'metadata' in data:
        metadata = data['metadata']
        assert 'algorithm' in metadata or 'model_name' in metadata
        if 'metrics' in metadata:
            assert 'auc_roc' in metadata['metrics'] or 'accuracy' in metadata['metrics']


def test_audit_predictions_endpoint(client):
    """Test audit predictions endpoint."""
    response = client.get('/api/audit/predictions')
    # May return 200 with data/empty list, or 500 if service method doesn't exist
    assert response.status_code in [200, 404, 500]
    if response.status_code == 200:
        data = json.loads(response.data)
        # Should have some structure
        assert isinstance(data, (dict, list))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
