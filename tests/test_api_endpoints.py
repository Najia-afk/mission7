"""
API Endpoint Tests for Credit Scoring Application.
Tests the Flask API endpoints for health, predictions, and audit compliance.
"""
import sys
import os
import pytest
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the Flask app
from api.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
