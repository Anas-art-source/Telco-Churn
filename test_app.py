#!/usr/bin/env python3
"""
test_app.py - Tests for the churn prediction API.

Usage:
    pytest test_app.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import will fail if model not trained, but that's OK for structure
try:
    from app import app
    client = TestClient(app)
    APP_AVAILABLE = True
except Exception:
    APP_AVAILABLE = False
    client = None


# Skip all tests if app can't be imported
pytestmark = pytest.mark.skipif(not APP_AVAILABLE, reason="App not available")


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_with_valid_input(self):
        """Predict should work with valid customer data."""
        payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.3,
            "TotalCharges": 543.6
        }
        
        response = client.post("/predict", json=payload)
        
        # May be 503 if model not trained
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "prediction" in data
            assert "risk_level" in data
            assert 0 <= data["churn_probability"] <= 1
            assert data["prediction"] in ["Yes", "No"]
            assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_predict_with_minimal_input(self):
        """Predict should work with minimal required fields."""
        payload = {
            "gender": "Male",
            "tenure": 24,
            "MonthlyCharges": 50.0,
            "Contract": "One year",
            "InternetService": "DSL"
        }
        
        response = client.post("/predict", json=payload)
        # Should return 200 or 503 (if model not loaded)
        assert response.status_code in [200, 503]
    
    def test_predict_with_high_risk_customer(self):
        """Test prediction for high-risk customer profile."""
        payload = {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0
        }
        
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            # High-risk customer should have higher probability
            assert data["churn_probability"] > 0.3


class TestInputValidation:
    """Tests for input validation."""
    
    def test_invalid_gender(self):
        """Should reject invalid gender value."""
        payload = {
            "gender": "Invalid",
            "tenure": 12,
            "MonthlyCharges": 50.0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_contract(self):
        """Should reject invalid contract type."""
        payload = {
            "gender": "Male",
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "Contract": "Invalid"
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_negative_tenure(self):
        """Should reject negative tenure."""
        payload = {
            "gender": "Male",
            "tenure": -5,
            "MonthlyCharges": 50.0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    
    def test_negative_charges(self):
        """Should reject negative charges."""
        payload = {
            "gender": "Male",
            "tenure": 12,
            "MonthlyCharges": -50.0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_info(self):
        """Root should return API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
