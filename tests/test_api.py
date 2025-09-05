"""
Unit tests for API endpoints.
"""

import unittest
from unittest.mock import Mock, patch
import json

from fastapi.testclient import TestClient
from api import app, get_predictor
from config import HouseTypes, EnergyCompanies, WeatherScenarios

class TestAPI(unittest.TestCase):
    """Test API endpoints."""
    
    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)
        
        # Mock predictor
        self.mock_predictor = Mock()
        self.mock_predictor.model_package = {
            'model_performance': {'test_r2': 0.988, 'test_rmse': 157.2}
        }
        self.mock_predictor.predict.return_value = {
            'prediction_kwh': 2500.0,
            'monthly_kwh': 208.3,
            'daily_kwh': 6.8,
            'estimated_monthly_cost': 52.1,
            'model_used': 'Random Forest'
        }
        
        # Override dependency
        app.dependency_overrides[get_predictor] = lambda: self.mock_predictor
    
    def tearDown(self):
        """Clean up after tests."""
        app.dependency_overrides.clear()
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_loaded", data)
        self.assertIn("available_house_types", data)
        self.assertIn("available_companies", data)
        self.assertIn("available_weather_scenarios", data)
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid input."""
        payload = {
            "postal_code": "1012",
            "city": "Amsterdam",
            "house_type": "3x25",
            "smart_meter": True,
            "weather_scenario": "normal"
        }
        
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("prediction_kwh", data)
        self.assertIn("monthly_kwh", data)
        self.assertIn("estimated_monthly_cost", data)
        self.assertIn("model_used", data)
        self.assertIn("comparison_to_average", data)
    
    def test_predict_endpoint_invalid_house_type(self):
        """Test prediction endpoint with invalid house type."""
        payload = {
            "house_type": "invalid_type",
            "weather_scenario": "normal"
        }
        
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_predict_endpoint_invalid_weather(self):
        """Test prediction endpoint with invalid weather scenario."""
        payload = {
            "house_type": "3x25",
            "weather_scenario": "extreme"
        }
        
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_predict_endpoint_invalid_active_connections(self):
        """Test prediction endpoint with invalid active connections percentage."""
        payload = {
            "house_type": "3x25",
            "active_connections_pct": 150  # Out of valid range
        }
        
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    def test_predict_endpoint_minimal_input(self):
        """Test prediction endpoint with minimal input (defaults)."""
        payload = {}
        
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("prediction_kwh", data)
        self.assertIn("input_summary", data)

if __name__ == "__main__":
    unittest.main()
