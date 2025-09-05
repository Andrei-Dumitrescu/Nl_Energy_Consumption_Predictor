"""
Unit tests for configuration module.
"""

import unittest
import os
from pathlib import Path

from config import (
    HouseTypes, EnergyCompanies, WeatherScenarios, ModelConstants,
    validate_config, API_PORT, WEATHER_API_TIMEOUT
)

class TestConfig(unittest.TestCase):
    """Test configuration module."""
    
    def test_house_types_constants(self):
        """Test house types constants are properly defined."""
        self.assertIn("1x25", HouseTypes.ALL)
        self.assertIn("3x50", HouseTypes.ALL)
        self.assertEqual(len(HouseTypes.ALL), 5)
    
    def test_energy_companies_constants(self):
        """Test energy companies constants."""
        self.assertIn("liander", EnergyCompanies.ALL)
        self.assertIn("enexis", EnergyCompanies.ALL)
        self.assertEqual(len(EnergyCompanies.ALL), 5)
    
    def test_weather_scenarios_constants(self):
        """Test weather scenarios constants."""
        self.assertIn("cold", WeatherScenarios.ALL)
        self.assertIn("normal", WeatherScenarios.ALL)
        self.assertIn("warm", WeatherScenarios.ALL)
        self.assertEqual(len(WeatherScenarios.ALL), 3)
    
    def test_connection_circuits_map(self):
        """Test connection circuits mapping."""
        self.assertEqual(ModelConstants.CONNECTION_CIRCUITS_MAP["1x25"], 10)
        self.assertEqual(ModelConstants.CONNECTION_CIRCUITS_MAP["3x50"], 22)
    
    def test_weather_parameters(self):
        """Test weather parameters mapping."""
        cold_params = ModelConstants.WEATHER_PARAMETERS["cold"]
        self.assertIn("avg_temp", cold_params)
        self.assertIn("total_precipitation", cold_params)
        
        normal_params = ModelConstants.WEATHER_PARAMETERS["normal"]
        self.assertIn("avg_temp", normal_params)
        
        warm_params = ModelConstants.WEATHER_PARAMETERS["warm"]
        self.assertIn("avg_temp", warm_params)
    
    def test_api_port_range(self):
        """Test API port is in valid range."""
        self.assertGreaterEqual(API_PORT, 1024)
        self.assertLessEqual(API_PORT, 65535)
    
    def test_weather_timeout_positive(self):
        """Test weather API timeout is positive."""
        self.assertGreater(WEATHER_API_TIMEOUT, 0)

if __name__ == "__main__":
    unittest.main()
