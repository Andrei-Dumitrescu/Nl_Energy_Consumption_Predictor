"""
Custom exceptions for Dutch Energy Consumption Predictor.

This module defines domain-specific exceptions for better error handling.
"""

class EnergyPredictorError(Exception):
    """Base exception for all energy predictor related errors."""
    pass

class ModelError(EnergyPredictorError):
    """Raised when there are issues with the ML model."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when the trained model file cannot be found."""
    pass

class ModelLoadError(ModelError):
    """Raised when the model fails to load."""
    pass

class PredictionError(ModelError):
    """Raised when prediction fails."""
    pass

class DataError(EnergyPredictorError):
    """Raised when there are issues with data processing."""
    pass

class DataNotFoundError(DataError):
    """Raised when required data cannot be found."""
    pass

class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass

class WeatherDataError(DataError):
    """Raised when weather data cannot be retrieved or processed."""
    pass

class ConfigurationError(EnergyPredictorError):
    """Raised when there are configuration issues."""
    pass

class APIError(EnergyPredictorError):
    """Raised when there are API-related issues."""
    pass

class ValidationError(EnergyPredictorError):
    """Raised when input validation fails."""
    pass
