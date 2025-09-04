#!/usr/bin/env python3
"""
FastAPI application for the Dutch Energy Consumption Predictor.
Provides REST API endpoints for making energy consumption predictions.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import uvicorn
from energy_predictor import EnergyPredictor
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dutch Energy Consumption Predictor API",
    description="Predict household energy consumption based on location, house type, and weather conditions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor_instance = None

def get_predictor():
    """Dependency to get the predictor instance."""
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = EnergyPredictor()
        if not predictor_instance.load_model():
            raise HTTPException(status_code=500, detail="Failed to load energy prediction model")
    return predictor_instance

# Pydantic models for request/response validation
class HouseTypeEnum(str):
    """Valid house/connection types."""
    TYPE_1X25 = "1x25"
    TYPE_1X35 = "1x35" 
    TYPE_3X25 = "3x25"
    TYPE_3X35 = "3x35"
    TYPE_3X50 = "3x50"


class WeatherScenarioEnum(str):
    """Valid weather scenarios."""
    COLD = "cold"
    NORMAL = "normal"
    WARM = "warm"

class PredictionRequest(BaseModel):
    """Request model for energy consumption prediction."""
    
    # Location (either postal_code or city, postal_code takes precedence)
    postal_code: Optional[str] = Field(None, description="4-digit Dutch postal code (e.g., '1012')")
    city: Optional[str] = Field(None, description="City name (used if postal_code not provided)")
    
    # House characteristics
    house_type: str = Field("3x25", description="House electrical connection type")
    
    # Neighborhood characteristics
    num_connections: Optional[int] = Field(None, description="Number of connections in area (default: 30 if not provided)")
    active_connections_pct: float = Field(88.0, description="Percentage of active connections (50-95)")
    
    # Technology
    smart_meter: bool = Field(True, description="Whether smart meter is installed")
    
    # Energy company
    energy_company: str = Field("liander", description="Energy distribution company")
    
    # Weather scenario
    weather_scenario: str = Field("normal", description="Weather scenario for prediction")
    
    @validator('house_type')
    def validate_house_type(cls, v):
        valid_types = ["1x25", "1x35", "3x25", "3x35", "3x50"]
        if v not in valid_types:
            raise ValueError(f"house_type must be one of {valid_types}")
        return v
    
    
    @validator('active_connections_pct')
    def validate_active_connections(cls, v):
        if not (50 <= v <= 95):
            raise ValueError("active_connections_pct must be between 50 and 95")
        return v
    
    @validator('energy_company')
    def validate_company(cls, v):
        valid_companies = ['liander', 'enexis', 'stedin', 'westland-infra', 'coteq']
        if v.lower() not in valid_companies:
            raise ValueError(f"energy_company must be one of {valid_companies}")
        return v.lower()
    
    @validator('weather_scenario')
    def validate_weather(cls, v):
        valid_weather = ["cold", "normal", "warm"]
        if v.lower() not in valid_weather:
            raise ValueError(f"weather_scenario must be one of {valid_weather}")
        return v.lower()
    
    @validator('postal_code')
    def validate_postal_code(cls, v):
        if v is not None:
            # Remove any spaces and validate format
            v = v.strip().replace(' ', '')
            if not (v.isdigit() and len(v) >= 2):
                raise ValueError("postal_code must be at least 2 digits")
            # Pad to 4 digits if needed
            v = v[:4].ljust(4, '0')
        return v

class PredictionResponse(BaseModel):
    """Response model for energy consumption prediction."""
    model_config = {"protected_namespaces": ()}
    
    prediction_kwh: float = Field(description="Predicted annual consumption in kWh")
    monthly_kwh: float = Field(description="Average monthly consumption in kWh")
    daily_kwh: float = Field(description="Average daily consumption in kWh")
    estimated_monthly_cost: float = Field(description="Estimated monthly cost in EUR")
    estimated_yearly_cost: float = Field(description="Estimated yearly cost in EUR")
    model_used: str = Field(description="Name of the model used for prediction")
    input_summary: Dict[str, Any] = Field(description="Summary of input parameters used")
    comparison_to_average: Dict[str, Any] = Field(description="Comparison to typical Dutch household")
    timestamp: datetime = Field(description="Timestamp of prediction")

class ModelInfo(BaseModel):
    """Model information response."""
    model_config = {"protected_namespaces": ()}
    
    model_loaded: bool
    model_performance: Dict[str, Any]
    available_house_types: List[str]
    available_companies: List[str]
    available_weather_scenarios: List[str]

@app.get("/", summary="Root endpoint", tags=["General"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Dutch Energy Consumption Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", summary="Health check", tags=["General"])
async def health_check(predictor: EnergyPredictor = Depends(get_predictor)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model_package is not None,
        "timestamp": datetime.now()
    }

@app.get("/model/info", response_model=ModelInfo, summary="Get model information", tags=["Model"])
async def get_model_info(predictor: EnergyPredictor = Depends(get_predictor)):
    """Get information about the loaded model."""
    return ModelInfo(
        model_loaded=predictor.model_package is not None,
        model_performance=predictor.model_package.get('model_performance', {}) if predictor.model_package else {},
        available_house_types=["1x25", "1x35", "3x25", "3x35", "3x50"],
        available_companies=['liander', 'enexis', 'stedin', 'westland-infra', 'coteq'],
        available_weather_scenarios=["cold", "normal", "warm"]
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict energy consumption", tags=["Prediction"])
async def predict_consumption(request: PredictionRequest, predictor: EnergyPredictor = Depends(get_predictor)):
    """
    Predict household energy consumption based on input parameters.
    
    The API accepts either a postal code or city name for location. If both are provided,
    postal code takes precedence. If neither is provided, a neutral default is used.
    """
    try:
        # Convert request to predictor input format
        user_inputs = _convert_request_to_predictor_input(request)
        
        # Make prediction
        results = predictor.predict(user_inputs)
        
        # Calculate comparison to average
        typical_consumption = 2223  # Training data average
        difference = results['prediction_kwh'] - typical_consumption
        percentage_diff = (difference / typical_consumption) * 100
        
        comparison = {
            "typical_dutch_household_kwh": typical_consumption,
            "difference_kwh": difference,
            "percentage_difference": percentage_diff,
            "comparison_text": _get_comparison_text(percentage_diff)
        }
        
        return PredictionResponse(
            prediction_kwh=results['prediction_kwh'],
            monthly_kwh=results['monthly_kwh'],
            daily_kwh=results['daily_kwh'],
            estimated_monthly_cost=results['estimated_monthly_cost'],
            estimated_yearly_cost=results['estimated_monthly_cost'] * 12,
            model_used=results['model_used'],
            input_summary=user_inputs,
            comparison_to_average=comparison,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

def _convert_request_to_predictor_input(request: PredictionRequest) -> Dict[str, Any]:
    """Convert API request to predictor input format."""
    
    # Handle location
    if request.postal_code:
        zipcode_from = request.postal_code
        city = request.city if request.city else "Utrecht"
    else:
        zipcode_from = "3500"  # Neutral default
        city = request.city if request.city else "Utrecht"
    
    # Auto-calculate circuits per household based on house type (same as CLI)
    circuits_map = {
        '1x25': 10, '1x35': 10, '3x25': 14, '3x35': 18, '3x50': 22
    }
    circuits = circuits_map.get(request.house_type, 14)
    
    # Handle number of connections (default to 30 if not provided)
    num_connections = request.num_connections if request.num_connections else 30
    
    # Weather parameters
    weather_params = _get_weather_params(request.weather_scenario)
    
    # Build user inputs
    user_inputs = {
        'zipcode_from': zipcode_from,
        'city': city,
        'company': request.energy_company,
        'type_of_connection': request.house_type,
        'circuits_per_household': circuits,
        'num_connections': num_connections,
        'perc_of_active_connections': request.active_connections_pct,
        'delivery_perc': 97.5,  # Standard Dutch grid reliability
        'smartmeter_perc': 75 if request.smart_meter else 25,
        'annual_consume': 3500 * circuits,  # Rough estimate, will be overridden by prediction
        **weather_params
    }
    
    return user_inputs

def _get_weather_params(scenario: str) -> Dict[str, float]:
    """Get weather parameters for the given scenario."""
    if scenario == 'cold':
        return {
            'avg_temp': 8.0,
            'total_precipitation': 900,
            'total_sunshine_hours': 1400,
            'avg_wind_speed': 4.5,
            'total_global_radiation': 3200,
        }
    elif scenario == 'warm':
        return {
            'avg_temp': 13.0,
            'total_precipitation': 800,
            'total_sunshine_hours': 1700,
            'avg_wind_speed': 3.8,
            'total_global_radiation': 3600,
        }
    else:  # normal
        return {
            'avg_temp': 10.5,
            'total_precipitation': 850,
            'total_sunshine_hours': 1580,
            'avg_wind_speed': 4.2,
            'total_global_radiation': 3400,
        }

def _get_comparison_text(percentage_diff: float) -> str:
    """Get descriptive text for consumption comparison."""
    if percentage_diff < -20:
        return "Much lower than typical Dutch household"
    elif percentage_diff < -10:
        return "Lower than typical Dutch household"
    elif percentage_diff < 10:
        return "Similar to typical Dutch household"
    elif percentage_diff < 20:
        return "Higher than typical Dutch household"
    else:
        return "Much higher than typical Dutch household"

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
