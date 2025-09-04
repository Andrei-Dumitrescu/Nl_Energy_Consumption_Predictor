# Dutch Energy Consumption Predictor API

This FastAPI application provides REST endpoints for predicting household energy consumption in the Netherlands.

## Quick Start

### Prerequisites

Install the required dependencies:
```bash
pip install fastapi uvicorn pydantic
```

### Running the API

1. Start the API server:
```bash
python api.py
```

2. The API will be available at:
   - **API Base URL**: http://127.0.0.1:8000
   - **Interactive Documentation**: http://127.0.0.1:8000/docs
   - **Alternative Documentation**: http://127.0.0.1:8000/redoc

### Testing the API

Run the test script to verify everything works:
```bash
python test_api.py
```

## API Endpoints

### General Endpoints

- **GET /**: Root endpoint with basic API information
- **GET /health**: Health check endpoint
- **GET /model/info**: Get information about the loaded prediction model

### Prediction Endpoint

- **POST /predict**: Make energy consumption predictions

## Usage Examples

### Basic Prediction (with postal code)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "postal_code": "1012",
       "city": "Amsterdam",
       "house_type": "3x25",
       "smart_meter": true
     }'
```

### Prediction with City Only

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "city": "Rotterdam",
       "house_type": "3x35",
       "num_connections": 500,
       "weather_scenario": "cold"
     }'
```

### Minimal Prediction (all defaults)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{}'
```

## Request Parameters

### Location Parameters
- `postal_code` (optional): 4-digit Dutch postal code (e.g., "1012")
- `city` (optional): City name (used if postal_code not provided)

### House Characteristics
- `house_type`: Connection type - "1x25", "1x35", "3x25", "3x35", "3x50" (default: "3x25")
  - Automatically determines circuits: 1x25/1x35→10, 3x25→14, 3x35→18, 3x50→22

### Neighborhood Characteristics
- `num_connections` (optional): Number of connections in area (default: 30 if not provided)
- `active_connections_pct`: Percentage of active connections, 50-95 (default: 88.0)

### Technology & Company
- `smart_meter`: Whether smart meter is installed (default: true)
- `energy_company`: Energy distribution company - "liander", "enexis", "stedin", "westland-infra", "coteq" (default: "liander")

### Weather
- `weather_scenario`: Weather scenario - "cold", "normal", "warm" (default: "normal")

## Response Format

```json
{
  "prediction_kwh": 3245.7,
  "monthly_kwh": 270.5,
  "daily_kwh": 8.9,
  "estimated_monthly_cost": 135.2,
  "estimated_yearly_cost": 1622.8,
  "model_used": "household_consumption",
  "input_summary": {
    "zipcode_from": "1012",
    "city": "Amsterdam",
    "house_type": "3x25",
    ...
  },
  "comparison_to_average": {
    "typical_dutch_household_kwh": 2223,
    "difference_kwh": 1022.7,
    "percentage_difference": 46.0,
    "comparison_text": "Higher than typical Dutch household"
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## Python Client Example

```python
import requests

# Basic prediction
response = requests.post("http://127.0.0.1:8000/predict", json={
    "postal_code": "3500",
    "house_type": "3x25",
    "smart_meter": True,
    "weather_scenario": "normal"
})

if response.status_code == 200:
    result = response.json()
    print(f"Predicted consumption: {result['prediction_kwh']:.0f} kWh/year")
    print(f"Monthly cost: €{result['estimated_monthly_cost']:.0f}")
else:
    print(f"Error: {response.text}")
```

## Location Handling

The API uses the same logic as the interactive predictor:

1. **If postal_code is provided**: Uses the postal code and optional city name
2. **If only city is provided**: Uses neutral postal code (3500) with the city name
3. **If neither is provided**: Uses neutral defaults (postal code 3500, city "Utrecht")

This ensures predictions remain unbiased when location information is incomplete.

## Error Handling

The API includes comprehensive validation:
- Invalid house types, companies, weather scenarios
- Out-of-range percentages
- Malformed postal codes
- Model loading failures

All errors return appropriate HTTP status codes with descriptive messages.
