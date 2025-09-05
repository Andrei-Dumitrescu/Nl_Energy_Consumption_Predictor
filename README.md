# Dutch Energy Consumption Predictor

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-deployment--ready-green)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![API](https://img.shields.io/badge/API-FastAPI-009688.svg)

An end-to-end machine learning pipeline for forecasting Dutch electricity & gas consumption patterns using comprehensive energy datasets combined with historical weather data from the Royal Netherlands Meteorological Institute (KNMI).

## Key Results

- **98.8% Accuracy** (R² = 0.988) on 3.5M+ household predictions
- **157 kWh/year RMSE** (7.1% relative error on average consumption)
- **Future-Proof**: Predicts energy consumption for any year (2025, 2030, etc.)
- **Production-Ready**: FastAPI endpoint + interactive CLI tool
- **Comprehensive**: 12 years of data (2009-2020) from 9 major Dutch energy companies
- **Real-Time**: Weather data integration from 6 KNMI stations across Netherlands

## Dataset Overview

This project integrates two comprehensive datasets to analyze energy consumption patterns in the Netherlands:

### Energy Data (Kaggle - Dutch Energy Dataset)
- **150 CSV files** covering 2009-2020
- **75 electricity** + **75 gas** consumption files
- **9 major energy companies**: Liander, Enexis, Stedin, Coteq, Endinet, Enduris, Rendo, Westland-infra
- **Detailed consumption data** by postal code areas across the Netherlands
- Variables: connections, delivery percentages, annual consumption, smart meter adoption

### Weather Data (KNMI - Royal Netherlands Meteorological Institute)
- **6 major weather stations** across the Netherlands
- **Daily weather data** automatically aggregated to annual summaries
- **7 weather variables**: temperature (avg/min/max), precipitation, sunshine hours, wind speed, global radiation
- **Automatic caching** system for efficient data retrieval
- **2009-2020 coverage** matching energy data timeframe

## Architecture

### Core Classes

#### `DutchEnergyDataset`
Comprehensive interface for accessing the Dutch energy consumption dataset:
```python
from dataset import DutchEnergyDataset

dataset = DutchEnergyDataset()

# Load data by company and year
liander_2020 = dataset.load_company_data("liander", "electricity", 2020)

# Load all data for a specific year
all_2020 = dataset.load_year_data(2020, "electricity")

# Get dataset overview
summary = dataset.get_dataset_summary()
```

#### `KNMIWeatherData`
Downloads and processes weather data from KNMI:
```python
from dataset import KNMIWeatherData

weather = KNMIWeatherData()

# Get national weather averages
national_weather = weather.get_national_annual_weather(2009, 2020)

# Get specific station data
station_data = weather.get_annual_aggregates("260", 2009, 2020)  # De Bilt
```

#### `IntegratedEnergyWeatherDataset`
Combines energy and weather data for comprehensive analysis:
```python
from dataset import IntegratedEnergyWeatherDataset

integrated = IntegratedEnergyWeatherDataset()

# Get combined annual data
combined_data = integrated.get_integrated_annual_data("electricity", 2009, 2020)

# Calculate weather-energy correlations
correlations = integrated.get_weather_energy_correlation("electricity")
```

## Key Features

### Data Integration
- **Automatic data alignment** by year (2009-2020)
- **Smart caching** to avoid redundant downloads
- **Error handling** for missing data periods
- **Metadata preservation** (source files, companies, stations)

### Analysis Capabilities
- **Energy consumption aggregation** by year, company, region
- **Weather pattern analysis** across multiple stations
- **Correlation analysis** between weather and consumption
- **National averages** from regional weather stations

### Weather-Energy Insights
Initial correlation analysis reveals:
- **Temperature impact**: -0.644 correlation (cooler → higher consumption)
- **Sunshine effect**: +0.245 correlation (more sun → higher consumption)
- **Precipitation**: Minimal direct impact on consumption
- **Seasonal patterns** captured through comprehensive weather variables

## Quick Start

### Prerequisites

**Python 3.8+** is required. Install dependencies using:

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy scikit-learn requests kagglehub
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Nl_Energy_Consumption_Predictor.git
   cd Nl_Energy_Consumption_Predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your Kaggle API credentials if needed
   ```

### Quick Demo (30 seconds)
```bash
# Train the model (first time only, ~2-3 minutes)
python model_training.py

# Make predictions interactively
python energy_predictor.py

# Or use command line
python energy_predictor.py --house-type 3X25 --location 35 --weather normal

# Start the API server
python api.py
# Then visit http://127.0.0.1:8000/docs for interactive API docs

# For comprehensive demo with visualizations
jupyter notebook demo.ipynb

# For beautiful web interface
python -m streamlit run streamlit_app.py
# Then visit http://localhost:8501

# For API testing
python test_api.py
```

4. **Verify installation**:
   ```bash
   python dataset.py
   ```

### Basic Usage
```python
# Initialize the integrated dataset
from dataset import IntegratedEnergyWeatherDataset

# Create integrated dataset
dataset = IntegratedEnergyWeatherDataset()

# Get 12 years of combined energy + weather data
data = dataset.get_integrated_annual_data("electricity", 2009, 2020)

# Analyze correlations
correlations = dataset.get_weather_energy_correlation("electricity")
print(correlations)
```

### Running the Demo
```bash
python dataset.py
```

This will:
1. Display energy dataset summary
2. Download sample weather data from KNMI
3. Create integrated dataset
4. Show weather-energy correlations
5. Demonstrate basic data loading

## More Usage Examples

### Batch Predictions
```python
# Process multiple households at once
from energy_predictor import EnergyPredictor

predictor = EnergyPredictor()
predictor.load_model()

households = [
    {'zipcode_from': '1012', 'house_type': '3x25', 'city': 'Amsterdam'},
    {'zipcode_from': '3500', 'house_type': '3x50', 'city': 'Utrecht'},
    {'zipcode_from': '3011', 'house_type': '1x25', 'city': 'Rotterdam'},
]

for house in households:
    result = predictor.predict(house)
    print(f"{house['city']}: {result['prediction_kwh']:.0f} kWh/year")
```

### API Integration Examples
```python
import requests

# Basic API call
response = requests.post("http://127.0.0.1:8000/predict", json={
    "postal_code": "1012",
    "house_type": "3x25",
    "weather_scenario": "normal"
})

if response.status_code == 200:
    result = response.json()
    print(f"Predicted: {result['prediction_kwh']:.0f} kWh/year")
    print(f"Monthly cost: €{result['estimated_monthly_cost']:.0f}")
```

### Different Weather Scenarios
```python
# Compare energy usage across weather scenarios
scenarios = ['cold', 'normal', 'warm']
house_data = {'zipcode_from': '1012', 'house_type': '3x25'}

for scenario in scenarios:
    input_data = {**house_data, 'weather_scenario': scenario}
    result = predictor.predict(input_data)
    print(f"{scenario.title()} year: {result['prediction_kwh']:.0f} kWh")
```

## Multiple User Interfaces

This project provides several ways to interact with the energy prediction model:

### 🖥️ **Command Line Interface**
```bash
# Interactive mode with guided questionnaire
python energy_predictor.py

# Quick command-line predictions
python energy_predictor.py --house-type 3X25 --location 35 --weather normal
```

### **Web Interface (Streamlit)**
```bash
python -m streamlit run streamlit_app.py
# Visit: http://localhost:8501
```
- **Beautiful, interactive web app**
- **Real-time predictions with visualizations**  
- **User-friendly forms and charts**
- **Prediction history and insights**

### **REST API (FastAPI)**
```bash
python api.py
# Visit: http://127.0.0.1:8000/docs
```
- **Production-ready API endpoints**
- **Interactive API documentation**
- **JSON request/response format**
- **Integration ready for other applications**

### 📓 **Jupyter Notebook Demo**
```bash
jupyter notebook demo.ipynb
```
- **Comprehensive technical demonstration**
- **Data exploration and visualization**
- **Model performance analysis**
- **Feature importance insights**

### 🧪 **API Testing**
```bash
python test_api.py              # Full test suite
python test_api.py --quick      # Quick basic tests
```
- **Comprehensive API testing**
- **Performance benchmarks**
- **Input validation verification**
- **Error handling tests**

## Machine Learning Pipeline

This project includes advanced machine learning capabilities for predicting energy consumption:

### `FixedEnergyConsumptionPredictor`
```python
from model_training import FixedEnergyConsumptionPredictor

# Initialize predictor
predictor = FixedEnergyConsumptionPredictor()

# Prepare training data with feature engineering (using all available years)
data = predictor.prepare_training_data("electricity", list(range(2009, 2021)))

# Train models with comprehensive features
results = predictor.train_model(data, 'household_consumption')

# Analyze feature importance
importance = predictor.analyze_feature_importance('household_consumption')
```

### Features
- **Household-level predictions** using connection type patterns
- **Future-proof modeling**: No year-based features, allowing predictions for any future year
- **Comprehensive feature engineering**: weather, geographic, and infrastructure connection features
- **Advanced preprocessing**: outlier removal, LASSO feature selection, robust scaling
- **Multiple algorithms**: Linear/Ridge/Lasso regression, Random Forest
- **Cross-validation**: Time-series aware validation for temporal data
- **Performance metrics**: R² = 0.988, RMSE = 157 kWh/year (7.1% relative error)

### Model Architecture
The prediction pipeline uses a **RandomForest** model trained on **3.5M+ data points** from 2009-2020, achieving state-of-the-art performance:

**Top Predictive Features:**
1. **Connection Efficiency** (58.4%) - Utilization patterns of electrical infrastructure
2. **Connection Type** (13.1%) - Household electrical capacity (1X25 to 3X50)
3. **Connections per Household** (12.1%) - Neighborhood density proxy
4. **Active Connection Count** (4.3%) - Infrastructure activity levels
5. **Weather Conditions** - Temperature, precipitation, sunshine patterns

**Training Details:**
- **Dataset**: 3,566,454 households across Netherlands (2009-2020)
- **Features**: 36 engineered features → 18 selected via LASSO
- **Performance**: 98.8% variance explained, 157 kWh/year RMSE
- **Validation**: Time-series cross-validation for temporal robustness

### Quick ML Demo
```bash
python example_ml_usage.py
```

## Energy Prediction Tool

### **Unified Interface** - Interactive & Command Line

**Train the Model:**
```bash
# Train on full dataset (2009-2020) and save model
python model_training.py
```

**Make Predictions:**
```bash
# Interactive mode (default) - comprehensive questionnaire
python energy_predictor.py

# Command-line mode - quick predictions
python energy_predictor.py --house-type 3X25 --location 35 --weather normal

# See all options
python energy_predictor.py --help
```

**Future-Ready Predictions:** The model can predict energy consumption for **any future year** (2025, 2030, etc.) because it doesn't rely on year-specific patterns but focuses on fundamental household and weather characteristics.

### **Interactive Mode**
Guided questionnaire with detailed inputs:
- House type & electrical connection (1X25 to 3X50)
- Location (postal code, city, energy company)
- Neighborhood characteristics (urban/suburban/rural)
- Connection details (activity %, smart meter status)
- Weather scenario (cold/normal/warm year)

### **Command-Line Mode**
Quick predictions with essential parameters:
```bash
# Basic usage with defaults (medium house, Utrecht, normal weather)
python energy_predictor.py --house-type 3X25 --location 35 --weather normal

# Large house in Amsterdam, cold year
python energy_predictor.py --house-type 3X50 --location 10 --weather cold

# Small apartment without smart meter
python energy_predictor.py --house-type 1X25 --location 20 --weather warm --no-smart-meter
```

**Parameters:**
- `--house-type`: `1X25`, `1X35`, `3X25`, `3X35`, `3X50`
- `--location`: First 2 digits of postal code (10=Amsterdam, 35=Utrecht, etc.)
- `--weather`: `cold`, `normal`, `warm`
- `--no-smart-meter`: Disable smart meter assumption

## Project Structure

```
Nl_Energy_Consumption_Predictor/
├── dataset.py                    # Data infrastructure & integration
├── model_training.py             # ML pipeline & prediction models
├── example_ml_usage.py           # Quick ML demo
├── energy_predictor.py           # Unified prediction tool (interactive + CLI)
├── requirements.txt              # Python dependencies
├── .env.example                 # Environment configuration template
├── .gitignore                   # Git ignore rules
├── SECURITY.md                  # Security policy
├── LICENSE                      # MIT license
├── energy_consumption_model.pkl # Saved model (auto-generated)
├── weather_cache/               # Cached KNMI weather data (gitignored)
├── .github/                     # GitHub workflows & dependabot config
└── README.md                    # Project documentation
```

## Data Sources

### Energy Data
- **Source**: [Dutch Energy Dataset on Kaggle](https://www.kaggle.com/datasets/lucabasa/dutch-energy)
- **Coverage**: 2009-2020, all major Dutch energy distributors
- **Format**: CSV files organized by company and energy type
- **License**: Kaggle dataset license

### Weather Data
- **Source**: [KNMI (Royal Netherlands Meteorological Institute)](https://www.knmi.nl/)
- **API**: `https://www.daggegevens.knmi.nl/klimatologie/daggegevens`
- **Coverage**: 6 major weather stations across Netherlands
- **License**: KNMI open data policy

## Available Weather Variables

| Variable | Description | Unit |
|----------|-------------|------|
| `avg_temp` | Annual average temperature | °C |
| `avg_min_temp` | Annual average minimum temperature | °C |
| `avg_max_temp` | Annual average maximum temperature | °C |
| `total_precipitation` | Annual total precipitation | mm |
| `avg_precipitation` | Annual average daily precipitation | mm |
| `total_sunshine_hours` | Annual total sunshine duration | hours |
| `avg_wind_speed` | Annual average wind speed | m/s |
| `total_global_radiation` | Annual total global radiation | J/cm² |

## Acknowledgments

- **Dutch Energy Dataset**: [Kaggle - Luca Basanisi](https://www.kaggle.com/datasets/lucabasa/dutch-energy)
- **Weather Data**: [KNMI - Royal Netherlands Meteorological Institute](https://www.knmi.nl/)
- **Energy Companies**: Liander, Enexis, Stedin, and other Dutch energy distributors
