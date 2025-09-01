# Dutch Energy Consumption Predictor 🇳🇱⚡

An end-to-end machine learning pipeline for forecasting Dutch electricity & gas consumption patterns using comprehensive energy datasets combined with historical weather data from the Royal Netherlands Meteorological Institute (KNMI).

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

## Getting Started

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

## Machine Learning Pipeline

This project includes advanced machine learning capabilities for predicting energy consumption:

### `FixedEnergyConsumptionPredictor`
```python
from model_training import FixedEnergyConsumptionPredictor

# Initialize predictor
predictor = FixedEnergyConsumptionPredictor()

# Prepare training data with feature engineering
data = predictor.prepare_training_data("electricity", [2018, 2019, 2020])

# Train models with comprehensive features
results = predictor.train_model(data, 'household_consumption')

# Analyze feature importance
importance = predictor.analyze_feature_importance('household_consumption')
```

### Features
- **Household-level predictions** using connection type patterns
- **Comprehensive feature engineering**: temporal, weather, geographic, and connection features
- **Advanced preprocessing**: outlier removal, feature selection, robust scaling
- **Multiple algorithms**: Linear/Ridge/Lasso regression, Random Forest
- **Cross-validation**: Time-series aware validation for temporal data
- **Performance metrics**: R², RMSE, MAE with detailed analysis

### Quick ML Demo
```bash
python example_ml_usage.py
```

## Project Structure

```
Nl_Energy_Consumption_Predictor/
├── dataset.py              # Data infrastructure & integration
├── model_training.py       # ML pipeline & prediction models
├── example_ml_usage.py     # Quick ML demo
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
├── .gitignore            # Git ignore rules
├── SECURITY.md           # Security policy
├── LICENSE               # MIT license
├── weather_cache/        # Cached KNMI weather data (gitignored)
├── .github/              # GitHub workflows & dependabot config
└── README.md             # Project documentation
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

## Security & Privacy

### Data Security
- ✅ **No personal data**: Only aggregated energy consumption by postal code areas
- ✅ **Public datasets**: Energy and weather data from official open sources
- ✅ **Environment variables**: Sensitive configuration kept in `.env` files
- ✅ **Dependency scanning**: Automated security audits via GitHub Actions
- ✅ **Secret scanning**: TruffleHog integration for credential detection

### Setup Security
1. **Never commit credentials**: Use `.env` files for API keys
2. **Keep dependencies updated**: Regular Dependabot security updates
3. **Review dependencies**: Run `pip audit` for vulnerability checks
4. **Secure data paths**: Auto-discovery instead of hardcoded paths

See [SECURITY.md](SECURITY.md) for detailed security policies.

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** with security in mind
4. **Run tests**: Ensure all checks pass
5. **Submit a pull request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run security audit
pip audit

# Format code
black .

# Run linting
flake8 .
```

## Use Cases

This integrated dataset enables analysis of:
- **Seasonal energy patterns** and weather dependencies
- **Climate impact** on energy consumption trends
- **Regional variations** in energy usage patterns
- **Smart meter adoption** correlation with consumption changes
- **Predictive modeling** for energy demand forecasting
- **Climate change impact** on energy infrastructure planning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dutch Energy Dataset**: [Kaggle - Luca Basanisi](https://www.kaggle.com/datasets/lucabasa/dutch-energy)
- **Weather Data**: [KNMI - Royal Netherlands Meteorological Institute](https://www.knmi.nl/)
- **Energy Companies**: Liander, Enexis, Stedin, and other Dutch energy distributors
