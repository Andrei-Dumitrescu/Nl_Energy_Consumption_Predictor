import kagglehub
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import glob
import requests
import io
from datetime import datetime, date
import numpy as np

class DutchEnergyDataset:
    """
    A class to easily access and work with the Dutch Energy Consumption dataset from Kaggle.
    
    This dataset contains electricity and gas consumption data from various Dutch energy
    distribution companies across multiple years (2009-2020).
    
    IMPORTANT DATA INTERPRETATION:
    - annual_consume: Total kWh consumption for the postal code area
    - num_connections: Number of electrical connection points/circuits (NOT households)
    - Connection types (1x25, 3x35, etc.): Electrical capacity indicators (phases x amperage)
    - For household consumption: Use annual_consume / num_connections directly
    - perc_of_active_connections: Percentage of connections that are active/in-use
    
    CONNECTION TYPE MEANINGS:
    - 1x25, 1x35: Small apartments (low electrical capacity)
    - 3x25: Medium apartments/houses (medium capacity)  
    - 3x35, 3x50+: Larger houses (high electrical capacity)
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the dataset accessor.
        
        Args:
            dataset_path: Path to the dataset. If None, tries to auto-discover kagglehub cache location.
        """
        if dataset_path is None:
            # Try to auto-discover kagglehub cache location
            try:
                import kagglehub
                # Download or get cached dataset path
                self.dataset_path = Path(kagglehub.dataset_download("lucabasa/dutch-energy"))
            except ImportError:
                raise ImportError("kagglehub not installed. Install with: pip install kagglehub")
            except Exception as e:
                raise FileNotFoundError(f"Could not auto-discover dataset. Please provide dataset_path manually. Error: {e}")
        else:
            self.dataset_path = Path(dataset_path)
            
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
            
        self.electricity_path = self.dataset_path / "Electricity"
        self.gas_path = self.dataset_path / "Gas"
        
        # Cache for loaded data
        self._electricity_files = None
        self._gas_files = None
    
    def list_electricity_files(self) -> List[str]:
        """List all available electricity CSV files."""
        if self._electricity_files is None:
            self._electricity_files = [f.name for f in self.electricity_path.glob("*.csv")]
        return sorted(self._electricity_files)
    
    def list_gas_files(self) -> List[str]:
        """List all available gas CSV files."""
        if self._gas_files is None:
            self._gas_files = [f.name for f in self.gas_path.glob("*.csv")]
        return sorted(self._gas_files)
    
    def get_companies(self, energy_type: str = "both") -> List[str]:
        """
        Get list of energy companies in the dataset.
        
        Args:
            energy_type: "electricity", "gas", or "both"
            
        Returns:
            List of unique company names
        """
        companies = set()
        
        if energy_type in ["electricity", "both"]:
            for filename in self.list_electricity_files():
                company = filename.split('_')[0]
                companies.add(company)
                
        if energy_type in ["gas", "both"]:
            for filename in self.list_gas_files():
                company = filename.split('_')[0]
                companies.add(company)
                
        return sorted(list(companies))
    
    def get_years(self, energy_type: str = "both") -> List[int]:
        """
        Get list of available years in the dataset.
        
        Args:
            energy_type: "electricity", "gas", or "both"
            
        Returns:
            List of years
        """
        years = set()
        
        if energy_type in ["electricity", "both"]:
            for filename in self.list_electricity_files():
                # Extract year from filename
                if filename.count('_') >= 2:
                    year_part = filename.split('_')[-1].replace('.csv', '')
                    if year_part.isdigit() and len(year_part) == 4:
                        years.add(int(year_part))
                    elif 'electricity_' in filename:
                        # Handle date format like 01012020
                        date_part = filename.split('electricity_')[-1].replace('.csv', '')
                        if len(date_part) == 8 and date_part.isdigit():
                            year = int(date_part[-4:])
                            years.add(year)
                            
        if energy_type in ["gas", "both"]:
            for filename in self.list_gas_files():
                if filename.count('_') >= 2:
                    year_part = filename.split('_')[-1].replace('.csv', '')
                    if year_part.isdigit() and len(year_part) == 4:
                        years.add(int(year_part))
                    elif 'gas_' in filename:
                        # Handle date format like 01012020
                        date_part = filename.split('gas_')[-1].replace('.csv', '')
                        if len(date_part) == 8 and date_part.isdigit():
                            year = int(date_part[-4:])
                            years.add(year)
                            
        return sorted(list(years))
    
    def load_file(self, filename: str, energy_type: str) -> pd.DataFrame:
        """
        Load a specific CSV file.
        
        Args:
            filename: Name of the CSV file
            energy_type: "electricity" or "gas"
            
        Returns:
            pandas DataFrame with the data
        """
        if energy_type == "electricity":
            file_path = self.electricity_path / filename
        elif energy_type == "gas":
            file_path = self.gas_path / filename
        else:
            raise ValueError("energy_type must be 'electricity' or 'gas'")
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    def load_company_data(self, company: str, energy_type: str, year: Optional[int] = None) -> pd.DataFrame:
        """
        Load data for a specific company.
        
        Args:
            company: Company name (e.g., "liander", "enexis")
            energy_type: "electricity" or "gas"
            year: Specific year to load (optional)
            
        Returns:
            pandas DataFrame with the company's data
        """
        if energy_type == "electricity":
            files = self.list_electricity_files()
        else:
            files = self.list_gas_files()
            
        # Filter files for the company
        company_files = [f for f in files if f.lower().startswith(company.lower())]
        
        if year:
            # Filter for specific year
            year_files = []
            for f in company_files:
                if str(year) in f:
                    year_files.append(f)
            company_files = year_files
            
        if not company_files:
            raise ValueError(f"No files found for company '{company}' and energy type '{energy_type}'" + 
                           (f" for year {year}" if year else ""))
        
        # Load and concatenate all files
        dataframes = []
        for filename in company_files:
            df = self.load_file(filename, energy_type)
            # Add metadata columns
            df['source_file'] = filename
            df['company'] = company
            df['energy_type'] = energy_type
            dataframes.append(df)
            
        return pd.concat(dataframes, ignore_index=True)
    
    def load_year_data(self, year: int, energy_type: str) -> pd.DataFrame:
        """
        Load all data for a specific year across all companies.
        
        Args:
            year: Year to load
            energy_type: "electricity" or "gas"
            
        Returns:
            pandas DataFrame with all companies' data for that year
        """
        if energy_type == "electricity":
            files = self.list_electricity_files()
        else:
            files = self.list_gas_files()
            
        # Filter files for the year
        year_files = [f for f in files if str(year) in f]
        
        if not year_files:
            raise ValueError(f"No files found for year {year} and energy type '{energy_type}'")
            
        # Load and concatenate all files
        dataframes = []
        for filename in year_files:
            df = self.load_file(filename, energy_type)
            # Add metadata columns
            df['source_file'] = filename
            df['year'] = year
            df['energy_type'] = energy_type
            # Extract company name from filename
            company = filename.split('_')[0]
            df['company'] = company
            dataframes.append(df)
            
        return pd.concat(dataframes, ignore_index=True)
    
    def load_all_data(self, energy_type: str) -> pd.DataFrame:
        """
        Load all data for a specific energy type.
        
        Args:
            energy_type: "electricity" or "gas"
            
        Returns:
            pandas DataFrame with all data
        """
        if energy_type == "electricity":
            files = self.list_electricity_files()
        else:
            files = self.list_gas_files()
            
        dataframes = []
        for filename in files:
            df = self.load_file(filename, energy_type)
            # Add metadata columns
            df['source_file'] = filename
            df['energy_type'] = energy_type
            # Extract company and year from filename
            company = filename.split('_')[0]
            df['company'] = company
            
            # Extract year
            if filename.count('_') >= 2:
                year_part = filename.split('_')[-1].replace('.csv', '')
                if year_part.isdigit() and len(year_part) == 4:
                    df['year'] = int(year_part)
                elif f'{energy_type}_' in filename:
                    date_part = filename.split(f'{energy_type}_')[-1].replace('.csv', '')
                    if len(date_part) == 8 and date_part.isdigit():
                        df['year'] = int(date_part[-4:])
                        
            dataframes.append(df)
            
        return pd.concat(dataframes, ignore_index=True)
    
    def get_dataset_summary(self) -> Dict:
        """Get a summary of the dataset."""
        return {
            "dataset_path": str(self.dataset_path),
            "electricity_files": len(self.list_electricity_files()),
            "gas_files": len(self.list_gas_files()),
            "companies": self.get_companies(),
            "years_electricity": self.get_years("electricity"),
            "years_gas": self.get_years("gas"),
            "total_files": len(self.list_electricity_files()) + len(self.list_gas_files())
        }
    
    def calculate_consumption_per_connection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate consumption per electrical connection for a DataFrame.
        
        Args:
            df: DataFrame with annual_consume and num_connections columns
            
        Returns:
            DataFrame with added consumption_per_connection column
        """
        df = df.copy()
        df['consumption_per_connection'] = df['annual_consume'] / np.maximum(df['num_connections'], 0.1)
        return df


class KNMIWeatherData:
    """
    A class to download and process weather data from KNMI (Royal Netherlands Meteorological Institute).
    
    This class provides functionality to download historical weather data and aggregate it to annual values
    that can be integrated with the Dutch Energy dataset.
    """
    
    # KNMI station information - major weather stations across Netherlands
    STATIONS = {
        "240": {"name": "Schiphol", "location": "Amsterdam area"},
        "260": {"name": "De Bilt", "location": "Central Netherlands"},
        "330": {"name": "Hoek van Holland", "location": "South-West Netherlands"},
        "370": {"name": "Eindhoven", "location": "South Netherlands"},
        "380": {"name": "Maastricht", "location": "South-East Netherlands"},
        "280": {"name": "Eelde", "location": "North Netherlands"}
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the KNMI weather data downloader.
        
        Args:
            cache_dir: Directory to cache downloaded weather data. If None, creates 'weather_cache' directory.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("weather_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # KNMI data download URL
        self.base_url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
        
    def download_station_data(self, station: str, start_year: int, end_year: int, 
                             variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download daily weather data for a specific KNMI station.
        
        Args:
            station: KNMI station number (e.g., "260" for De Bilt)
            start_year: Start year for data download
            end_year: End year for data download
            variables: List of weather variables to download. If None, downloads common variables.
            
        Returns:
            pandas DataFrame with daily weather data
        """
        if variables is None:
            # Common weather variables for energy analysis
            variables = ["TG", "TN", "TX", "RH", "DR", "FG", "Q"]
            # TG = daily mean temperature, TN = min temp, TX = max temp
            # RH = daily precipitation, DR = sunshine duration, FG = wind speed, Q = global radiation
        
        # Check cache first
        cache_file = self.cache_dir / f"knmi_{station}_{start_year}_{end_year}.csv"
        if cache_file.exists():
            print(f"Loading cached data for station {station} ({start_year}-{end_year})")
            return pd.read_csv(cache_file, parse_dates=['YYYYMMDD'])
        
        print(f"Downloading KNMI data for station {station} ({start_year}-{end_year})")
        
        # Prepare parameters for KNMI API
        params = {
            "stns": station,
            "vars": ":".join(variables),
            "start": f"{start_year}0101",
            "end": f"{end_year}1231"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse the CSV data from KNMI
            # KNMI returns data without explicit headers, we need to construct them
            lines = response.text.split('\n')
            
            # Find where the actual data starts (after comment lines)
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#'):
                    data_start = i
                    break
            
            # Extract data lines only
            data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
            
            if not data_lines:
                print("No data found in KNMI response")
                return pd.DataFrame()
            
            # Construct header based on requested variables
            header_parts = ["STN", "YYYYMMDD"] + variables
            header_line = ",".join(header_parts)
            
            # Create DataFrame from the data
            csv_data = header_line + '\n' + '\n'.join(data_lines)
            df = pd.read_csv(io.StringIO(csv_data), skipinitialspace=True)
            
            # Clean and process the data
            if 'YYYYMMDD' in df.columns:
                df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
                df['year'] = df['YYYYMMDD'].dt.year
                df['month'] = df['YYYYMMDD'].dt.month
                df['day'] = df['YYYYMMDD'].dt.day
            
            # Convert KNMI units to standard units
            df = self._convert_knmi_units(df)
            
            # Cache the data
            df.to_csv(cache_file, index=False)
            print(f"Data cached to {cache_file}")
            
            return df
            
        except requests.RequestException as e:
            print(f"Error downloading data from KNMI: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing KNMI data: {e}")
            return pd.DataFrame()
    
    def _convert_knmi_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert KNMI units to more standard units.
        
        KNMI uses specific units and scaling:
        - Temperature is in 0.1°C
        - Precipitation is in 0.1mm
        - Wind speed is in 0.1 m/s
        - Sunshine duration is in 0.1 hours
        """
        df = df.copy()
        
        # Temperature conversions (0.1°C to °C)
        temp_cols = ['TG', 'TN', 'TX']
        for col in temp_cols:
            if col in df.columns:
                df[col] = df[col] / 10.0
                
        # Precipitation conversion (0.1mm to mm)
        if 'RH' in df.columns:
            df['RH'] = df['RH'] / 10.0
            
        # Wind speed conversion (0.1 m/s to m/s)
        if 'FG' in df.columns:
            df['FG'] = df['FG'] / 10.0
            
        # Sunshine duration conversion (0.1 hours to hours)
        if 'DR' in df.columns:
            df['DR'] = df['DR'] / 10.0
            
        return df
    
    def get_annual_aggregates(self, station: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Get annual weather aggregates for a station.
        
        Args:
            station: KNMI station number
            start_year: Start year
            end_year: End year
            
        Returns:
            DataFrame with annual weather aggregates
        """
        # Download daily data
        daily_data = self.download_station_data(station, start_year, end_year)
        
        if daily_data.empty:
            return pd.DataFrame()
        
        # Calculate annual aggregates
        annual_data = daily_data.groupby('year').agg({
            'TG': 'mean',  # Mean temperature
            'TN': 'mean',  # Mean minimum temperature  
            'TX': 'mean',  # Mean maximum temperature
            'RH': ['sum', 'mean'],  # Total and mean precipitation
            'DR': 'sum',   # Total sunshine hours
            'FG': 'mean',  # Mean wind speed
            'Q': 'sum' if 'Q' in daily_data.columns else lambda x: np.nan  # Total global radiation
        }).round(2)
        
        # Flatten column names
        annual_data.columns = [
            'avg_temp', 'avg_min_temp', 'avg_max_temp', 
            'total_precipitation', 'avg_precipitation',
            'total_sunshine_hours', 'avg_wind_speed', 'total_global_radiation'
        ]
        
        # Add station info
        annual_data['station'] = station
        annual_data['station_name'] = self.STATIONS.get(station, {}).get('name', 'Unknown')
        annual_data['station_location'] = self.STATIONS.get(station, {}).get('location', 'Unknown')
        
        # Reset index to make year a column
        annual_data = annual_data.reset_index()
        
        return annual_data
    
    def get_national_annual_weather(self, start_year: int = 2009, end_year: int = 2020) -> pd.DataFrame:
        """
        Get national annual weather data by averaging multiple stations.
        
        Args:
            start_year: Start year for data
            end_year: End year for data
            
        Returns:
            DataFrame with national annual weather averages
        """
        # Use major stations representing different regions
        major_stations = ["240", "260", "330", "370"]  # Schiphol, De Bilt, Hoek van Holland, Eindhoven
        
        all_station_data = []
        
        for station in major_stations:
            station_data = self.get_annual_aggregates(station, start_year, end_year)
            if not station_data.empty:
                all_station_data.append(station_data)
        
        if not all_station_data:
            return pd.DataFrame()
        
        # Combine all station data
        combined_data = pd.concat(all_station_data, ignore_index=True)
        
        # Calculate national averages by year
        national_data = combined_data.groupby('year').agg({
            'avg_temp': 'mean',
            'avg_min_temp': 'mean',
            'avg_max_temp': 'mean',
            'total_precipitation': 'mean',  # Average precipitation across stations
            'avg_precipitation': 'mean',
            'total_sunshine_hours': 'mean',
            'avg_wind_speed': 'mean',
            'total_global_radiation': 'mean'
        }).round(2)
        
        # Add metadata
        national_data['data_source'] = 'KNMI'
        national_data['stations_used'] = ','.join(major_stations)
        national_data['coverage'] = 'National average'
        
        # Reset index
        national_data = national_data.reset_index()
        
        return national_data


class IntegratedEnergyWeatherDataset:
    """
    A class that combines Dutch Energy data with KNMI weather data for comprehensive analysis.
    """
    
    def __init__(self, energy_dataset_path: Optional[str] = None):
        """
        Initialize the integrated dataset.
        
        Args:
            energy_dataset_path: Path to the energy dataset
        """
        self.energy_dataset = DutchEnergyDataset(energy_dataset_path)
        self.weather_data = KNMIWeatherData()
        
    def get_integrated_annual_data(self, energy_type: str = "electricity", 
                                  start_year: int = 2009, end_year: int = 2020) -> pd.DataFrame:
        """
        Get integrated annual energy and weather data.
        
        Args:
            energy_type: "electricity" or "gas"
            start_year: Start year for analysis
            end_year: End year for analysis
            
        Returns:
            DataFrame with integrated energy and weather data by year
        """
        print(f"Loading {energy_type} data for years {start_year}-{end_year}...")
        
        # Get annual energy data aggregated by year
        energy_annual = []
        
        for year in range(start_year, end_year + 1):
            try:
                year_data = self.energy_dataset.load_year_data(year, energy_type)
                if not year_data.empty:
                    # Calculate annual aggregates
                    annual_summary = {
                        'year': year,
                        'total_connections': year_data['num_connections'].sum(),
                        'avg_annual_consume': year_data['annual_consume'].mean(),
                        'total_annual_consume': year_data['annual_consume'].sum(),
                        'avg_delivery_perc': year_data['delivery_perc'].mean(),
                        'avg_active_connections_perc': year_data['perc_of_active_connections'].mean(),
                        'avg_smartmeter_perc': year_data['smartmeter_perc'].mean(),
                        'num_companies': year_data['company'].nunique(),
                        'num_records': len(year_data)
                    }
                    energy_annual.append(annual_summary)
            except Exception as e:
                print(f"Warning: Could not load {energy_type} data for {year}: {e}")
        
        energy_df = pd.DataFrame(energy_annual)
        
        # Get weather data
        print("Loading weather data...")
        weather_df = self.weather_data.get_national_annual_weather(start_year, end_year)
        
        # Merge energy and weather data
        if energy_df.empty or weather_df.empty:
            print("Warning: Unable to load energy or weather data")
            return pd.DataFrame()
        
        integrated_df = pd.merge(energy_df, weather_df, on='year', how='inner')
        integrated_df['energy_type'] = energy_type
        
        return integrated_df
    
    def get_weather_energy_correlation(self, energy_type: str = "electricity") -> Dict:
        """
        Calculate correlations between weather variables and energy consumption.
        
        Args:
            energy_type: "electricity" or "gas"
            
        Returns:
            Dictionary with correlation analysis results
        """
        integrated_data = self.get_integrated_annual_data(energy_type)
        
        if integrated_data.empty:
            return {}
        
        # Calculate correlations
        weather_cols = ['avg_temp', 'total_precipitation', 'total_sunshine_hours', 'avg_wind_speed']
        energy_cols = ['avg_annual_consume', 'total_annual_consume']
        
        correlations = {}
        for weather_var in weather_cols:
            for energy_var in energy_cols:
                if weather_var in integrated_data.columns and energy_var in integrated_data.columns:
                    corr = integrated_data[weather_var].corr(integrated_data[energy_var])
                    correlations[f"{weather_var}_vs_{energy_var}"] = round(corr, 3)
        
        return {
            "energy_type": energy_type,
            "correlations": correlations,
            "data_years": sorted(integrated_data['year'].tolist()),
            "sample_size": len(integrated_data)
        }

# Initialize the dataset accessor
def main():
    """Main function to demonstrate the dataset infrastructure."""
    
    # Initialize the dataset
    dataset = DutchEnergyDataset()
    
    # Print dataset summary
    summary = dataset.get_dataset_summary()
    print("=== Dutch Energy Dataset Summary ===")
    print(f"Dataset path: {summary['dataset_path']}")
    print(f"Total files: {summary['total_files']}")
    print(f"Electricity files: {summary['electricity_files']}")
    print(f"Gas files: {summary['gas_files']}")
    print(f"Companies: {', '.join(summary['companies'])}")
    print(f"Years (Electricity): {summary['years_electricity']}")
    print(f"Years (Gas): {summary['years_gas']}")
    
    print("\n=== KNMI Weather Data Integration ===")
    
    # Test weather data integration
    print("\n1. Testing KNMI weather data download:")
    try:
        weather_data = KNMIWeatherData()
        print("   Available KNMI stations:")
        for station_id, info in weather_data.STATIONS.items():
            print(f"   - {station_id}: {info['name']} ({info['location']})")
        
        # Try to get a small sample of weather data (just 2 years for demo)
        print("\n2. Getting sample weather data for 2019-2020:")
        sample_weather = weather_data.get_national_annual_weather(2019, 2020)
        if not sample_weather.empty:
            print(f"   Loaded weather data for {len(sample_weather)} years")
            print("   Weather variables:", [col for col in sample_weather.columns if col not in ['year', 'data_source', 'stations_used', 'coverage']])
            print("   Sample data:")
            print(sample_weather)
        else:
            print("   Could not load weather data")
    except Exception as e:
        print(f"   Error testing weather data: {e}")
    
    print("\n=== Integrated Energy + Weather Analysis ===")
    
    # Test integrated dataset
    print("\n3. Testing integrated energy and weather dataset:")
    try:
        integrated_dataset = IntegratedEnergyWeatherDataset()
        
        # Get integrated data for a smaller range first
        integrated_data = integrated_dataset.get_integrated_annual_data("electricity", 2018, 2020)
        if not integrated_data.empty:
            print(f"   Loaded integrated data for {len(integrated_data)} years")
            print("   Columns:", list(integrated_data.columns))
            print("   Sample integrated data:")
            print(integrated_data[['year', 'avg_annual_consume', 'avg_temp', 'total_precipitation']].head())
            
            # Calculate correlations
            correlations = integrated_dataset.get_weather_energy_correlation("electricity")
            if correlations:
                print(f"\n   Weather-Energy Correlations for {correlations['energy_type']}:")
                for corr_name, corr_value in correlations['correlations'].items():
                    print(f"   - {corr_name}: {corr_value}")
        else:
            print("   Could not create integrated dataset")
    except Exception as e:
        print(f"   Error testing integrated dataset: {e}")
    
    print("\n=== Basic Energy Data Examples ===")
    
    # Example 1: Load specific company data
    print("\n4. Loading Liander electricity data for 2020:")
    try:
        liander_2020 = dataset.load_company_data("liander", "electricity", 2020)
        print(f"   Loaded {len(liander_2020)} records")
        print(f"   Columns: {list(liander_2020.columns)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Load all data for a specific year
    print("\n5. Loading all electricity data for 2020:")
    try:
        all_2020 = dataset.load_year_data(2020, "electricity")
        print(f"   Loaded {len(all_2020)} records from {all_2020['company'].nunique()} companies")
        print(f"   Companies in 2020: {sorted(all_2020['company'].unique())}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    main()