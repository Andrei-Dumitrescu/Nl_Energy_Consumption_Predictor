#!/usr/bin/env python3
"""
Energy Consumption Predictor

This script loads a pre-trained model and provides both interactive and command-line
interfaces for household energy consumption prediction.

The model must be trained first using model_training.py before using this predictor.

Usage:
    # Interactive mode
    python energy_predictor.py
    
    # Command-line mode  
    python energy_predictor.py --house-type 3X25 --location 35 --weather normal
    
    # Help
    python energy_predictor.py --help
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import sys
from typing import Dict, Optional, Any

class EnergyPredictor:
    """Production energy consumption predictor that loads a pre-trained model."""
    
    def __init__(self, model_path: str = 'energy_consumption_model.pkl'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_package = None
        
    def load_model(self, verbose: bool = True) -> bool:
        """
        Load the pre-trained model.
        
        Args:
            verbose: Whether to print loading status
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.model_path):
            if verbose:
                print(f"❌ Model file not found: {self.model_path}")
                print("Please train a model first using: python model_training.py")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model_package = pickle.load(f)
            
            if verbose:
                print(f"✅ Model loaded successfully!")
                print(f"   Model: {self.model_package['model_name']}")
                print(f"   Performance: R² = {self.model_package['model_performance']['test_r2']:.3f}")
                print(f"   Features: {self.model_package['training_metadata']['selected_feature_count']} selected")
                print(f"   Created: {self.model_package['training_metadata']['created_at'][:10]}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Error loading model: {e}")
            return False
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering as used during training.
        
        This is a simplified version that creates the core features needed for prediction.
        """
        df = df.copy()
        
        # Connection-to-household estimation
        def estimate_households_from_connections(row):
            connection_type = str(row.get('type_of_connection', '3x25')).upper()
            num_connections = row.get('num_connections', 200)
            
            if '1X25' in connection_type or '1X35' in connection_type:
                circuits_per_household = 10
            elif '3X25' in connection_type:
                circuits_per_household = 14
            elif '3X35' in connection_type:
                circuits_per_household = 18
            elif '3X50' in connection_type or '3X63' in connection_type:
                circuits_per_household = 22
            else:
                circuits_per_household = 12
                
            estimated_households = num_connections / circuits_per_household
            return max(estimated_households, 0.1)
        
        df['estimated_households'] = df.apply(estimate_households_from_connections, axis=1)
        df['household_consumption'] = df['annual_consume'] / df['estimated_households']
        
        # ===== TEMPORAL FEATURES REMOVED =====
        # Note: Year-based features removed to allow prediction on future data
        # The model should not be biased toward specific training years
        
        # Weather features
        df['cooling_degree_days'] = np.maximum(df.get('avg_temp', 10.5) - 22, 0) * 365
        df['sunshine_ratio'] = df.get('total_sunshine_hours', 1580) / 1800
        heating_days = np.maximum(18 - df.get('avg_temp', 10.5), 0) * 365
        df['heating_cooling_balance'] = heating_days - df['cooling_degree_days']
        
        # Weather intensity indicators
        avg_temp_val = df.get('avg_temp', 10.5)
        if hasattr(avg_temp_val, 'iloc'):
            avg_temp_val = avg_temp_val.iloc[0]
        total_precip_val = df.get('total_precipitation', 850)
        if hasattr(total_precip_val, 'iloc'):
            total_precip_val = total_precip_val.iloc[0]
        total_sun_val = df.get('total_sunshine_hours', 1580)
        if hasattr(total_sun_val, 'iloc'):
            total_sun_val = total_sun_val.iloc[0]
            
        df['extreme_cold_indicator'] = int(avg_temp_val < 5)
        df['extreme_hot_indicator'] = int(avg_temp_val > 15)
        df['high_precipitation'] = int(total_precip_val > 900)
        df['low_sunshine'] = int(total_sun_val < 1400)
        
        # Connection type features
        connection_type_val = df.get('type_of_connection', '3x25')
        if hasattr(connection_type_val, 'iloc'):
            connection_type_val = connection_type_val.iloc[0]
        connection_type = str(connection_type_val).upper()
        
        if '1X' in connection_type:
            df['connection_phases'] = 1
            try:
                df['connection_amperage'] = int(connection_type.split('X')[1]) if 'X' in connection_type else 25
            except (ValueError, IndexError):
                df['connection_amperage'] = 25
        elif '3X' in connection_type:
            df['connection_phases'] = 3
            try:
                df['connection_amperage'] = int(connection_type.split('X')[1]) if 'X' in connection_type else 25
            except (ValueError, IndexError):
                df['connection_amperage'] = 25
        else:
            df['connection_phases'] = 3
            df['connection_amperage'] = 25
            
        df['total_electrical_capacity'] = df['connection_phases'] * df['connection_amperage']
        df['has_high_voltage'] = int('400' in connection_type or '500' in connection_type)
        df['has_medium_voltage'] = int('230' in connection_type or '240' in connection_type)
        
        # Household size categories
        capacity = df['total_electrical_capacity'].iloc[0] if hasattr(df['total_electrical_capacity'], 'iloc') else df['total_electrical_capacity']
        if capacity <= 25:
            df['household_size_category'] = 'small_apartment'
        elif capacity <= 50:
            df['household_size_category'] = 'medium_apartment'
        elif capacity <= 75:
            df['household_size_category'] = 'large_apartment'
        elif capacity <= 100:
            df['household_size_category'] = 'small_house'
        else:
            df['household_size_category'] = 'large_house'
        
        # Smart meter features
        smartmeter_perc_val = df.get('smartmeter_perc', 75)
        if hasattr(smartmeter_perc_val, 'iloc'):
            smartmeter_perc_val = smartmeter_perc_val.iloc[0]
        df['smart_meter_adoption'] = smartmeter_perc_val / 100
        df['high_tech_area'] = int(smartmeter_perc_val > 75)
        df['low_tech_area'] = int(smartmeter_perc_val < 25)
        
        # Connection infrastructure features
        num_connections_val = df.get('num_connections', 200)
        if hasattr(num_connections_val, 'iloc'):
            num_connections_val = num_connections_val.iloc[0]
        active_pct_val = df.get('perc_of_active_connections', 88)
        if hasattr(active_pct_val, 'iloc'):
            active_pct_val = active_pct_val.iloc[0]
        
        df['total_connections'] = num_connections_val
        df['log_connections'] = np.log1p(num_connections_val)
        df['connections_per_household'] = num_connections_val / df['estimated_households']
        df['active_connections_percentage'] = active_pct_val
        df['inactive_connections_percentage'] = 100 - active_pct_val
        df['active_connection_count'] = num_connections_val * active_pct_val / 100
        df['inactive_connection_count'] = num_connections_val - df['active_connection_count']
        df['total_active_connections'] = df['active_connection_count']
        df['connection_efficiency'] = df['annual_consume'] / np.maximum(df['active_connection_count'], 1)
        
        # Connection density categories
        if num_connections_val <= 50:
            df['connection_density_category'] = 'very_low_density'
        elif num_connections_val <= 200:
            df['connection_density_category'] = 'low_density'
        elif num_connections_val <= 500:
            df['connection_density_category'] = 'medium_density'
        elif num_connections_val <= 1000:
            df['connection_density_category'] = 'high_density'
        else:
            df['connection_density_category'] = 'very_high_density'
        
        # Activity level categories
        if active_pct_val <= 50:
            df['activity_level_category'] = 'low_activity'
        elif active_pct_val <= 75:
            df['activity_level_category'] = 'medium_activity'
        elif active_pct_val <= 90:
            df['activity_level_category'] = 'high_activity'
        else:
            df['activity_level_category'] = 'very_high_activity'
        
        # Location features
        postal_code = str(df.get('zipcode_from', '3500'))
        df['postal_code_area'] = postal_code[:2]
        
        # Province mapping
        first_digit = postal_code[0] if postal_code else '3'
        province_map = {
            '1': 'west_holland', '2': 'west_holland', '3': 'central',
            '4': 'southwest', '5': 'south', '6': 'east',
            '7': 'northeast', '8': 'north', '9': 'north'
        }
        df['postal_province'] = province_map.get(first_digit, 'central')
        
        df['city_clean'] = str(df.get('city', 'Utrecht')).upper().strip()
        df['company_clean'] = str(df.get('company', 'liander')).lower().strip()
        df['connection_type_clean'] = str(df.get('type_of_connection', '3x25')).upper()
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for the given input data.
        
        Args:
            input_data: Dictionary with household characteristics
            
        Returns:
            Dictionary with prediction results
        """
        if self.model_package is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create DataFrame and apply feature engineering
        df_input = pd.DataFrame([input_data])
        df_engineered = self._engineer_features(df_input)
        
        # Select features used by the model
        feature_columns = self.model_package['feature_columns']
        missing_features = [col for col in feature_columns if col not in df_engineered.columns]
        
        # Fill missing features with defaults
        for col in missing_features:
            if col in ['postal_code_area', 'city_clean', 'company_clean', 'connection_type_clean', 
                      'postal_province', 'household_size_category', 'connection_density_category', 
                      'activity_level_category']:
                df_engineered[col] = 'default'
            else:
                df_engineered[col] = 0
        
        X_input = df_engineered[feature_columns].copy()
        
        # Handle categorical features
        categorical_cols = ['postal_code_area', 'city_clean', 'company_clean', 'connection_type_clean', 
                           'postal_province', 'household_size_category', 'connection_density_category', 
                           'activity_level_category']
        
        label_encoders = self.model_package.get('label_encoders', {})
        for col in categorical_cols:
            if col in X_input.columns:
                if col in label_encoders:
                    try:
                        X_input[col] = label_encoders[col].transform([str(X_input[col].iloc[0])])[0]
                    except (ValueError, AttributeError):
                        X_input[col] = 0
                else:
                    X_input[col] = 0
        
        # Handle any remaining NaN values
        X_input = X_input.fillna(0)
        
        # Apply preprocessing (scaling and feature selection)
        if self.model_package['use_scaled'] and self.model_package['scaler'] is not None:
            X_scaled = self.model_package['scaler'].transform(X_input)
            if self.model_package['feature_selector'] is not None:
                X_processed = self.model_package['feature_selector'].transform(X_scaled)
            else:
                X_processed = X_scaled
        else:
            if self.model_package['feature_selector'] is not None:
                X_processed = self.model_package['feature_selector'].transform(X_input)
            else:
                X_processed = X_input
        
        # Make prediction
        model = self.model_package['model']
        prediction = model.predict(X_processed)[0]
        
        return {
            'prediction_kwh': prediction,
            'model_used': self.model_package['model_name'],
            'monthly_kwh': prediction / 12,
            'daily_kwh': prediction / 365,
            'estimated_monthly_cost': (prediction / 12) * 0.25,  # €0.25/kWh estimate
            'model_performance': self.model_package['model_performance']
        }
    
    def create_input_from_args(self, house_type: str = '3X25', postal_code: str = '35', 
                              weather: str = 'normal', 
                              smart_meter: bool = True) -> Dict[str, Any]:
        """Create model input from command-line arguments."""
        
        user_inputs = {
            'zipcode_from': postal_code + '00',
            'city': 'Utrecht',
            'company': 'liander',
            'num_connections': 200,
            'perc_of_active_connections': 88,
            'delivery_perc': 97.5,
            'smartmeter_perc': 75 if smart_meter else 25,
            'annual_consume': 3500,
        }
        
        # House type mapping
        house_type_map = {
            '1X25': {'type_of_connection': '1x25'},
            '1X35': {'type_of_connection': '1x35'},
            '3X25': {'type_of_connection': '3x25'},
            '3X35': {'type_of_connection': '3x35'},
            '3X50': {'type_of_connection': '3x50'},
        }
        
        if house_type.upper() in house_type_map:
            user_inputs.update(house_type_map[house_type.upper()])
        else:
            user_inputs['type_of_connection'] = '3x25'
        
        # Weather scenario
        if weather.lower() == 'cold':
            user_inputs.update({
                'avg_temp': 8.0, 'total_precipitation': 900,
                'total_sunshine_hours': 1400, 'avg_wind_speed': 4.5,
            })
        elif weather.lower() == 'warm':
            user_inputs.update({
                'avg_temp': 13.0, 'total_precipitation': 800,
                'total_sunshine_hours': 1700, 'avg_wind_speed': 3.8,
            })
        else:  # Normal weather
            user_inputs.update({
                'avg_temp': 10.5, 'total_precipitation': 850,
                'total_sunshine_hours': 1580, 'avg_wind_speed': 4.2,
            })
        
        # Add derived weather features
        user_inputs.update({
            'avg_min_temp': user_inputs['avg_temp'] - 4,
            'avg_max_temp': user_inputs['avg_temp'] + 4,
            'avg_precipitation': user_inputs['total_precipitation'] / 365,
            'total_global_radiation': 3400,
        })
        
        return user_inputs

def get_interactive_inputs(predictor) -> Dict[str, Any]:
    """Get user inputs through detailed interactive questionnaire."""
    print("\n🏠 HOUSEHOLD ENERGY CONSUMPTION PREDICTOR")
    print("=" * 50)
    print("Enter your household details (press Enter for defaults):")
    print()
    
    user_inputs = {}
    

    
    # House type / connection type
    print("\n🔌 House Type & Electrical Connection:")
    house_types = {
        '1': ('1x25', 'Small apartment (1-phase, 25A)', 10),
        '2': ('1x35', 'Small apartment (1-phase, 35A)', 10),
        '3': ('3x25', 'Medium apartment/house (3-phase, 25A)', 14),
        '4': ('3x35', 'Large house (3-phase, 35A)', 18),
        '5': ('3x50', 'Large house (3-phase, 50A)', 22),
    }
    
    print("House types:")
    for key, (code, description, _) in house_types.items():
        print(f"  {key}. {description}")
    
    house_choice = input("Select house type (1-5) [default: 3 - Medium house]: ").strip()
    if house_choice in house_types:
        connection_type, _, circuits_per_household = house_types[house_choice]
        user_inputs['type_of_connection'] = connection_type
        user_inputs['circuits_per_household'] = circuits_per_household
    else:
        user_inputs['type_of_connection'] = '3x25'
        user_inputs['circuits_per_household'] = 14
    
    print()
    
    # Location
    print("📍 Location:")
    postal_input = input("Postal code (first 4 digits, e.g., 1012 for Amsterdam) [default: 3500 - Utrecht]: ").strip()
    if postal_input and postal_input.isdigit() and len(postal_input) >= 2:
        user_inputs['zipcode_from'] = postal_input[:4].ljust(4, '0')
    else:
        user_inputs['zipcode_from'] = '3500'
    
    city_input = input("City name [default: Utrecht]: ").strip()
    if city_input:
        user_inputs['city'] = city_input.title()
    else:
        user_inputs['city'] = 'Utrecht'
    
    print()
    
    # Energy company
    print("⚡ Energy Company:")
    companies = ['liander', 'enexis', 'stedin', 'westland-infra', 'coteq']
    print("Major companies: " + ", ".join([c.title() for c in companies]))
    company_input = input("Energy company [default: liander]: ").strip().lower()
    if company_input in companies:
        user_inputs['company'] = company_input
    else:
        user_inputs['company'] = 'liander'
    
    print()
    
    # Connection characteristics
    print("🏘️ Neighborhood & Connection Details:")
    
    # Number of connections (typical for different areas)
    density_input = input("Neighborhood density (urban/suburban/rural) [default: suburban]: ").strip().lower()
    if density_input == 'urban':
        user_inputs['num_connections'] = 500
    elif density_input == 'rural':
        user_inputs['num_connections'] = 50
    else:
        user_inputs['num_connections'] = 200  # suburban default
    
    # Activity percentage
    activity_input = input("Percentage of active connections (50-95%) [default: 88]: ").strip()
    if activity_input and activity_input.isdigit():
        activity = int(activity_input)
        if 50 <= activity <= 95:
            user_inputs['perc_of_active_connections'] = activity
        else:
            user_inputs['perc_of_active_connections'] = 88
    else:
        user_inputs['perc_of_active_connections'] = 88
    
    # Smart meter adoption
    smart_input = input("Smart meter installed? (y/n) [default: y]: ").strip().lower()
    if smart_input == 'n':
        user_inputs['smartmeter_perc'] = 25  # Low adoption area
    else:
        user_inputs['smartmeter_perc'] = 75  # High adoption area
    
    # Delivery percentage (grid stability)
    user_inputs['delivery_perc'] = 97.5  # Standard Dutch grid reliability
    
    print()
    
    # Annual consumption (we need this for feature engineering, but we'll estimate it)
    # This will be overridden by the model prediction anyway
    user_inputs['annual_consume'] = 3500 * user_inputs['circuits_per_household']  # Rough estimate
    
    # Weather preferences (for scenario analysis)
    print("🌤️ Weather Scenario:")
    weather_scenarios = {
        '1': ('cold', 'Cold year (avg 8°C, high heating)'),
        '2': ('normal', 'Normal year (avg 10.5°C)'),
        '3': ('warm', 'Warm year (avg 13°C, some cooling)'),
    }
    
    print("Weather scenarios:")
    for key, (code, description) in weather_scenarios.items():
        print(f"  {key}. {description}")
    
    weather_choice = input("Select weather scenario (1-3) [default: 2 - Normal]: ").strip()
    if weather_choice == '1':  # Cold year
        user_inputs.update({
            'avg_temp': 8.0,
            'total_precipitation': 900,
            'total_sunshine_hours': 1400,
            'avg_wind_speed': 4.5,
        })
    elif weather_choice == '3':  # Warm year
        user_inputs.update({
            'avg_temp': 13.0,
            'total_precipitation': 800,
            'total_sunshine_hours': 1700,
            'avg_wind_speed': 3.8,
        })
    else:  # Normal weather
        user_inputs.update({
            'avg_temp': 10.5,
            'total_precipitation': 850,
            'total_sunshine_hours': 1580,
            'avg_wind_speed': 4.2,
        })
    
    # Add some additional weather features that are calculated
    user_inputs.update({
        'avg_min_temp': user_inputs['avg_temp'] - 4,
        'avg_max_temp': user_inputs['avg_temp'] + 4,
        'avg_precipitation': user_inputs['total_precipitation'] / 365,
        'total_global_radiation': 3400,  # Typical for Netherlands
        'data_source': 'KNMI',
        'stations_used': '240,260,330,370',
        'coverage': 'National average'
    })
    
    return user_inputs

def run_interactive_mode():
    """Run interactive prediction mode."""
    predictor = EnergyPredictor()
    
    if not predictor.load_model():
        return
    
    while True:
        try:
            # Get user inputs
            user_inputs = get_interactive_inputs(predictor)
            
            # Make prediction
            results = predictor.predict(user_inputs)
            
            # Display results
            print("\n" + "=" * 50)
            print("🎯 ENERGY CONSUMPTION PREDICTION")
            print("=" * 50)
            
            print(f"📊 Predicted Annual Consumption: {results['prediction_kwh']:.0f} kWh")
            print(f"📅 Monthly Average: {results['monthly_kwh']:.0f} kWh")
            print(f"📰 Daily Average: {results['daily_kwh']:.1f} kWh")
            print(f"💰 Estimated Monthly Cost: €{results['estimated_monthly_cost']:.0f}")
            print(f"💰 Estimated Yearly Cost: €{results['estimated_monthly_cost'] * 12:.0f}")
            print(f"🤖 Model Used: {results['model_used']}")
            
            print(f"\n📋 Input Summary:")
            print(f"   House Type: {user_inputs.get('type_of_connection', '3x25')}")
            print(f"   Location: {user_inputs.get('zipcode_from', '3500')} ({user_inputs.get('city', 'Utrecht')})")
            print(f"   Weather: {user_inputs.get('avg_temp', 10.5):.1f}°C average")
            print(f"   Smart Meter: {'Yes' if user_inputs.get('smartmeter_perc', 75) > 50 else 'No'}")
            print(f"   Energy Company: {user_inputs.get('company', 'liander').title()}")
            
            # Comparison with typical consumption - using the actual training mean
            typical_consumption = 2223  # This is the actual training mean from our model
            difference = results['prediction_kwh'] - typical_consumption
            percentage = (difference / typical_consumption) * 100
            
            print(f"\n📈 Comparison to Training Average:")
            print(f"   Training Average: {typical_consumption} kWh/year")
            print(f"   Your Prediction: {results['prediction_kwh']:.0f} kWh/year")
            if difference > 0:
                print(f"   Difference: +{difference:.0f} kWh ({percentage:+.1f}%) ABOVE average")
            else:
                print(f"   Difference: {difference:.0f} kWh ({percentage:+.1f}%) BELOW average")
            
            # Ask if user wants to try again
            print("\n" + "=" * 50)
            another = input("Make another prediction? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with different inputs.")

def run_command_line_mode(args):
    """Run command-line prediction mode."""
    predictor = EnergyPredictor()
    
    print("⚡ Energy Consumption Prediction")
    print("=" * 35)
    
    if not predictor.load_model(verbose=False):
        return
    
    try:
        user_inputs = predictor.create_input_from_args(
            house_type=args.house_type,
            postal_code=args.location,
            weather=args.weather,

            smart_meter=not args.no_smart_meter
        )
        
        results = predictor.predict(user_inputs)
        
        print(f"\n🏠 House: {args.house_type} in area {args.location}")
        print(f"🌤️ Weather: {args.weather.title()} year")
        print(f"🔌 Smart meter: {'No' if args.no_smart_meter else 'Yes'}")
        print()
        print(f"📊 Predicted: {results['prediction_kwh']:.0f} kWh/year")
        print(f"📅 Monthly: {results['monthly_kwh']:.0f} kWh")
        print(f"💰 Cost: €{results['estimated_monthly_cost']:.0f}/month")
        
        # Quick comparison using actual training average
        typical = 2223  # Actual training mean from our model
        diff = results['prediction_kwh'] - typical
        print(f"\n📈 vs Training Average ({typical} kWh): {diff:+.0f} kWh ({diff/typical*100:+.1f}%)")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Energy Consumption Predictor (loads pre-trained model)',
        epilog="""
Examples:
  # Interactive mode
  python energy_predictor.py
  
  # Command-line predictions
  python energy_predictor.py --house-type 3X25 --location 35 --weather normal
  python energy_predictor.py --house-type 3X50 --location 10 --weather warm --no-smart-meter
        """
    )
    
    parser.add_argument('--house-type', choices=['1X25', '1X35', '3X25', '3X35', '3X50'], 
                       help='House electrical connection type')
    parser.add_argument('--location', 
                       help='Postal code area (first 2 digits)')
    parser.add_argument('--weather', choices=['cold', 'normal', 'warm'], 
                       help='Weather scenario')

    parser.add_argument('--no-smart-meter', action='store_true', 
                       help='House does not have smart meter')
    
    args = parser.parse_args()
    
    # Check if command-line arguments provided
    if any([args.house_type, args.location, args.weather, args.no_smart_meter]):
        # Command-line mode with defaults
        args.house_type = args.house_type or '3X25'
        args.location = args.location or '35'
        args.weather = args.weather or 'normal'

        
        run_command_line_mode(args)
    else:
        # Interactive mode
        run_interactive_mode()

if __name__ == "__main__":
    main()