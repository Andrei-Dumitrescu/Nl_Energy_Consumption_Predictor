"""
FIXED Machine Learning Model Training for Dutch Energy Consumption Prediction

This version properly uses individual postal code areas as training samples,
giving us hundreds of thousands of data points instead of just a few national averages.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from dataset import DutchEnergyDataset, KNMIWeatherData

class FixedEnergyConsumptionPredictor:
    """
    Fixed ML pipeline that uses individual postal code areas as training samples.
    """
    
    def __init__(self):
        """Initialize the predictor."""
        self.energy_dataset = DutchEnergyDataset()
        self.weather_data = KNMIWeatherData()
        self.trained_models = {}
        self.scaler = None
        
    def prepare_training_data(self, energy_type: str = "electricity", 
                            years: List[int] = [2018, 2019, 2020]) -> pd.DataFrame:
        """
        Prepare training data where each row = one postal code area.
        
        Args:
            energy_type: "electricity" or "gas"
            years: List of years to include
            
        Returns:
            DataFrame with thousands of training samples
        """
        print(f"Preparing training data for {energy_type}...")
        
        # Load energy data for all specified years
        all_energy_data = []
        for year in years:
            print(f"  Loading {year} data...")
            year_data = self.energy_dataset.load_year_data(year, energy_type)
            if not year_data.empty:
                year_data['year'] = year
                all_energy_data.append(year_data)
        
        if not all_energy_data:
            raise ValueError(f"No energy data found for years {years}")
        
        # Combine all years
        energy_df = pd.concat(all_energy_data, ignore_index=True)
        print(f"  Combined energy data: {len(energy_df):,} postal code areas")
        
        # Load weather data for the same years
        weather_df = self.weather_data.get_national_annual_weather(min(years), max(years))
        if weather_df.empty:
            raise ValueError("No weather data available")
        
        # Merge energy data with weather data by year
        combined_df = pd.merge(energy_df, weather_df, on='year', how='inner')
        print(f"  After weather merge: {len(combined_df):,} samples")
        
        # Feature engineering
        combined_df = self._engineer_features(combined_df)
        
        # Remove samples with missing critical data
        critical_columns = [
            'annual_consume', 'num_connections', 'avg_temp', 'total_precipitation',
            'delivery_perc', 'perc_of_active_connections', 'smartmeter_perc'
        ]
        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=critical_columns)
        print(f"  After removing missing data: {len(combined_df):,} samples (removed {initial_count - len(combined_df):,})")
        
        # Calculate preliminary household consumption for outlier removal
        def estimate_households_preliminary(row):
            connection_type = str(row.get('type_of_connection', '1x25')).upper()
            num_connections = row['num_connections']
            
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
                
            return max(num_connections / circuits_per_household, 0.1)
        
        combined_df['preliminary_estimated_households'] = combined_df.apply(estimate_households_preliminary, axis=1)
        combined_df['preliminary_household_consumption'] = combined_df['annual_consume'] / combined_df['preliminary_estimated_households']
        
        # Enhanced data filtering for realistic household consumption ranges
        initial_count = len(combined_df)
        
        # Filter 1: Remove areas with very low activity (keep this filter as it indicates data quality)
        combined_df = combined_df[combined_df['perc_of_active_connections'] >= 30]  # At least 30% active
        
        # Filter 2: Remove unrealistic household consumption values
        combined_df = combined_df[
            (combined_df['preliminary_household_consumption'] >= 500) &      # At least 500 kWh/year per household
            (combined_df['preliminary_household_consumption'] <= 8000)       # Max 8,000 kWh/year per household
        ]
        
        # Filter 3: IQR-based outlier removal within connection types (more conservative)
        def remove_outliers_iqr(df, column, factor=2.0):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        # Apply connection-type specific outlier removal
        if 'type_of_connection' in combined_df.columns:
            connection_groups = combined_df.groupby('type_of_connection')
            filtered_groups = []
            for name, group in connection_groups:
                if len(group) > 20:  # Only apply IQR if enough samples
                    group_filtered = remove_outliers_iqr(group, 'preliminary_household_consumption', factor=2.0)
                    filtered_groups.append(group_filtered)
                else:
                    filtered_groups.append(group)
            combined_df = pd.concat(filtered_groups, ignore_index=True)
        
        print(f"  After enhanced filtering: {len(combined_df):,} samples (removed {initial_count - len(combined_df):,})")
        
        return combined_df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features focused on household-level consumption prediction using connection-to-household estimation."""
        df = df.copy()
        
        # HOUSEHOLD CONSUMPTION CALCULATION:
        # Convert from connection-level to household-level consumption
        # Based on connection type patterns: different building types have different circuits per household
        
        def estimate_households_from_connections(row):
            """Estimate number of households based on connection type and count."""
            connection_type = str(row.get('type_of_connection', '1x25')).upper()
            num_connections = row['num_connections']
            
            # Connection density patterns based on building types:
            if '1X25' in connection_type or '1X35' in connection_type:
                # Small apartments: ~8-12 circuits per household
                circuits_per_household = 10
            elif '3X25' in connection_type:
                # Medium apartments/small houses: ~12-16 circuits per household  
                circuits_per_household = 14
            elif '3X35' in connection_type:
                # Houses: ~16-20 circuits per household
                circuits_per_household = 18
            elif '3X50' in connection_type or '3X63' in connection_type:
                # Large houses: ~20-25 circuits per household
                circuits_per_household = 22
            else:
                # Default fallback
                circuits_per_household = 12
                
            estimated_households = num_connections / circuits_per_household
            return max(estimated_households, 0.1)  # Avoid division by zero
        
        # Calculate estimated households and consumption per household
        df['estimated_households'] = df.apply(estimate_households_from_connections, axis=1)
        df['household_consumption'] = df['annual_consume'] / df['estimated_households']
        
        # ===== TEMPORAL FEATURES (Year trends & decade indicators) =====
        # Year trend variables (energy efficiency improvements over time)
        df['year_trend'] = df['year'] - 2009  # Normalize to start from 0
        df['year_squared'] = df['year_trend'] ** 2  # Non-linear time trends
        
        # Decade indicators (2010s vs late 2000s technology differences)
        df['decade_2000s'] = (df['year'] <= 2010).astype(int)  # Early period
        df['decade_2010s'] = (df['year'] > 2010).astype(int)   # Later period with better tech
        
        # Technology adoption era
        df['early_smart_era'] = (df['year'] <= 2015).astype(int)  # Pre-smart meter widespread adoption
        df['smart_era'] = (df['year'] > 2015).astype(int)         # Smart meter era
        
        # LOCATION FEATURES - Let the model learn location patterns itself
        # Use postal code first 2 digits for more granular location info
        df['postal_code_area'] = df['zipcode_from'].astype(str).str[:2]
        
        # Keep city as-is for the model to learn patterns
        df['city_clean'] = df['city'].str.upper().str.strip()
        
        # COMPANY FEATURES - Let the model learn company patterns itself
        # Keep company names as-is without arbitrary groupings
        df['company_clean'] = df['company'].str.lower().str.strip()
        
        # ===== ENHANCED CONNECTION TYPE FEATURES =====
        if 'type_of_connection' in df.columns:
            df['connection_type_clean'] = df['type_of_connection'].astype(str).str.upper().str.strip()
            
            # Extract capacity indicators (proxy for house size)
            df['connection_phases'] = df['connection_type_clean'].str.extract(r'(\d+)x', expand=False).fillna('1').astype(int)
            df['connection_amperage'] = df['connection_type_clean'].str.extract(r'x(\d+)', expand=False).fillna('25').astype(int)
            df['total_electrical_capacity'] = df['connection_phases'] * df['connection_amperage']
            
            # Extract voltage levels from connection types (proxy for building type)
            df['has_high_voltage'] = df['connection_type_clean'].str.contains('400|500|1000', na=False).astype(int)
            df['has_medium_voltage'] = df['connection_type_clean'].str.contains('230|240|250', na=False).astype(int)
            
            # Create household size categories from electrical capacity
            def categorize_household_size(capacity):
                if capacity <= 25:
                    return 'small_apartment'     # 1x25, small apartments
                elif capacity <= 50:
                    return 'medium_apartment'    # 2x25, medium apartments  
                elif capacity <= 75:
                    return 'large_apartment'     # 3x25, large apartments
                elif capacity <= 100:
                    return 'small_house'         # 3x35, small houses
                else:
                    return 'large_house'         # 3x50+, large houses
            
            df['household_size_category'] = df['total_electrical_capacity'].apply(categorize_household_size)
            
            # Smart meter adoption as technology indicator
            if 'smartmeter_perc' in df.columns:
                df['smart_meter_adoption'] = df['smartmeter_perc'] / 100
                df['high_tech_area'] = (df['smartmeter_perc'] > 75).astype(int)
                df['low_tech_area'] = (df['smartmeter_perc'] < 25).astype(int)
        
        # WEATHER FEATURES (National averages only - no location-specific interactions)
        df['temp_squared'] = df['avg_temp'] ** 2
        df['heating_degree_days'] = np.maximum(18 - df['avg_temp'], 0) * 365
        df['cooling_degree_days'] = np.maximum(df['avg_temp'] - 22, 0) * 365
        
        # Weather intensity indicators (national level)
        df['extreme_cold_indicator'] = (df['avg_temp'] < 5).astype(int)  # Very cold years
        df['extreme_hot_indicator'] = (df['avg_temp'] > 15).astype(int)   # Very warm years
        df['high_precipitation'] = (df['total_precipitation'] > 900).astype(int)  # Wet years
        df['low_sunshine'] = (df['total_sunshine_hours'] < 1400).astype(int)     # Dark years
        
        # Simple weather features only (no location-dependent interactions)
        df['sunshine_ratio'] = df['total_sunshine_hours'] / 1800  # Normalize to typical max
        df['heating_cooling_balance'] = df['heating_degree_days'] - df['cooling_degree_days']
        
        # GEOGRAPHIC CLUSTERING FEATURES (Option 4)
        # Create broader geographic regions for similar areas to learn from each other
        df['postal_first_digit'] = df['zipcode_from'].astype(str).str[0]
        df['postal_province'] = df['postal_first_digit'].map({
            '1': 'west_holland',      # Amsterdam, Den Haag area
            '2': 'west_holland',      # Rotterdam, Zuid-Holland
            '3': 'central',           # Utrecht, Flevoland
            '4': 'southwest',         # Zeeland, West-Brabant
            '5': 'south',             # Oost-Brabant, Limburg
            '6': 'east',              # Gelderland, Overijssel
            '7': 'northeast',         # Drenthe, Groningen
            '8': 'north',             # Friesland, Noord-Holland Noord
            '9': 'north'              # Noord-Holland
        }).fillna('other')
        
        # ===== CONNECTION-BASED FEATURES =====
        # Use raw connection counts and activity percentages as features
        # These provide direct insights into area density and utilization
        
        # Connection scale features
        df['total_connections'] = df['num_connections']  # Raw connection count
        df['log_connections'] = np.log1p(df['num_connections'])  # Log transform for better scaling
        df['connections_per_household'] = df['num_connections'] / df['estimated_households']
        
        # Activity and utilization features  
        df['active_connections_percentage'] = df['perc_of_active_connections']  # Raw percentage
        df['inactive_connections_percentage'] = 100 - df['perc_of_active_connections']
        df['active_connection_count'] = (df['num_connections'] * df['perc_of_active_connections'] / 100)
        df['inactive_connection_count'] = df['num_connections'] - df['active_connection_count']
        
        # Connection density categories
        def categorize_connection_density(connections):
            if connections <= 50:
                return 'very_low_density'      # Rural/sparse areas
            elif connections <= 200:
                return 'low_density'           # Suburban areas
            elif connections <= 500:
                return 'medium_density'        # Urban neighborhoods 
            elif connections <= 1000:
                return 'high_density'          # Dense urban areas
            else:
                return 'very_high_density'     # City centers/high-rises
        
        df['connection_density_category'] = df['num_connections'].apply(categorize_connection_density)
        
        # Activity level categories
        def categorize_activity_level(activity_pct):
            if activity_pct <= 50:
                return 'low_activity'      # <50% active
            elif activity_pct <= 75:
                return 'medium_activity'   # 50-75% active
            elif activity_pct <= 90:
                return 'high_activity'     # 75-90% active
            else:
                return 'very_high_activity' # >90% active
        
        df['activity_level_category'] = df['perc_of_active_connections'].apply(categorize_activity_level)
        
        # Interaction features - connections × activity
        df['total_active_connections'] = df['active_connection_count']
        df['connection_efficiency'] = df['annual_consume'] / np.maximum(df['active_connection_count'], 1)
        
        return df
    
    def get_feature_target_columns(self):
        """Define features with multicollinearity reduction and all enhancements."""
        feature_columns = [
            # Temporal features (energy efficiency improvements over time)
            'year_trend', 'year_squared', 'decade_2000s', 'decade_2010s',
            'early_smart_era', 'smart_era',
            
            # Weather features (reduced multicollinearity - removed temp_squared and heating_degree_days)
            'avg_temp', 'total_precipitation', 'total_sunshine_hours', 'avg_wind_speed',
            'cooling_degree_days', 'sunshine_ratio', 'heating_cooling_balance',
            
            # Weather intensity indicators (national level)
            'extreme_cold_indicator', 'extreme_hot_indicator', 
            'high_precipitation', 'low_sunshine',
            
            # Enhanced connection type features (proxy for house size/type)
            'connection_phases', 'connection_amperage', 'total_electrical_capacity',
            'has_high_voltage', 'has_medium_voltage', 'household_size_category',
            'connection_type_clean',
            
            # Smart meter technology features
            'smart_meter_adoption', 'high_tech_area', 'low_tech_area',
            
            # NEW: Connection-based features (infrastructure and utilization)
            'total_connections', 'log_connections', 'connections_per_household',
            'active_connections_percentage', 'inactive_connections_percentage',
            'active_connection_count', 'inactive_connection_count',
            'connection_density_category', 'activity_level_category',
            'total_active_connections', 'connection_efficiency',
            
            # Location features (independent of weather)
            'postal_code_area',  # 2-digit postal code areas
            'postal_province',   # Broader geographic clustering
            'city_clean',        # Specific cities
            'company_clean',     # Energy companies
        ]
        
        target_columns = [
            'household_consumption',     # kWh per household (estimated from connection patterns)
        ]
        
        return feature_columns, target_columns
    
    def train_model(self, data: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2) -> Dict:
        """
        Train models on the prepared data.
        
        Args:
            data: Prepared training data
            target_column: Target variable to predict
            test_size: Proportion for test set
            
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining models for {target_column}...")
        
        feature_columns, _ = self.get_feature_target_columns()
        
        # Filter features that exist in the data
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"Available features: {available_features}")
        
        # Prepare data and handle categorical + numerical features
        X = data[available_features].copy()
        y = data[target_column].copy()
        
        # Identify categorical and numerical columns
        categorical_cols = ['postal_code_area', 'city_clean', 'company_clean', 'connection_type_clean', 
                           'postal_province', 'household_size_category', 'connection_density_category', 
                           'activity_level_category']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Handle categorical features - encode for all models
        self.label_encoders = {}
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Handle any remaining NaN values in numerical columns
        for col in numerical_cols:
            if col in X.columns and X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} NaN values with median: {median_val:.2f}")
        
        print(f"Using {len(available_features)} features ({len(categorical_cols)} categorical, {len(numerical_cols)} numerical)")
        print(f"Training on {len(X):,} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features with RobustScaler (better for outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection using LASSO (remove redundant features)
        print("  Performing feature selection...")
        lasso_selector = SelectFromModel(Lasso(alpha=0.01), threshold='median')
        lasso_selector.fit(X_train_scaled, y_train)
        
        # Get selected features
        selected_features = lasso_selector.get_support()
        selected_feature_names = [available_features[i] for i in range(len(available_features)) if selected_features[i]]
        print(f"  Selected {len(selected_feature_names)} features via LASSO: {selected_feature_names[:5]}...")
        
        # Apply feature selection
        X_train_selected = lasso_selector.transform(X_train_scaled)
        X_test_selected = lasso_selector.transform(X_test_scaled)
        
        self.feature_selector = lasso_selector
        
        # Define models with hyperparameter tuning
        models_config = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {},
                'use_scaled': True
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'use_scaled': True
            },
            'Lasso Regression': {
                'model': Lasso(),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
                'use_scaled': True
            },
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=50,      # Reduced from 100 for speed
                    max_depth=15,         # Fixed optimal depth
                    min_samples_split=5,  # Prevent overfitting
                    min_samples_leaf=2,   # Prevent overfitting
                    random_state=42, 
                    n_jobs=1,            # Single thread to avoid multiprocessing issues
                    max_features='sqrt'   # Faster feature selection
                ),
                'params': {},            # No hyperparameter tuning for speed
                'use_scaled': False
            }
        }
        
        # Time-based cross-validation (since we have temporal data)
        if 'year' in data.columns:
            # Use TimeSeriesSplit for temporal data
            cv_splitter = TimeSeriesSplit(n_splits=3)
            print("  Using time-series cross-validation")
        else:
            cv_splitter = 5
            print("  Using standard cross-validation")
        
        results = {}
        for name, config in models_config.items():
            print(f"  Training {name}...")
            
            model = config['model']
            params = config['params']
            use_scaled = config['use_scaled']
            
            # Choose appropriate data
            if use_scaled:
                X_train_model = X_train_selected
                X_test_model = X_test_selected
            else:
                # For tree-based models, use original data with feature selection
                X_train_original_selected = X_train.iloc[:, selected_features]
                X_test_original_selected = X_test.iloc[:, selected_features]
                X_train_model = X_train_original_selected
                X_test_model = X_test_original_selected
            
            # For Random Forest with large datasets, use sampling for speed
            if 'Random Forest' in name and len(X_train_model) > 100000:
                print(f"    Large dataset detected ({len(X_train_model):,} samples), sampling 50k for Random Forest training...")
                sample_size = min(50000, len(X_train_model))
                sample_idx = np.random.choice(len(X_train_model), sample_size, replace=False)
                
                if use_scaled:
                    X_train_model_sampled = X_train_model[sample_idx]
                else:
                    X_train_model_sampled = X_train_model.iloc[sample_idx]
                y_train_sampled = y_train.iloc[sample_idx]
                
                # Use sampled data for training but full data for evaluation
                X_train_for_training = X_train_model_sampled
                y_train_for_training = y_train_sampled
            else:
                X_train_for_training = X_train_model
                y_train_for_training = y_train
            
            # Hyperparameter tuning with GridSearchCV
            if params:
                print(f"    Tuning hyperparameters...")
                grid_search = GridSearchCV(
                    model, params, cv=cv_splitter, scoring='r2', 
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_for_training, y_train_for_training)
                best_model = grid_search.best_estimator_
                print(f"    Best params: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train_for_training, y_train_for_training)
            
            # Make predictions
            y_pred_train = best_model.predict(X_train_model)
            y_pred_test = best_model.predict(X_test_model)
            
            # Cross-validation with best model (sample for speed with large datasets)
            if 'Random Forest' in name:
                # For Random Forest, use smaller CV sample for speed
                sample_size = min(3000, len(X_train_model))
                sample_idx = np.random.choice(len(X_train_model), sample_size, replace=False)
                if use_scaled:
                    cv_scores = cross_val_score(
                        best_model, X_train_model[sample_idx], y_train.iloc[sample_idx], 
                        cv=3, scoring='r2'  # Reduced CV folds for RF
                    )
                else:
                    cv_scores = cross_val_score(
                        best_model, X_train_model.iloc[sample_idx], y_train.iloc[sample_idx], 
                        cv=3, scoring='r2'  # Reduced CV folds for RF
                    )
            elif len(X_train_model) > 10000:
                sample_size = 5000
                sample_idx = np.random.choice(len(X_train_model), sample_size, replace=False)
                cv_scores = cross_val_score(
                    best_model, X_train_model[sample_idx], y_train.iloc[sample_idx], 
                    cv=cv_splitter, scoring='r2'
                )
            else:
                cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=cv_splitter, scoring='r2')
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {
                'model': best_model,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'use_scaled': use_scaled
            }
            
            print(f"    CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"    Test R²: {test_r2:.3f}, RMSE: {test_rmse:.1f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        print(f"\nBest model: {best_model_name} (Test R²: {results[best_model_name]['test_r2']:.3f})")
        
        # Store results
        self.trained_models[target_column] = {
            'best_model_name': best_model_name,
            'best_model': results[best_model_name]['model'],
            'feature_columns': available_features,
            'results': results,
            'test_data': (X_test, y_test)
        }
        
        return results
    
    def analyze_feature_importance(self, target_column: str) -> pd.DataFrame:
        """Analyze feature importance for the best model."""
        if target_column not in self.trained_models:
            print(f"No trained model found for {target_column}")
            return pd.DataFrame()
        
        model_info = self.trained_models[target_column]
        model = model_info['best_model']
        all_feature_names = model_info['feature_columns']
        
        # Get the selected features after feature selection
        if hasattr(self, 'feature_selector'):
            selected_features = self.feature_selector.get_support()
            selected_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if selected_features[i]]
        else:
            selected_feature_names = all_feature_names
        
        importance_df = pd.DataFrame({'Feature': selected_feature_names})
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance_df['Importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_df['Importance'] = np.abs(model.coef_)
        else:
            print(f"Model {type(model).__name__} does not have feature importance")
            return pd.DataFrame()
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 features for {target_column} (after feature selection):")
        print(f"Using {len(selected_feature_names)} selected features out of {len(all_feature_names)} total")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df


def main():
    """Demo the household consumption prediction with REGIONAL + WEATHER features."""
    print("🏠 HOUSEHOLD ENERGY CONSUMPTION PREDICTOR")
    print("🌍 REGIONAL + WEATHER Features (Testing Geographic Patterns)")
    print("=" * 60)
    
    # Initialize predictor
    predictor = FixedEnergyConsumptionPredictor()
    
    # Prepare data
    all_years = list(range(2009, 2021))
    data = predictor.prepare_training_data("electricity", [2019,2020])
    
    print(f"\n📊 Dataset Overview:")
    print(f"Training samples: {len(data):,}")
    print(f"Household consumption range: {data['household_consumption'].min():.0f} - {data['household_consumption'].max():.0f} kWh/household/year")
    print(f"Mean household consumption: {data['household_consumption'].mean():.0f} kWh/household/year")
    print(f"Note: Household estimates based on connection type patterns (circuits per dwelling)")
    
    # Train enhanced models with all improvements
    print(f"\n{'='*60}")
    print("🚀 COMPREHENSIVE ENHANCED PREDICTION MODEL")
    print("✅ Temporal features (year trends, decade indicators)")
    print("✅ Enhanced connection features (voltage, household categories)")
    print("✅ NEW: Connection infrastructure features (total connections, activity %)")
    print("✅ NEW: Connection density & utilization categories")
    print("✅ NEW: Connection efficiency metrics")
    print("✅ Smart meter adoption indicators")
    print("✅ IQR-based outlier removal (connection-type specific)")
    print("✅ RobustScaler (better for outliers)")
    print("✅ LASSO feature selection (multicollinearity reduction)")
    print("✅ GridSearchCV hyperparameter tuning")
    print("✅ Time-series cross-validation")
    print('='*60)
    
    target = 'household_consumption'
    results = predictor.train_model(data, target)
    predictor.analyze_feature_importance(target)
    
    # Show sample predictions
    print(f"\n📊 Sample Predictions vs Actual (Household Consumption):")
    model_info = predictor.trained_models[target]
    X_test, y_test = model_info['test_data']
    best_model = model_info['best_model']
    
    # Make predictions on test set
    if results[model_info['best_model_name']]['use_scaled']:
        # Use feature-selected and scaled data
        X_test_scaled = predictor.scaler.transform(X_test)
        X_test_selected = predictor.feature_selector.transform(X_test_scaled)
        y_pred = best_model.predict(X_test_selected)
    else:
        # Use feature-selected original data
        selected_features = predictor.feature_selector.get_support()
        X_test_selected = X_test.iloc[:, selected_features]
        y_pred = best_model.predict(X_test_selected)
    
    # Show random sample of predictions
    sample_indices = np.random.choice(len(y_test), 10, replace=False)
    comparison = pd.DataFrame({
        'Actual_kWh_household': y_test.iloc[sample_indices].values,
        'Predicted_kWh_household': y_pred[sample_indices],
        'Error_kWh': y_test.iloc[sample_indices].values - y_pred[sample_indices],
        'Error_%': ((y_test.iloc[sample_indices].values - y_pred[sample_indices]) / y_test.iloc[sample_indices].values * 100)
    })
    
    print(comparison.round(1))
    
    print(f"\n🎯 Model Performance:")
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_r2 = results[best_model_name]['test_r2']
    best_rmse = results[best_model_name]['test_rmse']
    
    print(f"Best Model: {best_model_name}")
    print(f"R²: {best_r2:.3f} ({best_r2*100:.1f}% of variance explained)")
    print(f"RMSE: {best_rmse:.0f} kWh/household/year")
    print(f"Mean household consumption: {data['household_consumption'].mean():.0f} kWh/household/year")
    print(f"Relative error: {(best_rmse / data['household_consumption'].mean() * 100):.1f}%")
    
    print(f"\n✅ Success! Predicting household energy consumption with comprehensive feature engineering!")
    print(f"Note: Model estimates households from electrical connection patterns per building type.")

if __name__ == "__main__":
    main()
