"""
Example usage of the Energy Consumption ML Pipeline
"""

from model_training import EnergyConsumptionPredictor

def quick_ml_demo():
    """Run a quick demo of the ML pipeline."""
    
    print("🤖 Energy Consumption ML Pipeline Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EnergyConsumptionPredictor()
    
    # Load electricity data
    print("Loading electricity consumption data...")
    data = predictor.load_and_prepare_data("electricity", 2015, 2020)  # Use recent years
    print(f"✅ Loaded {len(data)} years of data")
    
    # Quick training on average consumption
    target = 'avg_annual_consume'
    print(f"\nTraining models to predict: {target}")
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(data, target, test_size=0.3)
    
    # Train models
    results = predictor.train_models(X_train, y_train, target)
    
    # Evaluate
    evaluation = predictor.evaluate_models(X_test, y_test, target, results)
    
    print(f"\n📊 Model Performance:")
    print(evaluation[['Model', 'Test_R2', 'Test_RMSE']].head(3).to_string(index=False))
    
    # Show feature importance
    print(f"\n🎯 Feature Importance:")
    importance = predictor.analyze_feature_importance(target)
    
    # Make a prediction
    print(f"\n🔮 Example Prediction:")
    
    # Typical Dutch weather conditions
    weather_example = {
        'avg_temp': 10.5,           # Typical Netherlands temperature
        'avg_min_temp': 6.8,
        'avg_max_temp': 14.2,
        'total_precipitation': 850,  # mm per year
        'avg_precipitation': 2.3,
        'total_sunshine_hours': 1580,
        'avg_wind_speed': 4.2,      # m/s
        'total_global_radiation': 3400
    }
    
    # Typical grid characteristics
    grid_example = {
        'avg_delivery_perc': 97.5,
        'avg_active_connections_perc': 88.0,
        'avg_smartmeter_perc': 75.0,  # Modern smart meter adoption
        'num_companies': 5
    }
    
    try:
        prediction = predictor.predict_consumption(weather_example, grid_example, target)
        print(f"   Weather: {weather_example['avg_temp']}°C avg, {weather_example['total_precipitation']}mm rain")
        print(f"   Predicted {target}: {prediction:.0f} kWh/year per connection")
        
        # Show what influences the prediction
        best_model = predictor.trained_models[target]['best_model_name']
        print(f"   (Using {best_model})")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
    
    print(f"\n✨ Demo completed!")

if __name__ == "__main__":
    quick_ml_demo()
