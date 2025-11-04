import yaml
import os
import joblib
from datetime import datetime
from data.dataset_generator import DatasetGenerator
from data.preprocessor import DataPreprocessor
from models.linear_regression import ESGLinearRegression
from models.random_forest import ESGRandomForest
from models.xgboost_model import ESGXGBoost
from utils.metrics import calculate_metrics, print_metrics
from utils.visualization import plot_feature_importance, plot_predictions

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_dirs():
    """Create necessary output directories"""
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)

def train_model(model_type, X_train, X_test, y_train, y_test, feature_names, config):
    """Train and evaluate a model"""
    # Initialize model
    if model_type == 'linear_regression':
        model = ESGLinearRegression()
    elif model_type == 'random_forest':
        model = ESGRandomForest(
            n_estimators=config['models']['random_forest']['n_estimators'],
            random_state=config['random_state']
        )
    elif model_type == 'xgboost':
        model = ESGXGBoost(
            n_estimators=config['models']['xgboost']['n_estimators'],
            learning_rate=config['models']['xgboost']['learning_rate'],
            max_depth=config['models']['xgboost']['max_depth'],
            random_state=config['random_state']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"\nTraining {model_type}...")
    if model_type == 'xgboost':
        # For XGBoost, we can use early stopping
        eval_set = [(X_test, y_test)] if config['use_early_stopping'] else None
        model.train(X_train, y_train, eval_set=eval_set, 
                   early_stopping_rounds=config['early_stopping_rounds'])
    else:
        model.train(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Print metrics
    print(f"\n{model_type.upper()} - Training Metrics:")
    print_metrics(train_metrics)
    print(f"\n{model_type.upper()} - Test Metrics:")
    print_metrics(test_metrics)
    
    # Plot feature importance if available
    feature_importance = model.get_feature_importance(feature_names)
    if feature_importance:
        plot_feature_importance(feature_importance, f"{model_type.upper()} - Feature Importance")
    
    # Plot predictions vs actual
    plot_predictions(y_test, y_pred_test, f"{model_type.upper()} Predictions")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/saved_models/{model_type}_{timestamp}.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, test_metrics

def main():
    # Load configuration
    config = load_config()
    
    # Create output directories
    create_output_dirs()
    
    # Generate sample data
    print("Generating sample data...")
    data_gen = DatasetGenerator()
    df = data_gen.generate_sample_data(n_samples=config['n_samples'])
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        df, 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    # Get feature names
    feature_names = preprocessor.feature_columns
    
    # Train and evaluate models
    best_model = None
    best_score = -float('inf')
    
    for model_type in config['models_to_train']:
        try:
            model, metrics = train_model(
                model_type, X_train, X_test, y_train, y_test, 
                feature_names, config
            )
            
            # Track best model based on R2 score
            if metrics['r2'] > best_score:
                best_score = metrics['r2']
                best_model = model
                
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
