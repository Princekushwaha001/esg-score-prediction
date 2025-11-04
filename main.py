# #!/usr/bin/env python3
# """
# ESG Score Prediction - Main Entry Point
# """
# import argparse
# import yaml
# import os
# from pathlib import Path
#
# from data.dataset_generator import ESGDatasetGenerator
# from data.preprocessor import DataPreprocessor
# from models.linear_regression import LinearRegressionModel
# from models.random_forest import RandomForestModel
# from models.xgboost_model import XGBoostModel
# from utils.metrics import ModelEvaluator
# from utils.visualization import Visualizer
# import pandas as pd
# import numpy as np
# import joblib
# import json
#
#
# def load_config(config_path: str = 'config.yaml') -> dict:
#     """Load configuration from YAML file."""
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)
#
#
# def create_directories(config: dict):
#     """Create necessary output directories."""
#     dirs = [
#         'outputs',
#         config['output']['models_dir'],
#         config['output']['metrics_dir'],
#         config['output']['visualizations_dir']
#     ]
#     for d in dirs:
#         Path(d).mkdir(parents=True, exist_ok=True)
#
#
# def generate_dataset_step(config: dict):
#     """Generate and save ESG dataset."""
#     print("\n" + "=" * 60)
#     print("STEP 1: Generating Dataset")
#     print("=" * 60)
#
#     generator = ESGDatasetGenerator(random_seed=config['dataset']['random_seed'])
#     df = generator.generate(num_companies=config['dataset']['num_companies'])
#
#     output_path = config['dataset']['output_path']
#     generator.save_to_csv(df, output_path)
#
#     print(f"\n✓ Generated {len(df)} companies")
#     print(f"✓ Dataset saved to {output_path}")
#     print(f"\nDataset Statistics:")
#     print(df.describe())
#
#     return df
#
#
# def train_models_step(df: pd.DataFrame, config: dict):
#     """Train all ML models."""
#     print("\n" + "=" * 60)
#     print("STEP 2: Training Models")
#     print("=" * 60)
#
#     # Prepare data
#     preprocessor = DataPreprocessor(
#         test_size=config['training']['test_size'],
#         random_state=config['training']['random_state']
#     )
#
#     X, y = preprocessor.prepare_features(df)
#     X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)
#
#     print(f"\nTraining set: {len(X_train)} samples")
#     print(f"Test set: {len(X_test)} samples")
#
#     models = {}
#
#     # Train Linear Regression
#     print("\n[1/3] Training Linear Regression...")
#     lr_model = LinearRegressionModel()
#     lr_model.fit(X_train, y_train)
#     models['linear_regression'] = lr_model
#     print("✓ Linear Regression trained")
#
#     # Train Random Forest
#     print("\n[2/3] Training Random Forest...")
#     rf_config = config['models']['random_forest']
#     rf_model = RandomForestModel(
#         n_estimators=rf_config['n_estimators'],
#         max_depth=rf_config['max_depth'],
#         min_samples_split=rf_config['min_samples_split'],
#         random_state=rf_config['random_state']
#     )
#     rf_model.fit(X_train, y_train)
#     models['random_forest'] = rf_model
#     print("✓ Random Forest trained")
#
#     # Train XGBoost
#     print("\n[3/3] Training XGBoost...")
#     xgb_config = config['models']['xgboost']
#     xgb_model = XGBoostModel(
#         n_estimators=xgb_config['n_estimators'],
#         max_depth=xgb_config['max_depth'],
#         learning_rate=xgb_config['learning_rate'],
#         random_state=xgb_config['random_state']
#     )
#     xgb_model.fit(X_train, y_train)
#     models['xgboost'] = xgb_model
#     print("✓ XGBoost trained")
#
#     # Save models
#     models_dir = config['output']['models_dir']
#     for name, model in models.items():
#         model_path = f"{models_dir}/{name}_model.pkl"
#         joblib.dump(model, model_path)
#         print(f"✓ Saved {name} to {model_path}")
#
#     return models, (X_train, X_test, y_train, y_test)
#
#
# def evaluate_models_step(models: dict, test_data: tuple, config: dict):
#     """Evaluate all models and generate reports."""
#     print("\n" + "=" * 60)
#     print("STEP 3: Evaluating Models")
#     print("=" * 60)
#
#     X_train, X_test, y_train, y_test = test_data
#     evaluator = ModelEvaluator()
#     all_metrics = {}
#
#     print("\nModel Performance on Test Set:\n")
#     print(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
#     print("-" * 50)
#
#     for name, model in models.items():
#         y_pred = model.predict(X_test)
#         metrics = evaluator.evaluate_all(y_test, y_pred)
#         all_metrics[name] = metrics
#
#         print(f"{name:<20} {metrics['r2_score']:<10.4f} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f}")
#
#     # Save metrics
#     metrics_path = f"{config['output']['metrics_dir']}/evaluation_results.json"
#     with open(metrics_path, 'w') as f:
#         json.dump(all_metrics, f, indent=2)
#     print(f"\n✓ Metrics saved to {metrics_path}")
#
#     # Generate visualizations
#     print("\n" + "=" * 60)
#     print("STEP 4: Generating Visualizations")
#     print("=" * 60)
#
#     viz = Visualizer()
#     viz_dir = config['output']['visualizations_dir']
#
#     # Model comparison
#     viz.plot_model_comparison(
#         all_metrics,
#         save_path=f"{viz_dir}/model_comparison.png"
#     )
#
#     # Predictions vs Actual for each model
#     for name, model in models.items():
#         y_pred = model.predict(X_test)
#         viz.plot_predictions_vs_actual(
#             y_test, y_pred, name,
#             save_path=f"{viz_dir}/{name}_predictions.png"
#         )
#
#     print("\n✓ All visualizations generated")
#
#     return all_metrics
#
#
# def main():
#     parser = argparse.ArgumentParser(description='ESG Score Prediction System')
#     parser.add_argument('--config', default='config.yaml', help='Path to config file')
#     parser.add_argument('--generate-dataset', action='store_true', help='Generate new dataset')
#     parser.add_argument('--train', action='store_true', help='Train models')
#     parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
#     parser.add_argument('--all', action='store_true', help='Run complete pipeline')
#     parser.add_argument('--num-companies', type=int, help='Override number of companies')
#
#     args = parser.parse_args()
#
#     # Load configuration
#     config = load_config(args.config)
#
#     # Override config if specified
#     if args.num_companies:
#         config['dataset']['num_companies'] = args.num_companies
#
#     # Create output directories
#     create_directories(config)
#
#     print("\n" + "=" * 60)
#     print("ESG SCORE PREDICTION SYSTEM")
#     print("=" * 60)
#
#     # Execute steps
#     if args.all or args.generate_dataset:
#         df = generate_dataset_step(config)
#     else:
#         # Load existing dataset
#         df = pd.read_csv(config['dataset']['output_path'])
#
#     if args.all or args.train:
#         models, test_data = train_models_step(df, config)
#     else:
#         # Load existing models
#         models_dir = config['output']['models_dir']
#         models = {
#             'linear_regression': joblib.load(f"{models_dir}/linear_regression_model.pkl"),
#             'random_forest': joblib.load(f"{models_dir}/random_forest_model.pkl"),
#             'xgboost': joblib.load(f"{models_dir}/xgboost_model.pkl")
#         }
#         # Need to recreate test data
#         preprocessor = DataPreprocessor(
#             test_size=config['training']['test_size'],
#             random_state=config['training']['random_state']
#         )
#         X, y = preprocessor.prepare_features(df)
#         test_data = preprocessor.train_test_split(X, y)
#
#     if args.all or args.evaluate:
#         metrics = evaluate_models_step(models, test_data, config)
#
#     print("\n" + "=" * 60)
#     print("PIPELINE COMPLETED SUCCESSFULLY")
#     print("=" * 60)
#     print(f"\n✓ Dataset: {config['dataset']['output_path']}")
#     print(f"✓ Models: {config['output']['models_dir']}/")
#     print(f"✓ Metrics: {config['output']['metrics_dir']}/")
#     print(f"✓ Visualizations: {config['output']['visualizations_dir']}/")
#     print("\n")
#
#
# if __name__ == '__main__':
#     main()


# !/usr/bin/env python3
"""
ESG Score Prediction - Main Entry Point
Enhanced with robust error handling, logging, and validation
"""
import argparse
import yaml
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from data.dataset_generator import ESGDatasetGenerator
from data.preprocessor import DataPreprocessor
from models.linear_regression import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from utils.metrics import ModelEvaluator
from utils.visualization import Visualizer
import pandas as pd
import numpy as np
import joblib
import json

# Constants - Remove hardcoding
MODEL_NAMES = ['linear_regression', 'random_forest', 'xgboost']
REQUIRED_COLUMNS = ['CO2_Emissions', 'Energy_Use', 'Diversity_Index', 'Governance_Rating', 'ESG_Score']


def setup_logging(log_dir: str = 'outputs/logs') -> logging.Logger:
    """
    Setup comprehensive logging to both file and console with UTF-8 encoding.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/esg_pipeline_{timestamp}.log"

    # Create logger
    logger = logging.getLogger('ESGPipeline')
    logger.setLevel(logging.INFO)

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Force UTF-8 encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


def load_config(config_path: str = 'config.yaml', logger: Optional[logging.Logger] = None) -> dict:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to config file
        logger: Logger instance

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not os.path.exists(config_path):
        error_msg = f"Config file not found: {config_path}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['dataset', 'training', 'models', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        if logger:
            logger.info(f"✓ Configuration loaded from {config_path}")

        return config

    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML config: {e}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)


def validate_config(config: dict, logger: logging.Logger) -> bool:
    """
    Validate configuration values.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        True if valid, raises ValueError otherwise
    """
    try:
        # Validate dataset config
        if config['dataset']['num_companies'] < 10:
            raise ValueError("num_companies must be >= 10")

        # Validate training config
        test_size = config['training']['test_size']
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        # Validate model configs
        for model_name in MODEL_NAMES:
            if model_name not in config['models']:
                raise ValueError(f"Missing config for model: {model_name}")

        logger.info("✓ Configuration validated")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def create_directories(config: dict, logger: logging.Logger):
    """
    Create necessary output directories.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    dirs = [
        'outputs',
        'outputs/logs',
        config['output']['models_dir'],
        config['output']['metrics_dir'],
        config['output']['visualizations_dir']
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {d}")


def validate_dataset(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Validate dataset quality.

    Args:
        df: Dataset to validate
        logger: Logger instance

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
        raise ValueError("Dataset contains null values")

    # Check value ranges
    if (df['Diversity_Index'] < 0).any() or (df['Diversity_Index'] > 100).any():
        raise ValueError("Diversity_Index must be between 0 and 100")

    if (df['Governance_Rating'] < 1).any() or (df['Governance_Rating'] > 10).any():
        raise ValueError("Governance_Rating must be between 1 and 10")

    logger.info(f"✓ Dataset validated: {len(df)} samples, {len(df.columns)} features")
    return True


def generate_dataset_step(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate and save ESG dataset with validation.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Generated dataset
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Generating Dataset")
    logger.info("=" * 60)

    try:
        generator = ESGDatasetGenerator(random_seed=config['dataset']['random_seed'])
        df = generator.generate(num_companies=config['dataset']['num_companies'])

        # Validate dataset
        validate_dataset(df, logger)

        output_path = config['dataset']['output_path']
        generator.save_to_csv(df, output_path)

        logger.info(f"\n✓ Generated {len(df)} companies")
        logger.info(f"✓ Dataset saved to {output_path}")
        logger.info(f"\nDataset Statistics:")
        logger.info(f"\n{df.describe()}")

        return df

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


def train_models_step(df: pd.DataFrame, config: dict, logger: logging.Logger) -> Tuple[Dict, Tuple]:
    """
    Train all ML models with progress tracking and validation.

    Args:
        df: Training dataset
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Tuple of (models_dict, test_data_tuple)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Models")
    logger.info("=" * 60)

    try:
        # Prepare data
        preprocessor = DataPreprocessor(
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state']
        )

        X, y = preprocessor.prepare_features(df)
        X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

        logger.info(f"\nTraining set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Features: {X.shape[1]}")

        models = {}

        # Train Linear Regression
        logger.info("\n[1/3] Training Linear Regression...")
        try:
            lr_model = LinearRegressionModel()
            lr_model.fit(X_train, y_train)
            models['linear_regression'] = lr_model
            logger.info("✓ Linear Regression trained successfully")
        except Exception as e:
            logger.error(f"✗ Linear Regression training failed: {e}")
            raise

        # Train Random Forest
        logger.info("\n[2/3] Training Random Forest...")
        try:
            rf_config = config['models']['random_forest']
            rf_model = RandomForestModel(
                n_estimators=rf_config['n_estimators'],
                max_depth=rf_config['max_depth'],
                min_samples_split=rf_config['min_samples_split'],
                random_state=rf_config['random_state']
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = rf_model
            logger.info("✓ Random Forest trained successfully")
        except Exception as e:
            logger.error(f"✗ Random Forest training failed: {e}")
            raise

        # Train XGBoost
        logger.info("\n[3/3] Training XGBoost...")
        try:
            xgb_config = config['models']['xgboost']
            xgb_model = XGBoostModel(
                n_estimators=xgb_config['n_estimators'],
                max_depth=xgb_config['max_depth'],
                learning_rate=xgb_config['learning_rate'],
                random_state=xgb_config['random_state']
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
            logger.info("✓ XGBoost trained successfully")
        except Exception as e:
            logger.error(f"✗ XGBoost training failed: {e}")
            raise

        # Save models
        models_dir = config['output']['models_dir']
        for name, model in models.items():
            try:
                model_path = f"{models_dir}/{name}_model.pkl"
                joblib.dump(model, model_path)
                file_size = os.path.getsize(model_path) / 1024  # KB
                logger.info(f"✓ Saved {name} to {model_path} ({file_size:.2f} KB)")
            except Exception as e:
                logger.error(f"✗ Failed to save {name}: {e}")

        return models, (X_train, X_test, y_train, y_test)

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def evaluate_models_step(models: Dict, test_data: Tuple, config: dict, logger: logging.Logger) -> Dict:
    """
    Evaluate all models and generate comprehensive reports.

    Args:
        models: Dictionary of trained models
        test_data: Tuple of (X_train, X_test, y_train, y_test)
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Evaluating Models")
    logger.info("=" * 60)

    try:
        X_train, X_test, y_train, y_test = test_data
        evaluator = ModelEvaluator()
        all_metrics = {}

        logger.info("\nModel Performance on Test Set:\n")
        logger.info(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
        logger.info("-" * 50)

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = evaluator.evaluate_all(y_test, y_pred)
                all_metrics[name] = metrics

                logger.info(
                    f"{name:<20} {metrics['r2_score']:<10.4f} "
                    f"{metrics['mae']:<10.2f} {metrics['rmse']:<10.2f}"
                )
            except Exception as e:
                logger.error(f"✗ Evaluation failed for {name}: {e}")

        # Save metrics
        metrics_path = f"{config['output']['metrics_dir']}/evaluation_results.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"\n✓ Metrics saved to {metrics_path}")

        # Generate visualizations
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Generating Visualizations")
        logger.info("=" * 60)

        viz = Visualizer()
        viz_dir = config['output']['visualizations_dir']

        # Model comparison
        viz.plot_model_comparison(
            all_metrics,
            save_path=f"{viz_dir}/model_comparison.png"
        )
        logger.info(f"✓ Model comparison chart saved")

        # Predictions vs Actual for each model
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                viz.plot_predictions_vs_actual(
                    y_test, y_pred, name,
                    save_path=f"{viz_dir}/{name}_predictions.png"
                )
                logger.info(f"✓ {name} prediction plot saved")
            except Exception as e:
                logger.error(f"✗ Visualization failed for {name}: {e}")

        logger.info("\n✓ All visualizations generated")

        return all_metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    """Main entry point with comprehensive error handling."""
    start_time = datetime.now()

    parser = argparse.ArgumentParser(
        description='ESG Score Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                          # Run complete pipeline
  python main.py --generate-dataset             # Only generate dataset
  python main.py --train --evaluate             # Train and evaluate
  python main.py --all --num-companies 500      # Generate 500 companies
        """
    )
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--generate-dataset', action='store_true', help='Generate new dataset')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--num-companies', type=int, help='Override number of companies')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Load and validate configuration
        config = load_config(args.config, logger)
        validate_config(config, logger)

        # Override config if specified
        if args.num_companies:
            config['dataset']['num_companies'] = args.num_companies
            logger.info(f"Overriding num_companies: {args.num_companies}")

        # Create output directories
        create_directories(config, logger)

        logger.info("\n" + "=" * 60)
        logger.info("ESG SCORE PREDICTION SYSTEM")
        logger.info("=" * 60)

        # Execute steps
        if args.all or args.generate_dataset:
            df = generate_dataset_step(config, logger)
        else:
            # Load existing dataset
            dataset_path = config['dataset']['output_path']
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Dataset not found: {dataset_path}. "
                    f"Run with --generate-dataset first."
                )
            logger.info(f"Loading existing dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            validate_dataset(df, logger)

        if args.all or args.train:
            models, test_data = train_models_step(df, config, logger)
        else:
            # Load existing models
            models_dir = config['output']['models_dir']
            models = {}

            for model_name in MODEL_NAMES:
                model_path = f"{models_dir}/{model_name}_model.pkl"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Model not found: {model_path}. "
                        f"Run with --train first."
                    )
                models[model_name] = joblib.load(model_path)
                logger.info(f"✓ Loaded {model_name} from {model_path}")

            # Recreate test data
            preprocessor = DataPreprocessor(
                test_size=config['training']['test_size'],
                random_state=config['training']['random_state']
            )
            X, y = preprocessor.prepare_features(df)
            test_data = preprocessor.train_test_split(X, y)

        if args.all or args.evaluate:
            metrics = evaluate_models_step(models, test_data, config, logger)

        # Success summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"\n✓ Dataset: {config['dataset']['output_path']}")
        logger.info(f"✓ Models: {config['output']['models_dir']}/")
        logger.info(f"✓ Metrics: {config['output']['metrics_dir']}/")
        logger.info(f"✓ Visualizations: {config['output']['visualizations_dir']}/")
        logger.info(f"✓ Execution time: {duration:.2f} seconds")
        logger.info("\n")

        return 0  # Success exit code

    except KeyboardInterrupt:
        logger.warning("\n\n⚠ Pipeline interrupted by user")
        return 130  # Standard exit code for Ctrl+C

    except Exception as e:
        logger.error(f"\n\n✗ PIPELINE FAILED: {e}")
        logger.exception("Full error traceback:")
        return 1  # Failure exit code


if __name__ == '__main__':
    sys.exit(main())
