# 🌱 ESG Score Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning system for predicting Environmental, Social, and Governance (ESG) scores for companies. This project implements and compares multiple regression models, providing a complete pipeline from data processing to model deployment.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Multiple ML Models**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - XGBoost Regressor
- **End-to-End Pipeline**:
  - Data generation and preprocessing
  - Model training and validation
  - Performance evaluation and visualization
- **REST API**: Easy integration with other services
- **Comprehensive Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
  - Feature Importance Analysis
- **Configurable**: YAML-based configuration for all parameters

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/esg-score-prediction.git
   cd esg-score-prediction
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
esg-score-prediction/
├── data/                   # Data processing and generation
│   ├── raw/               # Raw data files
│   └── processed/         # Processed datasets
├── models/                # ML model implementations
│   ├── __init__.py
│   ├── base_model.py      # Base model interface
│   ├── linear_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
├── outputs/               # Generated outputs
│   ├── models/            # Trained model files (.pkl)
│   ├── metrics/           # Evaluation metrics (JSON)
│   └── visualizations/    # Performance plots
├── utils/                 # Utility functions
│   ├── data_processor.py  # Data processing utilities
│   ├── logger.py          # Logging configuration
│   └── visualization.py   # Plotting functions
├── tests/                 # Unit and integration tests
├── templates/             # HTML templates for web interface
├── static/                # Static files (CSS, JS)
├── .gitignore             # Git ignore file
├── app.py                 # Flask web application
├── config.yaml            # Configuration file
├── main.py                # Main training pipeline
├── predict.py             # Command-line prediction script
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🧠 Models

### 1. Linear Regression
- **Type**: Linear model
- **Use Case**: Baseline model for comparison
- **Strengths**:
  - Fast training and inference
  - Easy to interpret
  - Good for establishing baseline performance

### 2. Random Forest Regressor
- **Type**: Ensemble of decision trees
- **Use Case**: General-purpose regression
- **Strengths**:
  - Handles non-linear relationships
  - Robust to outliers
  - Good default choice for tabular data

### 3. XGBoost Regressor
- **Type**: Gradient Boosting
- **Use Case**: High-performance prediction
- **Strengths**:
  - State-of-the-art performance
  - Handles missing values
  - Built-in regularization

## 🛠️ Usage

### Command Line Interface

#### 1. Data Generation
Generate synthetic ESG dataset:
```bash
python main.py --generate-dataset --samples 10000
```

#### 2. Model Training
Train all models:
```bash
python main.py --train --models all
```

Train specific model:
```bash
python main.py --train --models random_forest xgboost
```

#### 3. Model Evaluation
Evaluate all trained models:
```bash
python main.py --evaluate
```

#### 4. Complete Pipeline
Run the entire pipeline (data generation → training → evaluation):
```bash
python main.py --all
```

### Making Predictions

#### Command Line
```bash
python predict.py --model random_forest \
                 --co2 45000 \
                 --energy 32000 \
                 --diversity 72 \
                 --governance 8 \
                 --output predictions.json
```

#### Python API
```python
from models.random_forest import RandomForestModel

# Initialize and load model
model = RandomForestModel()
model.load("outputs/models/random_forest_model.pkl")

# Make prediction
features = {
    'co2_emissions': 45000,
    'energy_consumption': 32000,
    'diversity_index': 72,
    'governance_score': 8
}
prediction = model.predict(features)
print(f"Predicted ESG Score: {prediction:.2f}")
```

## ⚙️ Configuration

Edit `config.yaml` to customize model parameters, file paths, and other settings:

```yaml
data:
  input_file: "data/processed/esg_dataset.csv"
  test_size: 0.2
  random_state: 42

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  xgboost:
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 7

output:
  models_dir: "outputs/models"
  metrics_dir: "outputs/metrics"
  plots_dir: "outputs/visualizations"
```

## 📊 Examples

### Example 1: Training and Evaluation
```bash
# Generate 10,000 samples of synthetic data
python main.py --generate-dataset --samples 10000

# Train and evaluate all models
python main.py --all
```

### Example 2: Batch Prediction
```bash
# Create input CSV file
cat > input_batch.csv << EOL
co2_emissions,energy_consumption,diversity_index,governance_score
45000,32000,72,8
38000,28000,65,7
52000,35000,68,6
EOL

# Run batch prediction
python predict.py --batch input_batch.csv --output batch_predictions.json
```

### Example 3: Web Interface
```bash
# Start the Flask web server
python app.py
```
Then open `http://localhost:5000` in your browser to use the interactive web interface.

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Test coverage report:
```bash
pytest --cov=. --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please contact [your.email@example.com](mailto:your.email@example.com)

## 📊 Results

Model performance is automatically saved in `outputs/metrics/` and visualizations in `outputs/visualizations/`.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or suggestions, please open an issue or contact [Your Name] at [your.email@example.com].
