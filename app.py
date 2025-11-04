#!/usr/bin/env python3
"""
ESG Score Prediction - Flask Web Application
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pathlib import Path
import json
import os

app = Flask(__name__)
CORS(app)

# Load models on startup
MODELS = {}

from flask import send_from_directory

# Add this route to serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def load_models():
    """Load all trained models."""
    model_names = ['linear_regression', 'random_forest', 'xgboost']
    for name in model_names:
        model_path = f"outputs/models/{name}_model.pkl"
        if Path(model_path).exists():
            MODELS[name] = joblib.load(model_path)
            print(f"✓ Loaded {name} model")
        else:
            print(f"⚠ Warning: {name} model not found at {model_path}")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Return list of available models."""
    return jsonify({
        'models': list(MODELS.keys()),
        'default': 'random_forest'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make ESG score prediction."""
    try:
        data = request.get_json()

        # Validate input
        required_fields = ['model', 'co2', 'energy', 'diversity', 'governance']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        model_name = data['model']
        if model_name not in MODELS:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # Prepare features
        co2 = float(data['co2'])
        energy = float(data['energy'])
        diversity = float(data['diversity'])
        governance = int(data['governance'])

        # Validate ranges
        if co2 < 0 or energy < 0:
            return jsonify({'error': 'CO2 and Energy must be positive'}), 400
        if not (0 <= diversity <= 100):
            return jsonify({'error': 'Diversity Index must be between 0-100'}), 400
        if not (1 <= governance <= 10):
            return jsonify({'error': 'Governance Rating must be between 1-10'}), 400

        # Make prediction
        X = np.array([[co2, energy, diversity, governance]])
        model = MODELS[model_name]
        esg_score = float(model.predict(X)[0])

        # Determine rating
        if esg_score >= 80:
            rating = "Excellent"
            emoji = "🟢"
            stars = 5
            description = "Outstanding sustainability performance"
        elif esg_score >= 60:
            rating = "Good"
            emoji = "🟢"
            stars = 4
            description = "Strong sustainability practices"
        elif esg_score >= 40:
            rating = "Fair"
            emoji = "🟡"
            stars = 3
            description = "Room for improvement"
        elif esg_score >= 20:
            rating = "Needs Improvement"
            emoji = "🟠"
            stars = 2
            description = "Significant action required"
        else:
            rating = "Poor"
            emoji = "🔴"
            stars = 1
            description = "Critical sustainability issues"

        # Calculate component scores (simplified)
        env_score = max(0, 100 - (co2 / 5000) - (energy / 2000))
        social_score = diversity * 0.8
        gov_score = governance * 8

        return jsonify({
            'success': True,
            'prediction': {
                'score': round(esg_score, 2),
                'rating': rating,
                'emoji': emoji,
                'stars': stars,
                'description': description
            },
            'components': {
                'environmental': round(min(100, max(0, env_score)), 2),
                'social': round(social_score, 2),
                'governance': round(gov_score, 2)
            },
            'inputs': {
                'co2': co2,
                'energy': energy,
                'diversity': diversity,
                'governance': governance,
                'model': model_name
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history from JSON file."""
    history_file = 'outputs/prediction_history.json'
    if Path(history_file).exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify([])


@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Return example scenarios."""
    examples = [
        {
            'name': 'Green Tech Startup',
            'co2': 15000,
            'energy': 8000,
            'diversity': 90,
            'governance': 10,
            'description': 'Low emissions, high diversity'
        },
        {
            'name': 'Average Manufacturing',
            'co2': 50000,
            'energy': 30000,
            'diversity': 65,
            'governance': 7,
            'description': 'Typical industrial company'
        },
        {
            'name': 'Heavy Industrial',
            'co2': 200000,
            'energy': 100000,
            'diversity': 30,
            'governance': 3,
            'description': 'High emissions, needs improvement'
        }
    ]
    return jsonify(examples)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ESG SCORE PREDICTION WEB APP")
    print("=" * 60)
    print("\nLoading models...")
    load_models()

    if not MODELS:
        print("\n⚠ ERROR: No models found!")
        print("Please run: python main.py --all")
        print("=" * 60 + "\n")
    else:
        print(f"\n✓ Loaded {len(MODELS)} models")
        print("\nStarting server...")
        print("Open your browser and visit: http://127.0.0.1:5000")
        print("=" * 60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)