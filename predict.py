#!/usr/bin/env python3
"""
Make Predictions with Trained ESG Score Models
"""
import argparse
import joblib
import numpy as np
from pathlib import Path



def load_model(model_name: str):
    """Load trained model from disk."""
    model_path = f"outputs/models/{model_name}_model.pkl"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first using: python main.py --all")

    return joblib.load(model_path)


def predict_esg_score(model, co2_emissions, energy_use, diversity_index, governance_rating):
    """Make ESG score prediction."""
    # Prepare input features
    X = np.array([[co2_emissions, energy_use, diversity_index, governance_rating]])

    # Make prediction
    prediction = model.predict(X)[0]

    return prediction


def main():
    parser = argparse.ArgumentParser(description='Predict ESG Score')
    parser.add_argument('--model', choices=['linear_regression', 'random_forest', 'xgboost'],
                        default='xgboost', help='Model to use for prediction')
    parser.add_argument('--co2', type=float, required=True,
                        help='CO2 Emissions (e.g., 50000)')
    parser.add_argument('--energy', type=float, required=True,
                        help='Energy Use (e.g., 30000)')
    parser.add_argument('--diversity', type=float, required=True,
                        help='Diversity Index (0-100, e.g., 65)')
    parser.add_argument('--governance', type=int, required=True,
                        help='Governance Rating (1-10, e.g., 7)')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ESG SCORE PREDICTION")
    print("=" * 60)

    # Load model
    print(f"\nLoading {args.model} model...")
    try:
        model = load_model(args.model)
        print("✓ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Make prediction
    print("Input Features:")
    print(f"  CO2 Emissions:     {args.co2:>12,.0f}")
    print(f"  Energy Use:        {args.energy:>12,.0f}")
    print(f"  Diversity Index:   {args.diversity:>12.1f}")
    print(f"  Governance Rating: {args.governance:>12}")
    print()

    esg_score = predict_esg_score(
        model,
        args.co2,
        args.energy,
        args.diversity,
        args.governance
    )

    print("=" * 60)
    print(f"Predicted ESG Score: {esg_score:.2f}")
    print("=" * 60)

    # Provide interpretation
    if esg_score >= 80:
        rating = "Excellent ⭐⭐⭐⭐⭐"
        color = "🟢"
    elif esg_score >= 60:
        rating = "Good ⭐⭐⭐⭐"
        color = "🟢"
    elif esg_score >= 40:
        rating = "Fair ⭐⭐⭐"
        color = "🟡"
    elif esg_score >= 20:
        rating = "Needs Improvement ⭐⭐"
        color = "🟠"
    else:
        rating = "Poor ⭐"
        color = "🔴"

    print(f"\nESG Rating: {color} {rating}")
    print()

    # Provide context
    print("Context:")
    print("  • ESG Score Range: 0-100")
    print("  • 80-100: Excellent sustainability performance")
    print("  • 60-79:  Good sustainability practices")
    print("  • 40-59:  Fair, room for improvement")
    print("  • 20-39:  Needs significant improvement")
    print("  • 0-19:   Poor sustainability performance")
    print()


if __name__ == '__main__':
    main()
