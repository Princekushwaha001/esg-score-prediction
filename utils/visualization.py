import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


class Visualizer:
    """Create visualizations for model evaluation."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_model_comparison(self, metrics: Dict[str, Dict[str, float]],
                              save_path: str = None):
        """Create bar plot comparing model performance."""
        models = list(metrics.keys())
        r2_scores = [metrics[m]['r2_score'] for m in models]
        mae_scores = [metrics[m]['mae'] for m in models]
        rmse_scores = [metrics[m]['rmse'] for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # R² Score
        axes[0].bar(models, r2_scores, color='steelblue')
        axes[0].set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim([0, 1])
        for i, v in enumerate(r2_scores):
            axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

        # MAE
        axes[1].bar(models, mae_scores, color='coral')
        axes[1].set_title('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE')
        for i, v in enumerate(mae_scores):
            axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

        # RMSE
        axes[2].bar(models, rmse_scores, color='mediumseagreen')
        axes[2].set_title('Root Mean Squared Error (Lower is Better)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('RMSE')
        for i, v in enumerate(rmse_scores):
            axes[2].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str, save_path: str = None):
        """Scatter plot of predictions vs actual values."""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='steelblue')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.xlabel('Actual ESG Score', fontweight='bold')
        plt.ylabel('Predicted ESG Score', fontweight='bold')
        plt.title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
