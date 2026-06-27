"""
Compare all trained machine learning models.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    MODEL_METRICS_FILE,
    FIGURE_DIR
)


def compare_models():
    """
    Create comparison table and bar chart.
    """

    metrics = pd.read_csv(MODEL_METRICS_FILE)

    print("\n")
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(metrics)

    metrics.to_csv(
        MODEL_METRICS_FILE,
        index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics.plot(
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        kind="bar",
        ax=ax
    )

    ax.set_title("Model Performance Comparison")

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1)

    plt.xticks(rotation=15)

    plt.tight_layout()

    output_path = FIGURE_DIR / "model_comparison.png"

    plt.savefig(output_path)

    plt.close()

    print(f"\nModel comparison plot saved to:\n{output_path}")
