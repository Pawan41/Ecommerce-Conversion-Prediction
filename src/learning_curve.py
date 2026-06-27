"""
Learning Curve visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from src.config import FIGURE_DIR


def plot_learning_curve(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
):
    """
    Plot the learning curve for a model.
    """

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training Score"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="s",
        label="Validation Score"
    )

    plt.title("Learning Curve")

    plt.xlabel("Training Samples")

    plt.ylabel("F1 Score")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_path = FIGURE_DIR / "learning_curve.png"

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Learning Curve saved to:\n{output_path}")
