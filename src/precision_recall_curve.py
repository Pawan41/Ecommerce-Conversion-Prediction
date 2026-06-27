"""
Precision-Recall Curve visualization.
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

from src.config import FIGURE_DIR


def plot_precision_recall_curve(
    model,
    X_test,
    y_test,
    model_name="Model"
):
    """
    Plot Precision-Recall Curve.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    ap_score = average_precision_score(
        y_test,
        probabilities
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap_score:.3f}"
    )

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")

    ax.set_title(
        f"Precision-Recall Curve - {model_name}"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURE_DIR /
        f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
    )

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Precision-Recall Curve saved to:\n{output_path}"
    )
