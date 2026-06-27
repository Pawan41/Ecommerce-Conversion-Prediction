"""
ROC Curve visualization.
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

from src.config import FIGURE_DIR


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name="Model"
):
    """
    Plot ROC Curve and save it.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}",
        linewidth=2
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.set_title(f"ROC Curve - {model_name}")

    ax.legend()

    fig.tight_layout()

    output_path = FIGURE_DIR / \
        f"{model_name.lower().replace(' ', '_')}_roc_curve.png"

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)

    print(
        f"ROC Curve saved to:\n{output_path}"
    )
