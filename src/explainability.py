"""
Model explainability using SHAP.
"""

import shap
import matplotlib.pyplot as plt

from pathlib import Path

from src.config import FIGURE_DIR


def plot_shap_summary(
    model,
    X_train,
    X_test
):
    """
    Generate SHAP summary plot.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    output_path = FIGURE_DIR / "shap_summary.png"

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"SHAP Summary saved to:\n{output_path}")
