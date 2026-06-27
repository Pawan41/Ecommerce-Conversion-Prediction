"""
Feature importance visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURE_DIR


def plot_feature_importance(
    model,
    feature_names,
    top_n=15
):
    """
    Plot feature importance for tree-based models.
    """

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    importance.head(top_n).plot(
        x="Feature",
        y="Importance",
        kind="barh",
        ax=ax,
        legend=False,
        color="steelblue"
    )

    ax.invert_yaxis()

    ax.set_title("Top Feature Importance")

    ax.set_xlabel("Importance Score")

    ax.set_ylabel("Feature")

    fig.tight_layout()

    output_path = FIGURE_DIR / "feature_importance.png"

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Feature Importance saved to:\n{output_path}"
    )
