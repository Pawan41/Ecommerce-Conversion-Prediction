"""
Model evaluation utilities.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from src.config import (
    CONFUSION_MATRIX_PLOT,
    ROC_CURVE_PLOT,
    PRECISION_RECALL_CURVE_PLOT,
    MODEL_METRICS_FILE
)

sns.set_theme(style="whitegrid")


def save_metrics(
    model_name,
    accuracy,
    precision,
    recall,
    f1
):
    """
    Save model metrics.
    """

    new_metric = pd.DataFrame(
        {
            "Model": [model_name],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1]
        }
    )

    # Load existing metrics if available
    if Path(MODEL_METRICS_FILE).exists():

        metrics = pd.read_csv(
            MODEL_METRICS_FILE
        )

        metrics = pd.concat(
            [metrics, new_metric],
            ignore_index=True
        )

        # Keep only the latest result for each model
        metrics = metrics.drop_duplicates(
            subset="Model",
            keep="last"
        )

    else:

        metrics = new_metric

    metrics.to_csv(
        MODEL_METRICS_FILE,
        index=False
    )

    print(f"\nMetrics saved to:\n{MODEL_METRICS_FILE}")


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model.
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    # Save metrics to CSV
    save_metrics(
        model_name,
        accuracy,
        precision,
        recall,
        f1
    )


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot and save confusion matrix.
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_PLOT,
        dpi=300
    )

    plt.close(fig)

    print(
        f"Confusion Matrix saved to:\n{CONFUSION_MATRIX_PLOT}"
    )
