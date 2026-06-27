"""
Save trained models.
"""

import joblib

from src.config import MODEL_DIR


def save_model(
    model,
    filename
):
    """
    Save a trained model.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = MODEL_DIR / filename

    joblib.dump(
        model,
        output_path
    )

    print(
        f"Model saved to:\n{output_path}"
    )
