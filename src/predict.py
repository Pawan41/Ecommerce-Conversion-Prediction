"""
Prediction pipeline using the saved model.
"""

from src.config import (
    PUBLIC_PREDICTION_PATH,
    PRIVATE_PREDICTION_PATH
)

from src.feature_engineering import create_features

import joblib
import pandas as pd

from src.config import MODEL_DIR


def load_saved_model():
    """
    Load trained XGBoost model.
    """

    model = joblib.load(
        MODEL_DIR / "xgboost_model.pkl"
    )

    return model


def load_preprocessor():
    """
    Load saved preprocessing pipeline.
    """

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    return preprocessor


def predict_dataset(
    model,
    preprocessor,
    df
):
    # Save User IDs
    user_ids = df["User_ID"].copy()

    # Apply feature engineering
    df = create_features(df)

    # Remove User_ID
    df = df.drop(columns=["User_ID"])

    # Transform features
    X = preprocessor.transform(df)

    # Predict
    predictions = model.predict(X)

    submission = pd.DataFrame({
        "User_ID": user_ids,
        "Converted": predictions
    })

    return submission


def save_predictions(
    predictions,
    output_path
):
    """
    Save predictions to CSV.
    """

    predictions.to_csv(
        output_path,
        index=False
    )

    print(f"\nPredictions saved to:\n{output_path}")
