"""
Data preprocessing pipeline for the E-Commerce Conversion Prediction project.

This module provides reusable functions to:
1. Split features and target.
2. Identify numerical and categorical features.
3. Create preprocessing pipelines.
4. Build a ColumnTransformer.
5. Return the complete preprocessing workflow.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


def split_features_target(df: pd.DataFrame):
    """
    Split the dataset into features (X) and target (y).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.
    """

    X = df.drop(columns=["User_ID", "Converted"])
    y = df["Converted"]

    return X, y


def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical feature columns.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    numerical_features : list
        List of numerical feature names.

    categorical_features : list
        List of categorical feature names.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    return numerical_features, categorical_features


def create_numerical_pipeline():
    """
    Create preprocessing pipeline for numerical features.

    Steps
    -----
    1. Fill missing values using median.
    2. Standardize numerical features.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    return numerical_pipeline


def create_categorical_pipeline():
    """
    Create preprocessing pipeline for categorical features.

    Steps
    -----
    1. Fill missing values using most frequent value.
    2. Apply One-Hot Encoding.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    return categorical_pipeline


def build_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Combine numerical and categorical preprocessing pipelines.

    Parameters
    ----------
    numerical_features : list
        Numerical column names.

    categorical_features : list
        Categorical column names.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                create_numerical_pipeline(),
                numerical_features
            ),
            (
                "categorical",
                create_categorical_pipeline(),
                categorical_features
            )
        ]
    )

    return preprocessor


def preprocess_data(df: pd.DataFrame):
    """
    Complete preprocessing workflow.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    preprocessor : ColumnTransformer
        Complete preprocessing pipeline.
    """

    X, y = split_features_target(df)

    numerical_features, categorical_features = get_feature_types(X)

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    return X, y, preprocessor
