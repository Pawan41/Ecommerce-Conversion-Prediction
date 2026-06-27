"""
Machine Learning models for the E-Commerce Conversion Prediction project.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.

    y_train : array-like
        Training labels.

    Returns
    -------
    model : LogisticRegression
        Trained Logistic Regression model.
    """

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(
    X_train,
    y_train
):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : array-like
        Training features.

    y_train : array-like
        Training labels.

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_xgboost(
    X_train,
    y_train
):
    """
    Train an XGBoost classifier.
    """

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model
