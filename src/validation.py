"""
Cross-validation utilities.
"""

import numpy as np

from sklearn.model_selection import cross_val_score


def cross_validate_model(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
):
    """
    Perform k-fold cross validation.

    Parameters
    ----------
    model : estimator
        Machine learning model.

    X : array-like
        Feature matrix.

    y : array-like
        Target labels.

    cv : int
        Number of folds.

    scoring : str
        Evaluation metric.

    Returns
    -------
    scores : ndarray
        Cross-validation scores.
    """

    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("\n" + "=" * 60)
    print(f"{cv}-Fold Cross Validation")
    print("=" * 60)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print("\nMean Score :", f"{np.mean(scores):.4f}")
    print("Std Score  :", f"{np.std(scores):.4f}")

    return scores
