"""
Hyperparameter tuning module.
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_random_forest(
    X_train,
    y_train
):
    """
    Tune Random Forest using RandomizedSearchCV.
    """

    param_grid = {

        "n_estimators": [100, 200, 300, 500],

        "max_depth": [5, 10, 20, None],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4],

        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestClassifier(
        random_state=42
    )

    search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=param_grid,

        n_iter=20,

        scoring="f1",

        cv=5,

        random_state=42,

        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    print("\nBest Parameters")

    print(search.best_params_)

    print("\nBest CV F1")

    print(search.best_score_)

    return search.best_estimator_
