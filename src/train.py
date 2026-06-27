"""
Model training utilities.
"""

from sklearn.model_selection import train_test_split


def split_train_test(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )
