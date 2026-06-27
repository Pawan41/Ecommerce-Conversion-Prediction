"""
Feature engineering module.
"""

import numpy as np


def create_features(df):
    """
    Create additional features for the model.
    """

    df = df.copy()

    # Avoid division by zero
    df["Views_Per_Product"] = (
        df["Pages_Viewed"] /
        (df["Products_Viewed"] + 1)
    )

    df["Purchase_Engagement"] = (
        df["Previous_Purchases"] *
        df["Products_Viewed"]
    )

    df["Time_Per_Page"] = (
        df["Time_On_Site"] /
        (df["Pages_Viewed"] + 1)
    )

    return df
