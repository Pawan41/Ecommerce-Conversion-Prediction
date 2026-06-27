"""
Functions for loading datasets.
"""

import pandas as pd

from src.config import (
    TRAIN_DATA_PATH,
    PUBLIC_TEST_PATH,
    PRIVATE_TEST_PATH
)


def load_train_data():
    """
    Load the training dataset.
    """

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    return train_df


def load_public_test_data():
    """
    Load the public test dataset.
    """

    public_test_df = pd.read_csv(PUBLIC_TEST_PATH)

    return public_test_df


def load_private_test_data():
    """
    Load the private test dataset.
    """

    private_test_df = pd.read_csv(PRIVATE_TEST_PATH)

    return private_test_df
