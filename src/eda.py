"""
Exploratory Data Analysis (EDA) functions.
"""

import pandas as pd


def dataset_shape(df):
    """
    Print the number of rows and columns.
    """

    print("=" * 50)
    print("Dataset Shape")
    print("=" * 50)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")


def dataset_info(df):
    """
    Display dataset information.
    """

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    df.info()


def missing_values(df):
    """
    Display missing values.
    """

    print("=" * 50)
    print("Missing Values")
    print("=" * 50)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False))


def duplicate_rows(df):
    """
    Display duplicate rows.
    """

    duplicates = df.duplicated().sum()

    print("=" * 50)
    print("Duplicate Rows")
    print("=" * 50)

    print(f"Duplicate Rows : {duplicates}")


def numerical_summary(df):
    """
    Summary statistics of numerical columns.
    """

    print("=" * 50)
    print("Numerical Summary")
    print("=" * 50)

    print(df.describe())


def categorical_summary(df):
    """
    Summary of categorical columns.
    """

    print("=" * 50)
    print("Categorical Columns")
    print("=" * 50)

    categorical = df.select_dtypes(include="object")

    print(categorical.describe())
