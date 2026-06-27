"""
Visualization functions for Exploratory Data Analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MISSING_VALUES_PLOT,
    TARGET_DISTRIBUTION_PLOT,
    CORRELATION_HEATMAP
)

# Better looking plots
sns.set_theme(style="whitegrid")


def plot_missing_values(df):
    """
    Create and save a bar chart showing missing values.
    """

    # Count missing values
    missing = df.isnull().sum()

    # Keep only columns with missing values
    missing = missing[missing > 0]

    # If no missing values exist
    if missing.empty:
        print("No missing values found.")
        return

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw bar chart
    missing.sort_values().plot(
        kind="bar",
        ax=ax,
        color="steelblue"
    )

    # Add title and labels
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Features")
    ax.set_ylabel("Missing Count")

    # Adjust layout
    fig.tight_layout()

    # Save figure
    fig.savefig(MISSING_VALUES_PLOT)

    # Close figure
    plt.close(fig)

    print(f"Missing value plot saved to:\n{MISSING_VALUES_PLOT}")


def plot_target_distribution(df):
    """
    Create and save the target class distribution plot.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Converted",
        hue="Converted",
        palette="Set2",
        legend=False,
        ax=ax
    )

    total = len(df)

    # Add count and percentage labels
    for p in ax.patches:
        count = int(p.get_height())
        percentage = (count / total) * 100

        ax.annotate(
            f"{count}\n({percentage:.1f}%)",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title("Target Variable Distribution")
    ax.set_xlabel("Converted")
    ax.set_ylabel("Number of Users")

    fig.tight_layout()

    fig.savefig(TARGET_DISTRIBUTION_PLOT, dpi=300)

    plt.close(fig)

    print(f"Target distribution plot saved to:\n{TARGET_DISTRIBUTION_PLOT}")


def plot_correlation_heatmap(df):
    """
    Create and save a correlation heatmap.
    """

    # Select only numerical columns
    numerical_df = df.select_dtypes(include=["number"])

    # Compute correlation matrix
    correlation = numerical_df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    # Title
    ax.set_title("Correlation Heatmap")

    # Adjust layout
    fig.tight_layout()

    # Save figure
    fig.savefig(CORRELATION_HEATMAP, dpi=300)

    # Close figure
    plt.close(fig)

    print(f"Correlation heatmap saved to:\n{CORRELATION_HEATMAP}")
