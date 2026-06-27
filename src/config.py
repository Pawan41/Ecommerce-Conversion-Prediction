"""
Configuration file for the E-Commerce Conversion Prediction project.
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset Paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
PUBLIC_TEST_PATH = RAW_DATA_DIR / "public_test.csv"
PRIVATE_TEST_PATH = RAW_DATA_DIR / "private_test.csv"

# Model Directory
MODEL_DIR = PROJECT_ROOT / "models"

# Report Directories
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RESULT_DIR = REPORT_DIR / "results"

# Random Seed
RANDOM_STATE = 42

# -----------------------------
# Figure File Paths
# -----------------------------

MISSING_VALUES_PLOT = FIGURE_DIR / "missing_values.png"

TARGET_DISTRIBUTION_PLOT = FIGURE_DIR / "target_distribution.png"

CORRELATION_HEATMAP = FIGURE_DIR / "correlation_heatmap.png"

NUMERICAL_FEATURES_PLOT = FIGURE_DIR / "numerical_features.png"

CATEGORICAL_FEATURES_PLOT = FIGURE_DIR / "categorical_features.png"


# -----------------------------
# Evaluation Outputs
# -----------------------------

CONFUSION_MATRIX_PLOT = FIGURE_DIR / "confusion_matrix.png"

ROC_CURVE_PLOT = FIGURE_DIR / "roc_curve.png"

PRECISION_RECALL_CURVE_PLOT = FIGURE_DIR / "precision_recall_curve.png"

MODEL_METRICS_FILE = RESULT_DIR / "model_metrics.csv"


# -----------------------------
# Prediction Outputs
# -----------------------------

PUBLIC_PREDICTION_PATH = RESULT_DIR / "public_predictions.csv"

PRIVATE_PREDICTION_PATH = RESULT_DIR / "private_predictions.csv"
