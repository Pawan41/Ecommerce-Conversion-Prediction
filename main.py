"""
This file is used to test each module as we build the project.
"""

from src.data_loader import (
    load_train_data,
    load_private_test_data,
    load_public_test_data
)

from src.eda import (
    dataset_shape,
    dataset_info,
    missing_values,
    duplicate_rows,
    numerical_summary,
    categorical_summary
)

from src.visualization import (
    plot_missing_values,
    plot_target_distribution,
    plot_correlation_heatmap
)

from src.preprocessing import preprocess_data
from src.train import split_train_test

from src.model import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost
)

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix
)

from src.config import (
    PUBLIC_PREDICTION_PATH,
    PRIVATE_PREDICTION_PATH
)

from src.feature_engineering import create_features
from src.tuning import tune_random_forest
from src.feature_importance import plot_feature_importance
from src.roc_curve import plot_roc_curve

from src.precision_recall_curve import (
    plot_precision_recall_curve
)

from src.save_model import save_model

from src.predict import (
    predict_dataset,
    save_predictions
)

from src.model_comparison import compare_models
from src.explainability import plot_shap_summary
from src.validation import cross_validate_model


def main():
    """
    Main function for testing the project pipeline.
    """

    print("=" * 60)
    print(" E-Commerce Conversion Prediction ")
    print("=" * 60)

    # Load datasets
    train_df = load_train_data()
    public_df = load_public_test_data()
    private_df = load_private_test_data()

    print("\n✅ Datasets loaded successfully!\n")

    # -------------------------------------------
    # Feature Engineering
    # -------------------------------------------

    print("\nStarting Feature Engineering...\n")

    train_df = create_features(train_df)

    print("✅ Feature Engineering Completed!")

    print(f"New Dataset Shape : {train_df.shape}")

    # ------------------------------------------------
    # Exploratory Data Analysis
    # ------------------------------------------------

    print("\nStarting Exploratory Data Analysis...\n")

    dataset_shape(train_df)

    dataset_info(train_df)

    missing_values(train_df)

    plot_missing_values(train_df)

    plot_target_distribution(train_df)

    plot_correlation_heatmap(train_df)

    duplicate_rows(train_df)

    numerical_summary(train_df)

    categorical_summary(train_df)

    # ------------------------------------------
    # Data Preprocessing
    # -------------------------------------------

    print("\nStarting Data Preprocessing...\n")

    X, y, preprocessor = preprocess_data(train_df)

    print("=" * 50)
    print("Feature Matrix")
    print("=" * 50)

    print(f"X Shape : {X.shape}")

    print(f"y Shape : {y.shape}")

    print("\nTarget Distribution")

    print(y.value_counts())

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    print("\nProcessed Feature Matrix")

    print(f"Processed Shape : {X_processed.shape}")

    print("\nSplitting Dataset...\n")

    X_train, X_test, y_train, y_test = split_train_test(
        X_processed,
        y
    )

    print("=" * 50)
    print("Train/Test Split")
    print("=" * 50)

    print(f"Training Features : {X_train.shape}")
    print(f"Testing Features  : {X_test.shape}")

    print(f"Training Labels   : {y_train.shape}")
    print(f"Testing Labels    : {y_test.shape}")

    # -------------------------------------------
    # Model Training
    # -------------------------------------------

    print("\nStarting Model Training...\n")

    model = train_logistic_regression(
        X_train,
        y_train
    )

    print("✅ Logistic Regression trained successfully!")

    # -------------------------------------------
    # Model Evaluation
    # ------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    plot_confusion_matrix(
        model,
        X_test,
        y_test
    )

    plot_roc_curve(
        model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    plot_precision_recall_curve(
        model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    print("\n")
    print("=" * 60)
    print("Training Random Forest")
    print("=" * 60)

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    print("✅ Random Forest trained successfully!")

    evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    plot_confusion_matrix(
        rf_model,
        X_test,
        y_test
    )

    plot_roc_curve(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    plot_precision_recall_curve(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    print("\n")
    print("=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)

    best_rf = tune_random_forest(
        X_train,
        y_train
    )

    print("✅ Tuned Random Forest trained!")

    evaluate_model(
        best_rf,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    plot_confusion_matrix(
        best_rf,
        X_test,
        y_test
    )

    plot_roc_curve(
        best_rf,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    plot_precision_recall_curve(
        best_rf,
        X_test,
        y_test,
        "Tuned Random Forest"
    )

    # -------------------------------------------
    # XGBoost Training
    # -------------------------------------------

    print("\n")
    print("=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    print("✅ XGBoost trained successfully!")

    # -------------------------------------------
    # XGBoost Evaluation
    # -------------------------------------------

    evaluate_model(
        xgb_model,
        X_test,
        y_test,
        "XGBoost"
    )

    plot_confusion_matrix(
        xgb_model,
        X_test,
        y_test
    )

    plot_feature_importance(
        xgb_model,
        feature_names
    )

    plot_roc_curve(
        xgb_model,
        X_test,
        y_test,
        "XGBoost"
    )

    plot_precision_recall_curve(
        xgb_model,
        X_test,
        y_test,
        "XGBoost"
    )

    plot_shap_summary(
        xgb_model,
        X_train,
        X_test
    )

    save_model(
        xgb_model,
        "xgboost_model.pkl"
    )

    save_model(
        preprocessor,
        "preprocessor.pkl"
    )

    print("\n")
    print("=" * 60)
    print("Cross Validation")
    print("=" * 60)

    cross_validate_model(
        xgb_model,
        X_processed,
        y,
        cv=5,
        scoring="f1"
    )

    # -------------------------------------------
    # Generate Competition Predictions
    # -------------------------------------------

    print("\n")
    print("=" * 60)
    print("Generating Competition Predictions")
    print("=" * 60)

    # Public Test
    public_predictions = predict_dataset(
        xgb_model,
        preprocessor,
        public_df
    )

    save_predictions(
        public_predictions,
        PUBLIC_PREDICTION_PATH
    )

    # Private Test
    private_predictions = predict_dataset(
        xgb_model,
        preprocessor,
        private_df
    )

    save_predictions(
        private_predictions,
        PRIVATE_PREDICTION_PATH
    )

    print("\n✅ Competition prediction files created successfully!")

    # -------------------------------------------
    # Model Comparison
    # -------------------------------------------

    compare_models()


if __name__ == "__main__":
    main()
