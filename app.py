import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.predict import (
    load_saved_model,
    load_preprocessor,
    predict_dataset
)

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="E-Commerce Conversion Prediction",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📊 Project Overview")

st.sidebar.success("🏆 Best Model: XGBoost")

st.sidebar.markdown("""
### Models Implemented

- Logistic Regression
- Random Forest
- Tuned Random Forest
- XGBoost
""")

try:
    metrics = pd.read_csv(
        "reports/results/model_metrics.csv"
    )

    st.sidebar.subheader("📈 Model Metrics")

    st.sidebar.dataframe(
        metrics,
        width="stretch"
    )

except:
    st.sidebar.warning("Metrics file not found.")

# -----------------------------
# Title
# -----------------------------

st.title("🛒 E-Commerce Conversion Prediction")

st.markdown(
    """
Predict whether an e-commerce customer will convert using the trained
**XGBoost Machine Learning Model**.
"""
)

st.divider()

# -----------------------------
# Project Statistics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset Size", "10,000")
col2.metric("Features", "15")
col3.metric("Models", "4")
col4.metric("Best F1 Score", "0.416")

st.divider()

# -----------------------------
# Upload Dataset
# -----------------------------

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        width="stretch"
    )

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("")

    if st.button("🚀 Predict Conversion"):

        model = load_saved_model()

        preprocessor = load_preprocessor()

        predictions = predict_dataset(
            model,
            preprocessor,
            df
        )

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Results")

        st.dataframe(
            predictions,
            width="stretch"
        )

        st.write("")

        st.subheader("Prediction Summary")

        converted = predictions["Converted"].sum()

        not_converted = len(predictions) - converted

        col1, col2 = st.columns(2)

        col1.metric(
            "Converted Customers",
            int(converted)
        )

        col2.metric(
            "Non-Converted Customers",
            int(not_converted)
        )

        # Pie Chart

        fig, ax = plt.subplots(figsize=(5, 5))

        predictions["Converted"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            labels=["Not Converted", "Converted"],
            ax=ax
        )

        ax.set_ylabel("")

        st.pyplot(fig)

        csv = predictions.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

st.divider()

# -----------------------------
# Model Insights
# -----------------------------

st.header("📈 Model Insights")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Feature Importance")

    st.image(
        "reports/figures/feature_importance.png",
        width="stretch"
    )

with col2:

    st.subheader("SHAP Summary")

    st.image(
        "reports/figures/shap_summary.png",
        width="stretch"
    )

st.divider()

# -----------------------------
# Model Comparison
# -----------------------------

st.header("📊 Model Comparison")

st.image(
    "reports/figures/model_comparison.png",
    width="stretch"
)

st.divider()

# -----------------------------
# Footer
# -----------------------------

st.markdown(
    """
<center>

### 👨‍💻 Developed by **Pawan Kumar**

**M.Tech in Computer Science (Data Science)**

Machine Learning Project using

**Logistic Regression • Random Forest • Tuned Random Forest • XGBoost • SHAP • Streamlit**

</center>
""",
    unsafe_allow_html=True
)
