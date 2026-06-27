# 🛒 E-Commerce Conversion Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

Predict whether an e-commerce customer will convert into a buyer using Machine Learning.

This project implements an **end-to-end Machine Learning pipeline**, beginning with exploratory data analysis and feature engineering, followed by training multiple classification models, hyperparameter tuning, explainability using SHAP, and deployment through an interactive Streamlit dashboard.

---

# 📌 Table of Contents

- Overview
- Problem Statement
- Project Workflow
- Dataset
- Features
- Project Structure
- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Machine Learning Models
- Hyperparameter Tuning
- Model Evaluation
- Explainable AI (SHAP)
- Streamlit Dashboard
- Installation
- Usage
- Results
- Future Improvements
- Technologies Used
- Author

---

# 📖 Overview

Customer conversion prediction is an important business problem in e-commerce.

Being able to identify potential buyers before they purchase helps businesses to:

- Improve targeted marketing
- Increase conversion rate
- Reduce advertising costs
- Personalize customer experience
- Improve customer retention

This project predicts whether a customer session will result in a purchase using supervised machine learning algorithms.

---

# 🎯 Problem Statement

Given user browsing behavior such as

- Pages viewed
- Products viewed
- Device type
- Traffic source
- Previous purchases
- Time spent on website

predict whether the customer will complete a purchase.

Target Variable:

```
Converted

0 → No Purchase

1 → Purchase
```

---

# 🚀 Project Workflow

```
Data Collection
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Data Preprocessing
       │
       ▼
Train/Test Split
       │
       ▼
Model Training
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Model Evaluation
       │
       ▼
Feature Importance
       │
       ▼
SHAP Explainability
       │
       ▼
Prediction Pipeline
       │
       ▼
Streamlit Deployment
```

---

# 📊 Dataset

The dataset contains **10,000 customer sessions**.

### Features

| Feature | Description |
|----------|-------------|
| Age | Customer age |
| Income | Annual income |
| City_Tier | Customer city tier |
| Device_Type | Mobile/Desktop/Tablet |
| Traffic_Source | Website traffic source |
| Pages_Viewed | Number of pages visited |
| Products_Viewed | Number of products viewed |
| Time_On_Site | Time spent on website |
| Previous_Purchases | Previous orders |
| Discount_Seen | Whether discount was shown |
| Browser_Version | Browser version |
| Campaign_Code | Marketing campaign |

Target:

```
Converted
```

---

# ⚙️ Feature Engineering

New features created include:

- Views_Per_Product
- Purchase_Engagement
- Time_Per_Page

These engineered features improve the predictive performance of the models.

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation
- Standard Scaling
- One-Hot Encoding
- Train-Test Split

Scikit-Learn Pipelines are used to ensure reproducibility.

---

# 🤖 Machine Learning Models

The following models were trained:

- Logistic Regression
- Random Forest
- Tuned Random Forest
- XGBoost

---

# 🔍 Hyperparameter Tuning

Random Forest was optimized using GridSearchCV.

Parameters tuned:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

---

# 📈 Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | 0.7105 | 0.5576 | 0.2982 | 0.3886 |
| Random Forest | 0.6935 | 0.5054 | 0.3047 | 0.3802 |
| Tuned Random Forest | 0.7065 | 0.5391 | 0.3355 | 0.4136 |
| XGBoost | 0.6980 | 0.5156 | 0.3485 | **0.4159** |

---

# 📉 Feature Importance

XGBoost feature importance was used to identify the most influential features affecting customer conversion.

Example:

- Pages Viewed
- Discount Seen
- Previous Purchases
- Traffic Source
- Time on Site

---

# 🧠 Explainable AI (SHAP)

The project includes SHAP explainability to understand:

- Feature contribution
- Global model interpretation
- Local prediction explanation

Generated visualization:

- SHAP Summary Plot

---

# 💻 Streamlit Dashboard

The project includes a fully interactive dashboard.

Features:

- Upload CSV file
- Predict customer conversion
- Download predictions
- View model comparison
- Display feature importance
- SHAP visualization
- Performance metrics

---

# 📂 Project Structure

```
Ecommerce-Conversion-Prediction
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data
│   ├── raw
│   └── processed
│
├── models
│   ├── xgboost_model.pkl
│   └── preprocessor.pkl
│
├── reports
│   ├── figures
│   └── results
│
├── src
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── tuning.py
│   ├── evaluate.py
│   ├── validation.py
│   ├── predict.py
│   └── save_model.py
```

---

# 🛠 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Ecommerce-Conversion-Prediction.git

cd Ecommerce-Conversion-Prediction
```

Create virtual environment

```bash
python -m venv venv
```

Activate

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training Pipeline

```bash
python main.py
```

---

# ▶️ Launch Dashboard

```bash
streamlit run app.py
```

---

# 📷 Dashboard Preview

### Home Page

> *(Add screenshot here)*

---

### Feature Importance

> *(Add screenshot here)*

---

### SHAP Summary

> *(Add screenshot here)*

---

### Model Comparison

> *(Add screenshot here)*

---

# 📌 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# 🔮 Future Improvements

- LightGBM
- CatBoost
- Optuna Hyperparameter Optimization
- FastAPI Deployment
- Docker Support
- CI/CD using GitHub Actions
- Cloud Deployment (AWS/GCP/Azure)

---

# 👨‍💻 Author

**Pawan Kumar**

M.Tech (Data Science)

Machine Learning | Data Science | Python | AI

---

⭐ If you found this project useful, consider giving it a star.