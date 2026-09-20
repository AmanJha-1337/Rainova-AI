# Rainova AI

<p align="center">
  <img src="assets/rainova-ai-logo.png" alt="Rainova AI Logo" width="220">
</p>

<p align="center">
  <b>Machine Learning Based Rainfall Prediction</b>
</p>

---

## Purpose of the Project

As a Civil Engineering student, rainfall is more than just a weather condition. It is an important factor in areas such as drainage design, water resource management, rainwater harvesting, flood analysis, agriculture, and infrastructure planning.

While working with these engineering applications, I became interested in understanding whether historical weather data could be used to predict rainfall using Machine Learning.

This led to the development of **Rainova AI** — a rainfall prediction project that combines environmental data, time-based patterns, feature engineering, and machine learning models to predict rainfall.

---

## Problem I Solved

Rainfall is affected by multiple environmental and atmospheric conditions, making it difficult to understand its behaviour using a single variable.

The problem addressed in this project is:

> **Can historical weather and environmental data be used to predict future rainfall using Machine Learning?**

To solve this problem, I developed an end-to-end machine learning pipeline that includes:

- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Time-based feature creation
- Feature interaction
- PCA analysis
- Multiple machine learning models
- Model comparison
- Model evaluation
- Prediction on unseen data
- Model serialization

---

## How Rainova AI Works

Rainova AI uses historical weather data for **Honiara** and predicts corrected daily precipitation (`PRECTOTCORR`) in mm/day.

The dataset contains environmental parameters such as:

- Temperature
- Minimum temperature
- Maximum temperature
- Dew point temperature
- Relative humidity
- Specific humidity
- Wind speed
- Soil wetness
- Solar radiation
- Previous rainfall

The project uses data from **1995 to 2025**.

The historical data from **1995–2020** is used for model development, while **2021–2025** is treated as unseen data for prediction.

---

## Project Architecture

```text
                    ┌─────────────────────┐
                    │   NASA POWER Data   │
                    │      1995–2025       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    │ Missing Values      │
                    │ Data Formatting      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │ Rainfall Patterns   │
                    │ Correlations        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Lag Features        │
                    │ Rolling Statistics  │
                    │ Time Features       │
                    │ Feature Interaction │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Preparation  │
                    │ Scaling + PCA       │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Machine Learning          │
              │                                 │
              │ Random Forest                   │
              │ Gradient Boosting               │
              │ XGBoost                         │
              │ Extra Trees                     │
              │ LightGBM                        │
              │ Decision Tree                   │
              │ SVR                             │
              │ KNN                             │
              │ AdaBoost                        │
              │ Voting / Stacking               │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ R² / RMSE / MAE     │
                    │ MSE / MAPE          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Best Model        │
                    │ Gradient Boosting   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌──────────────────┐
          │ Unseen Data     │   │ Saved ML Model   │
          │ 2021–2025       │   │ Pickle (.pkl)    │
          └────────┬────────┘   └──────────────────┘
                   │
                   ▼
          ┌─────────────────────┐
          │ Rainfall Prediction │
          │ Actual vs Predicted │
          └─────────────────────┘
```
##Project Structure

Rainova-AI/
│
├── assets/
│   └── rainova-ai-logo.png
│
├── Honiara.csv
│
├── rainfall_prediction.py
├── model1.py
│
├── best_rainfall_model.pkl
├── feature_names.pkl
│
├── model_comparison.csv
├── model_comparison_full.csv
├── unseen_predictions_2021_2025.csv
│
├── eda_plots.png
├── feature_importance.png
├── pca_variance.png
├── results_plots.png
├── unseen_predictions_plot.png
│
└── pdf_content.txt

---

##Author
Aman Kumar Jha
