"""
Honiara Rainfall Prediction - Full Pipeline (Aligned with CCAReport Template)
=============================================================================
- Data Pre-processing
- Feature Engineering (lag, rolling, cyclic, interaction)
- Year-wise Split (1995-2020) & Unseen Data (2021-2025)
- Principal Component Analysis (PCA)
- Multi-model comparison (Train & Test Metrics: RMSE, MAE, MSE, R2)
- Best model selection
- Prediction on Unseen Data & Visualization
- Pickling Model
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor,
                               VotingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[!] LightGBM not installed - skipping LGBMRegressor")

# ================================================================
# 1. LOAD DATASET
# ================================================================
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df = pd.read_csv("Honiara.csv", skiprows=20)
print(f"   Shape after load : {df.shape}")

# ================================================================
# 2. DATA CLEANING
# ================================================================
print("=" * 70)
print("2. DATA CLEANING")
print("=" * 70)

missing_sentinel = -999.0
for col in df.columns:
    df[col] = df[col].replace(missing_sentinel, np.nan)

# Drop columns with heavy leading-NaN blocks
cols_with_leading_nan = []
for c in df.columns:
    if df[c].isnull().any():
        first_valid = df[c].first_valid_index()
        leading = first_valid if first_valid is not None else len(df)
        if leading > 100:
            cols_with_leading_nan.append(c)
if cols_with_leading_nan:
    df = df.drop(columns=cols_with_leading_nan)

drop_thresh = 0.5
cols_to_drop = [c for c in df.columns if df[c].isnull().mean() > drop_thresh]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)

df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str), format='%Y%j')
df = df.sort_values('DATE').reset_index(drop=True)

remaining_miss = df.isnull().sum().sum()
if remaining_miss > 0:
    df = df.ffill()
    leading_nans = df.isnull().any(axis=1).sum()
    if leading_nans > 0:
        df = df.dropna().reset_index(drop=True)

print(f"   Final shape after cleaning: {df.shape}")

# ================================================================
# 3. FEATURE ENGINEERING
# ================================================================
print("=" * 70)
print("3. FEATURE ENGINEERING")
print("=" * 70)

target = 'PRECTOTCORR'

for lag in [1, 2, 3, 5, 7, 14]:
    df[f'rain_lag{lag}'] = df[target].shift(lag)

shifted_target = df[target].shift(1)
for window in [3, 7, 14, 30]:
    df[f'rain_roll_mean_{window}'] = shifted_target.rolling(window=window).mean()
    df[f'rain_roll_std_{window}']  = shifted_target.rolling(window=window).std()
    df[f'rain_roll_max_{window}']  = shifted_target.rolling(window=window).max()

df['month'] = df['DATE'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['doy_sin']   = np.sin(2 * np.pi * df['DOY'] / 365)
df['doy_cos']   = np.cos(2 * np.pi * df['DOY'] / 365)

df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
df['dewpoint_depression'] = df['T2M'] - df['T2MDEW']
df['humidity_x_soilwet'] = df['RH2M'] * df['GWETTOP']
df['temp_x_humidity']    = df['T2M'] * df['RH2M']
df['was_rainy_yesterday'] = (df[target].shift(1) > 0.5).astype(int)
df['rain_ewm_7'] = df[target].ewm(span=7).mean().shift(1)

df = df.dropna().reset_index(drop=True)

# ================================================================
# 4. TRAIN / TEST SPLIT (YEAR-WISE RANDOM 80:20)
# ================================================================
print("=" * 70)
print("4. DATA SPLITTING (Year-wise)")
print("=" * 70)

df_95_20 = df[(df['YEAR'] >= 1995) & (df['YEAR'] <= 2020)].copy()
unseen_df = df[(df['YEAR'] >= 2021) & (df['YEAR'] <= 2025)].copy()

years = df_95_20['YEAR'].unique()
train_years, test_years = train_test_split(years, test_size=0.2, random_state=42)

train_df = df_95_20[df_95_20['YEAR'].isin(train_years)].copy()
test_df = df_95_20[df_95_20['YEAR'].isin(test_years)].copy()

print(f"   Train years (80%): {sorted(train_years)}")
print(f"   Test years  (20%): {sorted(test_years)}")

exclude_cols = ['YEAR', 'DOY', 'DATE', 'month', target]
features = [c for c in df.columns if c not in exclude_cols]

X_train = train_df[features].values
y_train_raw = train_df[target].values
X_test  = test_df[features].values
y_test  = test_df[target].values

y_train = np.log1p(y_train_raw)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ================================================================
print("=" * 70)
print("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 70)

pca = PCA().fit(X_train_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance by Components')
plt.grid(True)
plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()

n_90 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
n_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"   Components for 90% variance: {n_90}")
print(f"   Components for 95% variance: {n_95}")
print("   [OK] PCA plot saved -> pca_variance.png")

# ================================================================
# 6. MODEL COMPARISON
# ================================================================
print("=" * 70)
print("6. MODEL COMPARISON")
print("=" * 70)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbosity=0),
    'Extra Trees': ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
    'SVR (RBF)': SVR(kernel='rbf', C=5, epsilon=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=150, learning_rate=0.05, random_state=42),
}

if HAS_LGBM:
    models['LightGBM'] = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)

# Base models for Ensembling
stack_estimators = [
    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbosity=0)),
    ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
]
if HAS_LGBM:
    stack_estimators.append(('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)))

# Add Ensembles to the mix
models['Voting Regressor'] = VotingRegressor(estimators=stack_estimators, n_jobs=-1)
models['Stacking Regressor'] = StackingRegressor(estimators=stack_estimators, final_estimator=Ridge(), n_jobs=-1)

results = []

for name, model in models.items():
    if name in ['SVR (RBF)', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # Invert log1p for proper evaluation
    y_train_pred_orig = np.maximum(np.expm1(y_train_pred), 0)
    y_test_pred_orig = np.maximum(np.expm1(y_test_pred), 0)

    # Train Metrics
    train_mse = mean_squared_error(y_train_raw, y_train_pred_orig)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_raw, y_train_pred_orig)
    train_r2 = r2_score(y_train_raw, y_train_pred_orig)

    # Test Metrics
    test_mse = mean_squared_error(y_test, y_test_pred_orig)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred_orig)
    test_r2 = r2_score(y_test, y_test_pred_orig)

    results.append({
        'Model': name,
        'Train R2': train_r2, 'Test R2': test_r2,
        'Train RMSE': train_rmse, 'Test RMSE': test_rmse,
        'Train MAE': train_mae, 'Test MAE': test_mae,
        'Train MSE': train_mse, 'Test MSE': test_mse,
        'model_obj': model
    })
    print(f"   {name:<20s} | Test R2: {test_r2:.4f} | Test RMSE: {test_rmse:.3f}")

df_results = pd.DataFrame(results).drop(columns=['model_obj'])
df_results.to_csv('model_comparison_full.csv', index=False)

# ================================================================
# 7. BEST MODEL & PREDICTION ON UNSEEN DATA (2021-2025)
# ================================================================
print("=" * 70)
print("7. UNSEEN DATA PREDICTION (2021-2025)")
print("=" * 70)

best_result = sorted(results, key=lambda x: x['Test R2'], reverse=True)[0]
best_model = best_result['model_obj']
best_name = best_result['Model']

print(f"   >>> Best Model Selected: {best_name}")

X_unseen = unseen_df[features].values
y_unseen_actual = unseen_df[target].values
unseen_dates = unseen_df['DATE']

if best_name in ['SVR (RBF)', 'KNN']:
    X_unseen_scaled = scaler.transform(X_unseen)
    y_unseen_pred_log = best_model.predict(X_unseen_scaled)
else:
    y_unseen_pred_log = best_model.predict(X_unseen)

y_unseen_pred = np.maximum(np.expm1(y_unseen_pred_log), 0)

unseen_r2 = r2_score(y_unseen_actual, y_unseen_pred)
unseen_rmse = np.sqrt(mean_squared_error(y_unseen_actual, y_unseen_pred))
print(f"   Unseen Data R2  : {unseen_r2:.4f}")
print(f"   Unseen Data RMSE: {unseen_rmse:.3f}")

df_unseen_results = pd.DataFrame({
    'Date': unseen_dates,
    'Actual Rainfall (mm)': y_unseen_actual,
    'Predicted Rainfall (mm)': y_unseen_pred
})
df_unseen_results.to_csv('unseen_predictions_2021_2025.csv', index=False)
print("   [OK] Predictions saved -> unseen_predictions_2021_2025.csv")

# Visualization for Unseen Data
plt.figure(figsize=(14, 6))
# Plot first 100 days for clarity
plt.plot(unseen_dates[:100], y_unseen_actual[:100], label='Actual', marker='o', markersize=3)
plt.plot(unseen_dates[:100], y_unseen_pred[:100], label='Predicted', marker='x', markersize=3)
plt.title(f'Actual vs Predicted Rainfall (First 100 days of Unseen Data) - {best_name}')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm/day)')
plt.legend()
plt.tight_layout()
plt.savefig('unseen_predictions_plot.png', dpi=150)
plt.close()

# ================================================================
# 8. CREATE PICKLE FILE
# ================================================================
print("=" * 70)
print("8. SAVING PICKLE FILES")
print("=" * 70)

with open('best_rainfall_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)
if best_name in ['SVR (RBF)', 'KNN']:
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

print("   [OK] Model saved to best_rainfall_model.pkl")
print("   [OK] Feature names saved to feature_names.pkl")

print("\nDONE!")
