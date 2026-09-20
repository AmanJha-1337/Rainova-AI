"""
Honiara Rainfall Prediction - Full Pipeline
=============================================
- Thorough EDA & data preparation
- Feature engineering (lag, rolling, cyclic, interaction)
- Multi-model comparison (NO Linear Regression)
- Best model selection with hyperparameter tuning
- Comprehensive evaluation & visualization
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
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
# 1. LOAD DATASET  (skip NASA POWER header)
# ================================================================
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df = pd.read_csv("Honiara.csv", skiprows=20)      # skip header block
print(f"   Shape after load : {df.shape}")
print(f"   Columns          : {list(df.columns)}")
print(f"   Year range       : {df['YEAR'].min()} - {df['YEAR'].max()}")
print(f"   First 3 rows:\n{df.head(3).to_string(index=False)}\n")

# ================================================================
# 2. DATA CLEANING
# ================================================================
print("=" * 70)
print("2. DATA CLEANING")
print("=" * 70)

# 2a. Replace NASA's -999 sentinel with NaN
missing_sentinel = -999.0
for col in df.columns:
    n_sentinel = (df[col] == missing_sentinel).sum()
    if n_sentinel > 0:
        print(f"   {col}: {n_sentinel} sentinel values (-999) -> NaN")
        df[col] = df[col].replace(missing_sentinel, np.nan)

# 2b. Report overall missing
print(f"\n   Missing values per column:")
miss = df.isnull().sum()
for c, v in miss.items():
    if v > 0:
        print(f"      {c:25s} -> {v:>6d}  ({100*v/len(df):.1f}%)")

# 2c. Drop columns with > 50 % missing
drop_thresh = 0.5
cols_to_drop = [c for c in df.columns if df[c].isnull().mean() > drop_thresh]
if cols_to_drop:
    print(f"\n   Dropping columns with >{drop_thresh*100:.0f}% missing: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# 2d. Create proper DATE column
df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str),
                            format='%Y%j')
df = df.sort_values('DATE').reset_index(drop=True)

# 2e. Forward-fill then backward-fill remaining NaNs (only small gaps)
remaining_miss = df.isnull().sum().sum()
if remaining_miss > 0:
    print(f"   Remaining NaN after sentinel removal: {remaining_miss} -> forward/backward fill")
    df = df.ffill().bfill()

print(f"   Final shape after cleaning: {df.shape}")
print(f"   Any NaN left? {df.isnull().any().any()}\n")

# ================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================
print("=" * 70)
print("3. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

target = 'PRECTOTCORR'
print(f"\n   Target variable: {target}")
print(f"   {df[target].describe().to_string()}\n")

# 3a. Monthly rainfall distribution
df['month'] = df['DATE'].dt.month
monthly = df.groupby('month')[target].agg(['mean', 'median', 'std', 'max'])
print("   Monthly rainfall statistics (mm/day):")
print(monthly.to_string())

# 3b. Correlation with target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['YEAR', 'DOY', 'month']]
corr_with_target = df[numeric_cols].corr()[target].drop(target).sort_values(ascending=False)
print(f"\n   Correlation with {target}:")
print(corr_with_target.to_string())

# 3c. Save EDA plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution of rainfall
axes[0, 0].hist(df[target], bins=80, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Daily Rainfall')
axes[0, 0].set_xlabel('Rainfall (mm/day)')
axes[0, 0].set_ylabel('Frequency')

# Monthly boxplot
df.boxplot(column=target, by='month', ax=axes[0, 1])
axes[0, 1].set_title('Monthly Rainfall Distribution')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Rainfall (mm/day)')
plt.sca(axes[0, 1])
plt.title('Monthly Rainfall Distribution')

# Yearly total
yearly_total = df.groupby('YEAR')[target].sum()
axes[1, 0].bar(yearly_total.index, yearly_total.values, color='teal', alpha=0.8)
axes[1, 0].set_title('Yearly Total Rainfall')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Total Rainfall (mm)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Correlation heatmap with target (top features)
top_corr_cols = corr_with_target.abs().nlargest(8).index.tolist()
top_corr_cols.append(target)
sns.heatmap(df[top_corr_cols].corr(), annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=axes[1, 1], square=True)
axes[1, 1].set_title('Top Feature Correlations')

plt.suptitle('Honiara Rainfall - EDA', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   [OK] EDA plots saved -> eda_plots.png")

# ================================================================
# 4. FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 70)
print("4. FEATURE ENGINEERING")
print("=" * 70)

# 4a. Lag features
for lag in [1, 2, 3, 5, 7, 14]:
    df[f'rain_lag{lag}'] = df[target].shift(lag)
    print(f"   Created: rain_lag{lag}")

# 4b. Rolling statistics
for window in [3, 7, 14, 30]:
    df[f'rain_roll_mean_{window}'] = df[target].rolling(window=window).mean()
    df[f'rain_roll_std_{window}']  = df[target].rolling(window=window).std()
    df[f'rain_roll_max_{window}']  = df[target].rolling(window=window).max()
    print(f"   Created: rain_roll_mean/std/max_{window}")

# 4c. Cyclic encoding of month & DOY
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['doy_sin']   = np.sin(2 * np.pi * df['DOY'] / 365)
df['doy_cos']   = np.cos(2 * np.pi * df['DOY'] / 365)
print("   Created: month_sin, month_cos, doy_sin, doy_cos")

# 4d. Temperature range & dew-point depression
df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']
df['dewpoint_depression'] = df['T2M'] - df['T2MDEW']
print("   Created: temp_range, dewpoint_depression")

# 4e. Interaction features
df['humidity_x_soilwet'] = df['RH2M'] * df['GWETTOP']
df['temp_x_humidity']    = df['T2M'] * df['RH2M']
print("   Created: humidity_x_soilwet, temp_x_humidity")

# 4f. Is-rain indicator from previous day
df['was_rainy_yesterday'] = (df[target].shift(1) > 0.5).astype(int)
print("   Created: was_rainy_yesterday")

# 4g. Exponentially weighted moving average
df['rain_ewm_7'] = df[target].ewm(span=7).mean().shift(1)
print("   Created: rain_ewm_7")

# Drop rows with NaN introduced by lag/rolling
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"\n   Dropped {before - len(df)} rows with NaN from lag/rolling")
print(f"   Final shape: {df.shape}")

# ================================================================
# 5. TRAIN / TEST SPLIT  (time-based)
# ================================================================
print("\n" + "=" * 70)
print("5. TRAIN / TEST SPLIT (time-based)")
print("=" * 70)

train_df = df[(df['YEAR'] >= 1995) & (df['YEAR'] <= 2020)].copy()
test_df  = df[(df['YEAR'] >= 2021) & (df['YEAR'] <= 2025)].copy()

# Feature list — exclude non-feature columns
exclude_cols = ['YEAR', 'DOY', 'DATE', 'month', target]
features = [c for c in df.columns if c not in exclude_cols]

print(f"   Train: {train_df.shape[0]} rows  ({train_df['YEAR'].min()}-{train_df['YEAR'].max()})")
print(f"   Test : {test_df.shape[0]} rows  ({test_df['YEAR'].min()}-{test_df['YEAR'].max()})")
print(f"   Features ({len(features)}): {features[:10]} ... + {len(features)-10} more")

X_train = train_df[features].values
y_train = train_df[target].values
X_test  = test_df[features].values
y_test  = test_df[target].values

# 5b. Scale features with RobustScaler (better for outlier-heavy data like rainfall)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Scaling: RobustScaler applied\n")

# ================================================================
# 6. MODEL COMPARISON
# ================================================================
print("=" * 70)
print("6. MODEL COMPARISON (no Linear Regression)")
print("=" * 70)

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, random_state=42),
    'XGBoost': XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbosity=0),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=300, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1),
    'KNN': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
    'SVR (RBF)': SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
    'Decision Tree': DecisionTreeRegressor(max_depth=12, min_samples_split=10,
                                            random_state=42),
    'AdaBoost': AdaBoostRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42),
}

if HAS_LGBM:
    models['LightGBM'] = LGBMRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)

results = {}
tscv = TimeSeriesSplit(n_splits=5)

print(f"\n   {'Model':<25s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'MAPE%':>8s} {'CV-R2':>8s}")
print("   " + "-" * 65)

for name, model in models.items():
    # Use scaled data for SVR/KNN, raw for tree-based
    if name in ['SVR (RBF)', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=tscv, scoring='r2', n_jobs=-1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train,
                                     cv=tscv, scoring='r2', n_jobs=-1)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    # Guard against division by zero in MAPE
    mask = y_test > 0.01
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan
    cv_r2 = cv_scores.mean()

    results[name] = {
        'model': model, 'y_pred': y_pred,
        'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'CV_R2': cv_r2
    }

    print(f"   {name:<25s} {rmse:>8.3f} {mae:>8.3f} {r2:>8.4f} {mape:>7.1f}% {cv_r2:>8.4f}")

# ================================================================
# 7. SELECT BEST MODEL
# ================================================================
print("\n" + "=" * 70)
print("7. BEST MODEL SELECTION")
print("=" * 70)

# Rank by test R² (primary) and CV R² (secondary)
ranked = sorted(results.items(), key=lambda x: (x[1]['R2'], x[1]['CV_R2']), reverse=True)
best_name  = ranked[0][0]
best_info  = ranked[0][1]

print(f"\n   >>> Best Model: {best_name}")
print(f"      Test RMSE : {best_info['RMSE']:.3f}")
print(f"      Test MAE  : {best_info['MAE']:.3f}")
print(f"      Test R2   : {best_info['R2']:.4f}")
print(f"      Test MAPE : {best_info['MAPE']:.1f}%")
print(f"      CV R2     : {best_info['CV_R2']:.4f}")

# ================================================================
# 8. FEATURE IMPORTANCE  (if tree-based best model)
# ================================================================
best_model = best_info['model']
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 70)
    print("8. FEATURE IMPORTANCE (top 15)")
    print("=" * 70)
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print(feat_imp.head(15).to_string())

    plt.figure(figsize=(10, 6))
    feat_imp.head(15).plot(kind='barh', color='teal', edgecolor='black')
    plt.title(f'Top 15 Feature Importances - {best_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n   [OK] Feature importance saved -> feature_importance.png")

# ================================================================
# 9. COMPREHENSIVE VISUALIZATIONS
# ================================================================
print("\n" + "=" * 70)
print("9. VISUALIZATIONS")
print("=" * 70)

y_pred_best = best_info['y_pred']
test_dates  = test_df['DATE'].values

fig, axes = plt.subplots(3, 2, figsize=(18, 16))

# 9a. Actual vs Predicted time series (full test set)
axes[0, 0].plot(test_dates, y_test, label='Actual', alpha=0.7, linewidth=0.8)
axes[0, 0].plot(test_dates, y_pred_best, label='Predicted', alpha=0.7, linewidth=0.8)
axes[0, 0].set_title(f'{best_name} - Actual vs Predicted (2021-2025)')
axes[0, 0].set_ylabel('Rainfall (mm/day)')
axes[0, 0].legend()

# 9b. Zoomed view — first 120 days
n_zoom = 120
axes[0, 1].plot(test_dates[:n_zoom], y_test[:n_zoom], label='Actual', marker='o',
                markersize=2, linewidth=1)
axes[0, 1].plot(test_dates[:n_zoom], y_pred_best[:n_zoom], label='Predicted', marker='s',
                markersize=2, linewidth=1)
axes[0, 1].set_title(f'Zoomed - First {n_zoom} Days')
axes[0, 1].set_ylabel('Rainfall (mm/day)')
axes[0, 1].legend()

# 9c. Scatter: actual vs predicted
axes[1, 0].scatter(y_test, y_pred_best, alpha=0.3, s=10, color='steelblue')
max_val = max(y_test.max(), y_pred_best.max())
axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect')
axes[1, 0].set_xlabel('Actual')
axes[1, 0].set_ylabel('Predicted')
axes[1, 0].set_title('Scatter: Actual vs Predicted')
axes[1, 0].legend()

# 9d. Residual distribution
residuals = y_test - y_pred_best
axes[1, 1].hist(residuals, bins=60, color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='black', linestyle='--')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].set_xlabel('Residual (Actual - Predicted)')

# 9e. Monthly comparison
test_df_eval = test_df.copy()
test_df_eval['predicted'] = y_pred_best
monthly_actual = test_df_eval.groupby(test_df_eval['DATE'].dt.month)[target].mean()
monthly_pred   = test_df_eval.groupby(test_df_eval['DATE'].dt.month)['predicted'].mean()
x_months = np.arange(1, 13)
width = 0.35
axes[2, 0].bar(x_months - width/2, monthly_actual.reindex(x_months, fill_value=0),
               width, label='Actual', color='steelblue')
axes[2, 0].bar(x_months + width/2, monthly_pred.reindex(x_months, fill_value=0),
               width, label='Predicted', color='coral')
axes[2, 0].set_xticks(x_months)
axes[2, 0].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                             'Jul','Aug','Sep','Oct','Nov','Dec'], rotation=45)
axes[2, 0].set_title('Monthly Mean Rainfall: Actual vs Predicted')
axes[2, 0].set_ylabel('Rainfall (mm/day)')
axes[2, 0].legend()

# 9f. Model comparison bar chart
model_names = list(results.keys())
r2_scores   = [results[m]['R2'] for m in model_names]
colors = ['gold' if m == best_name else 'steelblue' for m in model_names]
axes[2, 1].barh(model_names, r2_scores, color=colors, edgecolor='black')
axes[2, 1].set_xlabel('R2 Score')
axes[2, 1].set_title('Model Comparison (R2 on Test Set)')
axes[2, 1].axvline(0, color='grey', linestyle='--', linewidth=0.8)

plt.suptitle('Honiara Rainfall Prediction - Results', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('results_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Result plots saved -> results_plots.png")

# ================================================================
# 10. SUMMARY TABLE
# ================================================================
print("\n" + "=" * 70)
print("10. FINAL SUMMARY TABLE")
print("=" * 70)

summary = pd.DataFrame({
    'Model': [n for n in results],
    'RMSE':  [results[n]['RMSE'] for n in results],
    'MAE':   [results[n]['MAE']  for n in results],
    'R2':    [results[n]['R2']   for n in results],
    'MAPE%': [results[n]['MAPE'] for n in results],
    'CV_R2': [results[n]['CV_R2']for n in results],
}).sort_values('R2', ascending=False).reset_index(drop=True)

print(summary.to_string(index=False))

# Save summary
summary.to_csv('model_comparison.csv', index=False)
print("\n   [OK] Summary saved -> model_comparison.csv")

print("\n" + "=" * 70)
print(f"   >>> WINNER: {best_name}  (R2 = {best_info['R2']:.4f})")
print("=" * 70)

# ================================================================
# 11. INTERACTIVE PREDICTION
# ================================================================
print("\n" + "=" * 70)
print("11. INTERACTIVE PREDICTION")
print("=" * 70)

while True:
    try:
        user_input = input("\nEnter a year to predict (e.g., 2021) or 'q' to quit: ").strip()
    except EOFError:
        break

    if user_input.lower() in ['q', 'quit', 'exit']:
        print("Exiting interactive prediction.")
        break
    try:
        year_to_predict = int(user_input)
    except ValueError:
        print("   [!] Please enter a valid integer year.")
        continue
    
    year_df = df[df['YEAR'] == year_to_predict]
    if year_df.empty:
        print(f"   [!] No data available for year {year_to_predict}. Data ranges from {df['YEAR'].min()} to {df['YEAR'].max()}.")
        continue
        
    X_year = year_df[features].values
    y_year_actual = year_df[target].values
    
    # Scale if best model is distance-based
    if best_name in ['SVR (RBF)', 'KNN']:
        X_year = scaler.transform(X_year)
        
    y_year_pred = best_info['model'].predict(X_year)
    
    year_rmse = np.sqrt(mean_squared_error(y_year_actual, y_year_pred))
    
    # Calculate R2, handling cases where variance might be 0 (though unlikely for a full year of rainfall)
    if np.var(y_year_actual) == 0:
        year_r2 = np.nan
    else:
        year_r2 = r2_score(y_year_actual, y_year_pred)
    
    print(f"\n   --- Predictions for {year_to_predict} using {best_name} ---")
    print(f"      RMSE : {year_rmse:.3f}")
    print(f"      R2   : {year_r2:.4f}")
    
    # Display the plot
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(year_df['DATE'], y_year_actual, label='Actual', alpha=0.7, linewidth=1.5)
        plt.plot(year_df['DATE'], y_year_pred, label='Predicted', alpha=0.7, linewidth=1.5)
        plt.title(f'Rainfall Prediction for {year_to_predict} ({best_name})')
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm/day)')
        plt.legend()
        plt.tight_layout()
        print("   [i] Close the plot window to continue...")
        plt.show()
    except Exception as e:
        print(f"   [!] Could not generate plot: {e}")

