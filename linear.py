# ==========================================================
# IMPORT REQUIRED LIBRARIES
# ==========================================================

import numpy as np
import pandas as pd

from myml.preprocessing import *
from myml.regression import *
from myml.feature_selection import *
from myml.diagnostics import *
from myml.visualization import *

# ==========================================================
# FEATURE NORMALIZATION (FIX FOR LASSO ISSUE)
# ==========================================================

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std

def clean_nan(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("magicbricks_hyderabad.csv")

print("\nDATASET HEAD\n")
print(data.head())

# ==========================================================
# MISSING VALUES
# ==========================================================

print("\nMISSING VALUES\n")
print(data.isnull().sum())

numeric_columns = ['bhk', 'area_sqft', 'price_per_sqft']

data[numeric_columns] = fill_missing_mean(data[numeric_columns])

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = data[['bhk', 'area_sqft', 'price_per_sqft']].values
y = data['price_lakhs'].values

# ==========================================================
# NORMALIZE FEATURES (IMPORTANT FIX)
# ==========================================================

X = normalize_features(X)

# ==========================================================
# OUTLIERS
# ==========================================================

print("\nOUTLIERS\n")
outliers = detect_outliers_zscore(data['price_lakhs'])
print(outliers)

# ==========================================================
# ADD INTERCEPT
# ==========================================================

X = add_intercept(X)

print("\nFEATURE MATRIX WITH INTERCEPT\n")
print(X[:5])

# ==========================================================
# OLS REGRESSION
# ==========================================================

print("\nOLS REGRESSION\n")

ols_model = OLSRegression()
ols_model.fit(X, y)

ols_predictions = ols_model.predict(X)

print("\nOLS COEFFICIENTS\n")
print(ols_model.beta)

# ==========================================================
# RIDGE REGRESSION
# ==========================================================

print("\nRIDGE REGRESSION\n")

ridge_model = RidgeRegression(alpha=1.0)
ridge_model.fit(X, y)

ridge_predictions = ridge_model.predict(X)

print("\nRIDGE COEFFICIENTS\n")
print(ridge_model.beta)

# ==========================================================
# LASSO REGRESSION (FIXED STABILITY)
# ==========================================================

print("\nLASSO REGRESSION\n")

lasso_model = LassoRegression(alpha=0.01)
lasso_model.fit(X, y)

lasso_predictions = lasso_model.predict(X)

# FIX NaN ISSUE
lasso_predictions = clean_nan(lasso_predictions)
y_clean = clean_nan(y)

print("\nLASSO COEFFICIENTS\n")
print(lasso_model.beta)

# ==========================================================
# FORWARD SELECTION
# ==========================================================

print("\nFORWARD SELECTION\n")
selected_features = forward_selection(X, y)
print(selected_features)

# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

print("\nOLS PREDICTIONS\n")
print(ols_predictions[:10])

print("\nRIDGE PREDICTIONS\n")
print(ridge_predictions[:10])

print("\nLASSO PREDICTIONS\n")
print(lasso_predictions[:10])

# ==========================================================
# RESIDUALS
# ==========================================================

residuals = y - ols_predictions

# ==========================================================
# NORMALITY TEST
# ==========================================================

print("\nNORMALITY TEST\n")
stat, p = normality_test(residuals)

print("Statistic:", stat)
print("P-value:", p)

if p > 0.05:
    print("Residuals are Normally Distributed")
else:
    print("Residuals are NOT Normally Distributed")

# ==========================================================
# MULTICOLLINEARITY (VIF)
# ==========================================================

print("\nVIF VALUES\n")
vif_values = vif(X)
print(vif_values)

# ==========================================================
# HETEROSCEDASTICITY
# ==========================================================

print("\nHETEROSCEDASTICITY\n")
hetero = heteroscedasticity(residuals)
print("Variance of Residuals:", hetero)

# ==========================================================
# OLS EVALUATION
# ==========================================================

mse = np.mean((y - ols_predictions) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - ols_predictions))

ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - ols_predictions) ** 2)
r2 = 1 - (ss_residual / ss_total)

print("\nOLS MODEL EVALUATION\n")
print("MSE :", mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R2 SCORE :", r2)

# ==========================================================
# RIDGE EVALUATION
# ==========================================================

ridge_mse = np.mean((y - ridge_predictions) ** 2)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = np.mean(np.abs(y - ridge_predictions))

ss_residual = np.sum((y - ridge_predictions) ** 2)
ridge_r2 = 1 - (ss_residual / ss_total)

print("\nRIDGE MODEL EVALUATION\n")
print("MSE :", ridge_mse)
print("RMSE :", ridge_rmse)
print("MAE :", ridge_mae)
print("R2 SCORE :", ridge_r2)

# ==========================================================
# RIDGE LOSS
# ==========================================================

alpha = 1.0

ridge_loss = (
    np.mean((y - ridge_predictions) ** 2)
    + alpha * np.sum(ridge_model.beta ** 2)
)

print("\nRIDGE LOSS\n")
print(ridge_loss)

# ==========================================================
# LASSO EVALUATION (FIXED)
# ==========================================================

lasso_mse = np.mean((y_clean - lasso_predictions) ** 2)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = np.mean(np.abs(y_clean - lasso_predictions))

ss_total = np.sum((y_clean - np.mean(y_clean)) ** 2)
ss_residual = np.sum((y_clean - lasso_predictions) ** 2)

lasso_r2 = 1 - (ss_residual / ss_total)

print("\nLASSO MODEL EVALUATION\n")
print("MSE :", lasso_mse)
print("RMSE :", lasso_rmse)
print("MAE :", lasso_mae)
print("R2 SCORE :", lasso_r2)

# ==========================================================
# VISUALIZATION
# ==========================================================

plot_actual_vs_predicted(y, ols_predictions)
residual_plot(residuals)

# ==========================================================
# CUSTOM PREDICTION (FIXED)
# ==========================================================

print("\nCUSTOM PREDICTION\n")

# after normalization + intercept
new_house = np.array([[3, 1800, 0.05]])
new_house = normalize_features(new_house)
new_house = add_intercept(new_house)

predicted_price = ols_model.predict(new_house)

print("Predicted House Price (Lakhs):")
print(predicted_price)

# ==========================================================
# SAMPLE MODEL TEST
# ==========================================================

print("\nSIMPLE SAMPLE EXAMPLE\n")

sample_data = pd.read_csv("magicbricks_hyderabad.csv")
sample_data[numeric_columns] = fill_missing_mean(sample_data[numeric_columns])

X_sample = sample_data[['bhk', 'area_sqft', 'price_per_sqft']].values
y_sample = sample_data['price_lakhs'].values

X_sample = normalize_features(X_sample)
X_sample = add_intercept(X_sample)

sample_model = OLSRegression()
sample_model.fit(X_sample, y_sample)

sample_predictions = sample_model.predict(X_sample)

plot_actual_vs_predicted(y_sample, sample_predictions)

print("\nSAMPLE PREDICTIONS\n")
print(sample_predictions[:10])

print("\nPROGRAM COMPLETED SUCCESSFULLY")