import numpy as np
import pandas as pd

# ==========================================================
# MISSING VALUE HANDLING (NUMPY VERSION)
# ==========================================================

def fill_missing_mean(X):
    X = np.array(X, dtype=float)

    for i in range(X.shape[1]):
        col = X[:, i]
        mean_val = np.nanmean(col)
        col[np.isnan(col)] = mean_val
        X[:, i] = col

    return X


# ==========================================================
# MISSING VALUE HANDLING (DATAFRAME VERSION)
# ==========================================================

def fill_missing_mean_df(df):
    return df.fillna(df.mean())


# ==========================================================
# FEATURE NORMALIZATION
# ==========================================================

def normalize_features(X):
    X = np.array(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std == 0] = 1  # avoid division by zero

    return (X - mean) / std


# ==========================================================
# ADD INTERCEPT TERM
# ==========================================================

def add_intercept(X):
    X = np.array(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


# ==========================================================
# OUTLIER DETECTION (Z-SCORE METHOD)
# ==========================================================

def detect_outliers_zscore(data, threshold=3):
    data = np.array(data, dtype=float)

    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return np.array([])

    z_scores = (data - mean) / std
    return np.where(np.abs(z_scores) > threshold)