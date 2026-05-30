import numpy as np
from scipy.stats import shapiro

def normality_test(residuals):
    stat, p = shapiro(residuals)
    return stat, p

def vif(X):
    vif_scores = []
    for i in range(X.shape[1]):
        Xi = X[:, i]
        X_other = np.delete(X, i, axis=1)
        beta = np.linalg.inv(X_other.T @ X_other) @ X_other.T @ Xi
        pred = X_other @ beta
        r2 = 1 - (np.sum((Xi - pred) ** 2) / np.sum((Xi - np.mean(Xi)) ** 2))
        vif_value = 1 / (1 - r2)
        vif_scores.append(vif_value)
    return vif_scores

def heteroscedasticity(residuals):
    return np.var(residuals)
