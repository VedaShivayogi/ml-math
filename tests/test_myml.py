"""
Tests for myml package.
Run with: pytest tests/
"""
import numpy as np
import pytest
from myml.preprocessing import fill_missing_mean, normalize_features, add_intercept, detect_outliers_zscore
from myml.regression import OLSRegression, RidgeRegression, LassoRegression
from myml.feature_selection import forward_selection
from myml.diagnostics import normality_test, vif, heteroscedasticity
from myml.prediction import predict_values

# Non-collinear data so OLS matrix is invertible
X = np.array([[1, 2], [3, 5], [5, 3], [7, 9]], dtype=float)
y = np.array([2.0, 4.5, 5.0, 8.5])


def test_fill_missing_mean():
    X_nan = np.array([[1.0, np.nan], [3.0, 4.0]])
    result = fill_missing_mean(X_nan)
    assert not np.isnan(result).any()


def test_normalize_features():
    result = normalize_features(X)
    assert result.shape == X.shape


def test_add_intercept():
    result = add_intercept(X)
    assert result.shape == (4, 3)
    assert np.all(result[:, 0] == 1)


def test_detect_outliers_zscore():
    # Use a smaller threshold so 100 is definitely caught
    data = np.array([1, 2, 3, 4, 100], dtype=float)
    outliers = detect_outliers_zscore(data, threshold=1.5)
    assert len(outliers[0]) > 0


def test_ols_regression():
    Xi = add_intercept(X)
    model = OLSRegression()
    model.fit(Xi, y)
    preds = model.predict(Xi)
    assert preds.shape == y.shape


def test_ridge_regression():
    Xi = add_intercept(X)
    model = RidgeRegression(alpha=0.1)
    model.fit(Xi, y)
    preds = model.predict(Xi)
    assert preds.shape == y.shape


def test_lasso_regression():
    Xi = add_intercept(X)
    model = LassoRegression(alpha=0.01, iterations=100)
    model.fit(Xi, y)
    preds = model.predict(Xi)
    assert preds.shape == y.shape


def test_forward_selection():
    selected = forward_selection(X, y)
    assert len(selected) == X.shape[1]


def test_normality_test():
    residuals = np.array([0.1, -0.2, 0.05, 0.15])
    stat, p = normality_test(residuals)
    assert 0 <= p <= 1


def test_vif():
    scores = vif(X)
    assert len(scores) == X.shape[1]


def test_heteroscedasticity():
    residuals = np.array([0.1, -0.2, 0.05, 0.15])
    var = heteroscedasticity(residuals)
    assert var >= 0


def test_predict_values():
    beta = np.array([0.0, 1.0, 1.0])
    Xi = add_intercept(X)
    preds = predict_values(Xi, beta)
    assert preds.shape == (4,)
