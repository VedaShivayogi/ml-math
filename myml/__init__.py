"""
myml — Custom ML Regression Package
Built from scratch without Scikit-learn.
"""

from .preprocessing import (
    fill_missing_mean,
    fill_missing_mean_df,
    normalize_features,
    add_intercept,
    detect_outliers_zscore,
)
from .regression import OLSRegression, RidgeRegression, LassoRegression
from .feature_selection import forward_selection, backward_elimination
from .diagnostics import normality_test, vif, heteroscedasticity
from .visualization import plot_actual_vs_predicted, residual_plot
from .prediction import predict_values

__version__ = "0.1.0"
__all__ = [
    # Preprocessing
    "fill_missing_mean",
    "fill_missing_mean_df",
    "normalize_features",
    "add_intercept",
    "detect_outliers_zscore",
    # Regression models
    "OLSRegression",
    "RidgeRegression",
    "LassoRegression",
    # Feature selection
    "forward_selection",
    "backward_elimination",
    # Diagnostics
    "normality_test",
    "vif",
    "heteroscedasticity",
    # Visualization
    "plot_actual_vs_predicted",
    "residual_plot",
    # Prediction
    "predict_values",
]
