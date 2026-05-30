<<<<<<< HEAD
# myml-regression

A custom machine learning regression package built **from scratch** without Scikit-learn.

## Features

| Module | Functions / Classes |
|---|---|
| **Regression** | `OLSRegression`, `RidgeRegression`, `LassoRegression` |
| **Preprocessing** | `fill_missing_mean`, `fill_missing_mean_df`, `normalize_features`, `add_intercept`, `detect_outliers_zscore` |
| **Feature Selection** | `forward_selection`, `backward_elimination` |
| **Diagnostics** | `normality_test` (Shapiro-Wilk), `vif` (Variance Inflation Factor), `heteroscedasticity` |
| **Visualization** | `plot_actual_vs_predicted`, `residual_plot` |
| **Prediction** | `predict_values` |

## Installation

### From PyPI (once published)
```bash
pip install myml-regression
```

### From source
```bash
git clone https://github.com/yourusername/myml-regression.git
cd myml-regression
pip install .
```

### From local wheel
```bash
pip install dist/myml_regression-0.1.0-py3-none-any.whl
```

## Quick Start

```python
import numpy as np
from myml.preprocessing import add_intercept, normalize_features, fill_missing_mean
from myml.regression import OLSRegression, RidgeRegression, LassoRegression
from myml.feature_selection import forward_selection
from myml.diagnostics import normality_test, vif

# --- Sample data ---
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
y = np.array([1.5, 3.5, 5.5, 7.5])

# Preprocessing
X_norm = normalize_features(X)
X_int  = add_intercept(X_norm)

# OLS Regression
ols = OLSRegression()
ols.fit(X_int, y)
print("OLS predictions:", ols.predict(X_int))

# Ridge Regression
ridge = RidgeRegression(alpha=0.5)
ridge.fit(X_int, y)
print("Ridge predictions:", ridge.predict(X_int))

# Lasso Regression
lasso = LassoRegression(alpha=0.01, iterations=1000)
lasso.fit(X_int, y)
print("Lasso predictions:", lasso.predict(X_int))

# Diagnostics
residuals = y - ols.predict(X_int)
stat, p = normality_test(residuals)
print(f"Shapiro-Wilk p-value: {p:.4f}")
vif_scores = vif(X_int)
print("VIF scores:", vif_scores)
```

## Dependencies

- `numpy >= 1.21`
- `pandas >= 1.3`
- `matplotlib >= 3.4`
- `scipy >= 1.7`

## Publishing to PyPI

```bash
# Install build tools
pip install build twine

# Build distributions
python -m build

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Project Structure

```
myml_package/
├── myml/
│   ├── __init__.py
│   ├── regression.py        # OLS, Ridge, Lasso
│   ├── preprocessing.py     # Missing values, normalization, outliers
│   ├── feature_selection.py # Forward selection, backward elimination
│   ├── diagnostics.py       # Shapiro-Wilk, VIF, heteroscedasticity
│   ├── visualization.py     # Actual vs Predicted, Residual plot
│   └── prediction.py        # predict_values helper
├── tests/
│   └── test_myml.py
├── pyproject.toml
├── setup.py
├── setup.cfg
├── MANIFEST.in
├── LICENSE
└── README.md
```

## License

MIT
=======
# ml-math
>>>>>>> ee03b2abad22dafef3a8f8aefe3a821e230c4ab5
