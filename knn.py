# ==========================================================
# KNN FROM SCRATCH USING MAGICBRICKS DATASET (FULL DATA)
# ==========================================================

import numpy as np
import pandas as pd
from collections import Counter

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("magicbricks_hyderabad.csv")

# ==========================================================
# SHOW FULL DATASET (SAFE VERSION)
# ==========================================================

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print("\n================ FULL DATASET ================\n")

print(data.to_string())   # FULL DATA PRINT

# ==========================================================
# DATASET INFO
# ==========================================================

print("\n================ DATASET INFO ================\n")
print(data.info())

print("\n================ COLUMNS ================\n")
print(data.columns)

print("\n================ STATISTICS ================\n")
print(data.describe())

# ==========================================================
# MISSING VALUE HANDLING
# ==========================================================

numeric_columns = ['bhk', 'area_sqft', 'price_per_sqft', 'price_lakhs']
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = data[['bhk', 'area_sqft', 'price_per_sqft']].values
y = data['price_lakhs'].values

# ==========================================================
# NORMALIZATION (IMPORTANT FOR KNN)
# ==========================================================

def normalize(X):
    X = np.array(X, dtype=float)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals + 1e-8)

X = normalize(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# ==========================================================
# EUCLIDEAN DISTANCE
# ==========================================================

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# ==========================================================
# KNN MODEL
# ==========================================================

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])

        neighbors = distances[:self.k]

        values = [val for _, val in neighbors]

        # regression output (mean of neighbors)
        return np.mean(values)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# ==========================================================
# FIND BEST K
# ==========================================================

def find_best_k(X_train, y_train, X_test, y_test, max_k=11):
    best_k = 1
    best_error = float("inf")

    print("\n================ K VALUES TEST ================\n")

    for k in range(1, max_k + 1, 2):
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = np.mean((y_test - preds) ** 2)

        print(f"K={k} -> MSE: {mse:.4f}")

        if mse < best_error:
            best_error = mse
            best_k = k

    print("\nBEST K =", best_k)
    return best_k

# ==========================================================
# TRAIN BEST MODEL
# ==========================================================

best_k = find_best_k(X_train, y_train, X_test, y_test)

print("\nFINAL MODEL WITH BEST K =", best_k)

model = KNNClassifier(k=best_k)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

mse = np.mean((y_test - predictions) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - predictions))

print("\n================ RESULTS ================\n")
print("MSE :", mse)
print("RMSE:", rmse)
print("MAE :", mae)

# ==========================================================
# CUSTOM PREDICTION
# ==========================================================

print("\n================ CUSTOM PREDICTION ================\n")

new_house = np.array([[3, 1800, 0.05]])

min_vals = X.min(axis=0)
max_vals = X.max(axis=0)

new_house = (new_house - min_vals) / (max_vals - min_vals + 1e-8)

result = model.predict(new_house)

print("Predicted Price (Lakhs):", result[0])

# ==========================================================
# END
# ==========================================================

print("\nPROGRAM COMPLETED SUCCESSFULLY")