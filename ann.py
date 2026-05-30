# ==========================================================
# FULL ANN (SINGLE NEURON) - MAGICBRICKS DATASET
# USING COMPLETE DATASET (NO SHORTCUTS)
# ==========================================================

import numpy as np
import pandas as pd

# ==========================================================
# LOAD FULL DATASET
# ==========================================================

data = pd.read_csv("magicbricks_hyderabad.csv")

# ==========================================================
# SHOW FULL DATASET (WARNING: LARGE)
# ==========================================================

print("\n================ FULL DATASET ================\n")
print(data.to_string(index=False))

print("\n================ DATASET INFO ================\n")
print(data.info())

print("\n================ DATASET SHAPE ================\n")
print(data.shape)

# ==========================================================
# HANDLE MISSING VALUES (FULL DATASET SAFE)
# ==========================================================

cols = ['bhk', 'area_sqft', 'price_per_sqft', 'price_lakhs']
data[cols] = data[cols].fillna(data[cols].mean())

# ==========================================================
# FEATURES & TARGET (FULL DATASET USED)
# ==========================================================

X = data[['bhk', 'area_sqft', 'price_per_sqft']].values
y = data['price_lakhs'].values

# ==========================================================
# NORMALIZATION (FULL DATASET BASED)
# ==========================================================

X_min = X.min(axis=0)
X_max = X.max(axis=0)

X = (X - X_min) / (X_max - X_min + 1e-8)

y_min = y.min()
y_max = y.max()

y = (y - y_min) / (y_max - y_min + 1e-8)

# ==========================================================
# ADD BIAS COLUMN
# ==========================================================

X = np.hstack((np.ones((X.shape[0], 1)), X))

# ==========================================================
# INITIAL WEIGHTS
# ==========================================================

np.random.seed(42)
old_weights = np.random.randn(X.shape[1]) * 0.1
weights = old_weights.copy()

print("\n================ OLD WEIGHTS ================\n")
print(old_weights)

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

learning_rate = 0.01
E = 1e-4
epochs = 50

mse_history = []

# ==========================================================
# TRAINING ON FULL DATASET
# ==========================================================

for epoch in range(epochs):

    total_error = 0

    for i in range(len(X)):   # FULL DATA USED HERE

        # forward pass
        f1_x = np.dot(X[i], weights)

        # error
        error = y[i] - f1_x

        # update
        weights += learning_rate * error * X[i]

        total_error += error ** 2

    mse = total_error / len(X)
    mse_history.append(mse)

    print(f"Epoch {epoch+1} | MSE = {mse:.6f}")

    if mse < E:
        print("\nSTOPPED EARLY: Error threshold reached")
        break

# ==========================================================
# FINAL WEIGHTS
# ==========================================================

new_weights = weights

print("\n================ NEW WEIGHTS ================\n")
print(new_weights)

# ==========================================================
# WEIGHT COMPARISON TABLE
# ==========================================================

print("\n================ WEIGHT COMPARISON TABLE ================\n")

weight_table = pd.DataFrame({
    "Weight Index": [f"w{i}" for i in range(len(old_weights))],
    "Old Weights": old_weights,
    "New Weights": new_weights,
    "Change (Δ)": new_weights - old_weights
})

print(weight_table)

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(x_input):

    x_input = np.array(x_input, dtype=float)

    # scale using FULL dataset values
    x_input = (x_input - X_min) / (X_max - X_min + 1e-8)

    x_input = np.insert(x_input, 0, 1)

    pred = np.dot(x_input, new_weights)

    pred = pred * (y_max - y_min) + y_min

    return pred

# ==========================================================
# CUSTOM PREDICTION
# ==========================================================

print("\n================ CUSTOM PREDICTION ================\n")

new_house = [3, 1800, 0.05]

result = predict(new_house)

print("Predicted Price (Lakhs):", result)

# ==========================================================
# MSE TABLE
# ==========================================================

print("\n================ MSE TABLE ================\n")

mse_table = pd.DataFrame({
    "Epoch": list(range(1, len(mse_history) + 1)),
    "MSE": mse_history
})

print(mse_table)

# ==========================================================
# END
# ==========================================================

print("\nPROGRAM COMPLETED SUCCESSFULLY")