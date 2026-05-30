import numpy as np

def forward_selection(X, y):
    selected = []
    remaining = list(range(X.shape[1]))
    while remaining:
        scores = []
        for feature in remaining:
            temp = selected + [feature]
            X_temp = X[:, temp]
            beta = np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ y
            pred = X_temp @ beta
            mse = np.mean((y - pred) ** 2)
            scores.append((mse, feature))
        scores.sort()
        best_feature = scores[0][1]
        selected.append(best_feature)
        remaining.remove(best_feature)
    return selected

def backward_elimination(X):
    return list(range(X.shape[1]))
