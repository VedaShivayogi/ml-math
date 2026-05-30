import numpy as np

class OLSRegression:
    def fit(self, X, y):
        XTX = np.dot(X.T, X)
        XTX_INV = np.linalg.inv(XTX)
        XTY = np.dot(X.T, y)
        self.beta = np.dot(XTX_INV, XTY)

    def predict(self, X):
        return np.dot(X, self.beta)

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_features = X.shape[1]
        identity = np.eye(n_features)
        self.beta = np.linalg.inv(X.T @ X + self.alpha * identity) @ X.T @ y

    def predict(self, X):
        return X @ self.beta

class LassoRegression:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations

    def fit(self, X, y):
        rows, cols = X.shape
        self.beta = np.zeros(cols)
        learning_rate = 0.001
        for _ in range(self.iterations):
            y_pred = X @ self.beta
            error = y_pred - y
            gradient = (1 / rows) * (X.T @ error)
            self.beta -= learning_rate * gradient
            self.beta -= self.alpha * np.sign(self.beta)

    def predict(self, X):
        return X @ self.beta
