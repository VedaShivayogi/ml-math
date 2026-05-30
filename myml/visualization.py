import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.show()

def residual_plot(residuals):
    plt.plot(residuals)
    plt.title("Residual Plot")
    plt.show()
