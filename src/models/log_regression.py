from src.data.build_dataset import load_and_merge_data

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def run_log_regression():
    # Load data
    df = load_and_merge_data()

    # Remove missing values
    df = df.dropna(subset=["GDP", "trade_co2"])

    # Remove zero/negative values for log transform
    df = df[(df["GDP"] > 0) & (df["trade_co2"] > 0)]

    # Log transform
    X = np.log(df[["GDP"]])
    y = np.log(df["trade_co2"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)

    print("Log Regression R² score:", r2)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.scatter(X_test, y_test, label="Actual")
    plt.scatter(X_test, y_pred, label="Predicted")

    plt.xlabel("Log GDP")
    plt.ylabel("Log Trade CO2")
    plt.title("Log-Log Regression")
    plt.legend()
      
    plt.savefig("../../outputs/log_regression.png")
    
    
    plt.show()
    
    
    # Residuals
    residuals = y_test - y_pred
    
    # Residual plot
    plt.figure(figsize=(10, 6))

    plt.scatter(y_pred, residuals)

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    
    plt.savefig("../../outputs/log_residual.png")
    
    plt.show()
    
    
    




if __name__ == "__main__":
    run_log_regression()