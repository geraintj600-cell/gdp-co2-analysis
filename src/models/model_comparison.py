from src.data.build_dataset import load_and_merge_data
from src.models.model_comparison_plot import plot_model_comparison
from src.models.residual_analysis import residual_analysis

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compare_models():

    df = load_and_merge_data()

    df = df.dropna(subset=["GDP", "trade_co2"])

    df = df[
        (df["GDP"] > 0) &
        (df["trade_co2"] > 0)
    ]

    # ---------- MODEL 1: GDP ----------

    X_gdp = df[["GDP"]]
    y = df["trade_co2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_gdp, y,
        test_size=0.2,
        random_state=42
    )

    gdp_model = LinearRegression()

    gdp_model.fit(X_train, y_train)

    gdp_pred = gdp_model.predict(X_test)

    gdp_r2 = r2_score(y_test, gdp_pred)

    # ---------- MODEL 2: GDP + YEAR ----------

    X_year = df[["GDP", "year"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_year, y,
        test_size=0.2,
        random_state=42
    )

    year_model = LinearRegression()

    year_model.fit(X_train, y_train)

    year_pred = year_model.predict(X_test)

    year_r2 = r2_score(y_test, year_pred)

    # ---------- MODEL 3: LOG MODEL ----------

    X_log = np.log(df[["GDP"]])

    y_log = np.log(df["trade_co2"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_log,
        y_log,
        test_size=0.2,
        random_state=42
    )

    log_model = LinearRegression()

    log_model.fit(X_train, y_train)

    log_pred = log_model.predict(X_test)

    log_r2 = r2_score(y_test, log_pred)
    
    # Results
    print("\nModel Comparison")
    print("----------------------------")
    print(f"GDP Model R²: {gdp_r2:.3f}")
    print(f"GDP + Year Model R²: {year_r2:.3f}")
    print(f"Log Regression R²: {log_r2:.3f}")

    # Diagnostics
    residual_analysis(y_test, log_pred)

    # Comparison plot
    plot_model_comparison(
        gdp_r2,
        year_r2,
        log_r2
        )


def main():

    compare_models()


if __name__ == "__main__":
    main()