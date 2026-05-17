import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_merge_data():
    # Load data
    gdp_df = pd.read_csv(os.path.join(BASE_DIR, "../../data/raw/gdp.csv"))
    co2_df = pd.read_csv(os.path.join(BASE_DIR, "../../data/raw/co2.csv"))

    # Filter CO2 data
    co2_df = co2_df[co2_df["year"] >= 1950]

    # Standardize column names
    gdp_df = gdp_df.rename(columns={
        "Entity": "country",
        "Year": "year"
    })

    # Merge datasets
    merged_df = pd.merge(
        gdp_df,
        co2_df,
        on=["country", "year"],
        how="inner"
    )

    # Clean missing values
    merged_df = merged_df.dropna(subset=["GDP", "trade_co2"])

    return merged_df


# Optional: allow running this file directly for testing
if __name__ == "__main__":
    df = load_and_merge_data()
    print(df.head())
    print(df.shape)