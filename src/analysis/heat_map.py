from src.data.build_dataset import load_and_merge_data
import matplotlib.pyplot as plt
import numpy as np
import os


def heat_map(df):

    df = df.copy()

    # Feature engineering
    df["co2_per_gdp"] = df["trade_co2"] / df["GDP"]

    corr_data = df[[
        "GDP",
        "trade_co2",
        "year",
        "co2_per_gdp"
    ]]

    corr_matrix = corr_data.corr()

    print(corr_matrix)

    plt.figure(figsize=(8, 6))

    plt.imshow(corr_matrix, cmap="coolwarm")
    plt.colorbar()

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=45
    )

    plt.yticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns
    )

    plt.title("Correlation Heatmap")

    # Annotate cells
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(
                j,
                i,
                round(corr_matrix.iloc[i, j], 2),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Create outputs folder if missing
    output_dir = os.path.join(
            BASE_DIR,
            "../../outputs"
            )
    
    os.makedirs(output_dir, exist_ok=True)

    # Build full save path
    output_path = os.path.join(
    output_dir,
    "heat_map.png"
    )

    plt.savefig(output_path)
    
    print(f"Saved heatmap to: {output_path}")

    plt.show()


def plot_log_relationship(df):

    df = df.copy()

    # Clean data
    df = df.dropna(subset=["GDP", "trade_co2"])

    df = df[
        (df["GDP"] > 0) &
        (df["trade_co2"] > 0)
    ]

    # Log transform
    log_gdp = np.log(df["GDP"])
    log_co2 = np.log(df["trade_co2"])

    plt.figure(figsize=(8, 6))

    # Scatter plot
    plt.scatter(
        log_gdp,
        log_co2,
        alpha=0.5
    )

    # Regression line
    z = np.polyfit(log_gdp, log_co2, 1)
    p = np.poly1d(z)

    plt.plot(
        log_gdp,
        p(log_gdp)
    )

    plt.xlabel("Log GDP")
    plt.ylabel("Log Trade CO₂")
    plt.title("Log-Log GDP vs Trade CO₂ Relationship")

    plt.tight_layout()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(
        BASE_DIR,
        "../../outputs/log_relationship.png"
    )

    plt.savefig(output_path)

    print(f"Saved log plot to: {output_path}")

    plt.show()


def main():

    df = load_and_merge_data()

    heat_map(df)

    plot_log_relationship(df)


if __name__ == "__main__":
    main()