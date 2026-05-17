from src.data.build_dataset import load_and_merge_data
import matplotlib.pyplot as plt


def get_top_emitters(df, n=10):
    return (
        df.groupby("country")["trade_co2"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )


def plot_country_trends(df, countries):
    plt.figure()

    for country in countries:
        temp = df[df["country"] == country].sort_values("year")
        plt.plot(temp["year"], temp["trade_co2"], label=country)

    plt.xlabel("Year")
    plt.ylabel("Trade CO2")
    plt.title("CO2 Trends by Country")
    plt.legend()
    plt.show()


def main():
    # Load data
    df = load_and_merge_data()

    # Get top emitters
    top_emitters = get_top_emitters(df, 5)

    print("Top Emitters:")
    print(top_emitters)
    print(df.head())


    # Plot trends
    top_countries = top_emitters.index
    plot_country_trends(df, top_countries)


if __name__ == "__main__":
    main()