from src.data.build_dataset import load_and_merge_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def train_model():
    df = load_and_merge_data()

    df = df.dropna(subset=["GDP", "trade_co2"])

    X = df[["GDP"]]
    y = df["trade_co2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R² score:", r2_score(y_test, y_pred))


if __name__ == "__main__":
    train_model()