import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from utils import load_data, FEATURES


def train_model():
    df = load_data()

    X = df[FEATURES]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, "aqi_model.pkl")

    print("Model trained successfully!")
    print("Mean Absolute Error:", round(mae, 2))
    print("R2 Score:", round(r2, 2))


if __name__ == "__main__":
    train_model()