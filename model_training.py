"""Feature engineering, chronological evaluation, and model comparison."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "trend", "day_of_week", "month", "day_of_year",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14"
]

def make_features(series: pd.Series) -> pd.DataFrame:
    """Create features using only observations available before each target date."""
    frame = pd.DataFrame(index=series.index)
    frame["trend"] = np.arange(len(series))
    frame["day_of_week"] = series.index.dayofweek
    frame["month"] = series.index.month
    frame["day_of_year"] = series.index.dayofyear
    frame["lag_1"] = series.shift(1)
    frame["lag_7"] = series.shift(7)
    frame["lag_14"] = series.shift(14)
    frame["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    frame["rolling_mean_14"] = series.shift(1).rolling(14).mean()
    return frame

def evaluate_models(series: pd.Series):
    features = make_features(series)
    data = features.copy()
    data["target"] = series
    data = data.dropna()

    split_position = int(len(data) * 0.80)
    train = data.iloc[:split_position]
    test = data.iloc[split_position:]

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=350, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pd.Series(pred, index=test.index, name=name)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        }

    # Seasonal-naive baseline: predict today's sales using sales from 7 days ago.
    baseline_pred = test["lag_7"].to_numpy()
    predictions["Seasonal Naive (7-day)"] = pd.Series(
        baseline_pred, index=test.index, name="Seasonal Naive (7-day)"
    )
    results["Seasonal Naive (7-day)"] = {
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "R2": r2_score(y_test, baseline_pred)
    }

    metrics = pd.DataFrame(results).T.sort_values("MAE")
    best_name = metrics.index[0]
    best_model = models[best_name] if best_name in models else models["Random Forest"]
    # The best model is refit on all available supervised observations for future forecasting.
    best_model.fit(data[FEATURES], data["target"])
    return metrics, predictions, y_test, best_name, best_model, len(series)

def recursive_forecast(series: pd.Series, model, horizon: int = 30) -> pd.DataFrame:
    history = series.copy()
    future_dates = pd.date_range(
        history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecast_values = []

    for date in future_dates:
        values = {
            "trend": len(history),
            "day_of_week": date.dayofweek,
            "month": date.month,
            "day_of_year": date.dayofyear,
            "lag_1": history.iloc[-1],
            "lag_7": history.iloc[-7],
            "lag_14": history.iloc[-14],
            "rolling_mean_7": history.iloc[-7:].mean(),
            "rolling_mean_14": history.iloc[-14:].mean()
        }
        row = pd.DataFrame([values], columns=FEATURES)
        prediction = float(model.predict(row)[0])
        prediction = max(0, prediction)
        forecast_values.append(prediction)
        history.loc[date] = prediction

    return pd.DataFrame({"Date": future_dates, "Predicted_Sales": forecast_values})
