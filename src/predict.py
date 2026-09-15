from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.all_models import load_daily_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"


def predict_all_models():

    daily = load_daily_data()

    log_series = np.log1p(
        daily
    )

    next_date = (
        daily.index[-1]
        + pd.Timedelta(days=1)
    )

    # ========================================================
    # ARIMA
    # ========================================================

    arima = joblib.load(
        MODEL_DIR
        / "arima_model.joblib"
    )

    arima_log = arima.forecast(
        steps=1
    ).iloc[0]

    arima_prediction = np.expm1(
        arima_log
    )

    # ========================================================
    # SARIMA
    # ========================================================

    sarima = joblib.load(
        MODEL_DIR
        / "sarima_model.joblib"
    )

    sarima_log = sarima.forecast(
        steps=1
    ).iloc[0]

    sarima_prediction = np.expm1(
        sarima_log
    )

    # ========================================================
    # PROPHET
    # ========================================================

    prophet = joblib.load(
        MODEL_DIR
        / "prophet_model.joblib"
    )

    future = prophet.make_future_dataframe(
        periods=1,
        freq="D",
    )

    future["is_weekend"] = (
        future["ds"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    prophet_forecast = prophet.predict(
        future
    )

    prophet_prediction = (
        prophet_forecast["yhat"]
        .iloc[-1]
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    from xgboost import XGBRegressor

    xgb = XGBRegressor()
    xgb.load_model(
        str(MODEL_DIR / "xgboost_model.json")
    )

    future_features = {}

    for lag in range(1, 15):

        future_features[
            f"lag_{lag}"
        ] = log_series.iloc[-lag]

    future_features[
        "rolling_mean_3"
    ] = log_series.iloc[-3:].mean()

    future_features[
        "rolling_mean_7"
    ] = log_series.iloc[-7:].mean()

    future_features[
        "rolling_std_3"
    ] = log_series.iloc[-3:].std()

    future_features[
        "rolling_std_7"
    ] = log_series.iloc[-7:].std()

    future_features[
        "dayofweek"
    ] = next_date.dayofweek

    future_features[
        "month"
    ] = next_date.month

    future_features[
        "is_weekend"
    ] = int(
        next_date.dayofweek in [5, 6]
    )

    xgb_input = pd.DataFrame(
        [future_features],
        index=[next_date],
    )

    xgb_log = xgb.predict(
        xgb_input
    )[0]

    xgb_prediction = np.expm1(
        xgb_log
    )

    return {
        "date": next_date,
        "ARIMA": float(
            arima_prediction
        ),
        "SARIMA": float(
            sarima_prediction
        ),
        "Prophet": float(
            prophet_prediction
        ),
        "XGBoost": float(
            xgb_prediction
        ),
    }