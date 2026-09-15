from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Metro_Interstate_Traffic_Volume.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 30

ARIMA_ORDER = (2, 1, 2)

SARIMA_ORDER = (2, 1, 2)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)


# ============================================================
# DATA PREPARATION
# ============================================================

def load_daily_data(return_interpolated_flag=False):

    df = pd.read_csv(DATA_PATH)

    df["date_time"] = pd.to_datetime(
        df["date_time"]
    )

    df = df.set_index("date_time")

    df = df.sort_index()

    daily = (
        df["traffic_volume"]
        .resample("D")
        .mean()
    )

    # The raw sensor data has real gaps (the largest spans ~10
    # months, from 2014-08 to 2015-06). interpolate() fills those
    # with a straight line so downstream models get a continuous
    # series, but that stretch is NOT observed data. Track which
    # days were filled so it can be labeled rather than presented
    # as real history.
    was_missing = daily.isnull()

    daily = daily.interpolate()

    if return_interpolated_flag:
        return daily, was_missing

    return daily


# ============================================================
# XGBOOST FEATURES
# ============================================================

def create_xgb_dataset(daily):

    # The notebook applies log1p before
    # creating the ML features.
    df = pd.DataFrame(
        {
            "traffic_volume": np.log1p(daily)
        },
        index=daily.index,
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in range(1, 15):

        df[f"lag_{lag}"] = (
            df["traffic_volume"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    # NOTE: shift(1) first so the window for day T only ever
    # looks at days before T. Without this, rolling(3)/rolling(7)
    # include day T's own value, which leaks the target into the
    # feature and also doesn't match what predict.py can supply at
    # real inference time (tomorrow's value is never known).
    df["rolling_mean_3"] = (
        df["traffic_volume"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["rolling_mean_7"] = (
        df["traffic_volume"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_std_3"] = (
        df["traffic_volume"]
        .shift(1)
        .rolling(3)
        .std()
    )

    df["rolling_std_7"] = (
        df["traffic_volume"]
        .shift(1)
        .rolling(7)
        .std()
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df["dayofweek"] = (
        df.index.dayofweek
    )

    df["month"] = (
        df.index.month
    )

    df["is_weekend"] = (
        df["dayofweek"]
        .isin([5, 6])
        .astype(int)
    )

    df.dropna(
        inplace=True
    )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mape = (
        np.mean(
            np.abs(
                (actual - predicted)
                / actual
            )
        )
        * 100
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models():

    print("\nLoading data...")

    daily, was_missing = load_daily_data(
        return_interpolated_flag=True
    )

    print(
        f"Daily observations: {len(daily)}"
    )

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    train_raw = daily.iloc[:-TEST_SIZE]

    test_raw = daily.iloc[-TEST_SIZE:]

    train_log = np.log1p(
        train_raw
    )

    test_log = np.log1p(
        test_raw
    )

    actual_test = np.expm1(
        test_log
    )

    # ========================================================
    # 1. ARIMA
    # ========================================================

    print("\nTraining ARIMA...")

    arima_model = ARIMA(
        train_log,
        order=ARIMA_ORDER,
    )

    arima_fit = arima_model.fit()

    arima_pred_log = (
        arima_fit
        .forecast(
            steps=TEST_SIZE
        )
    )

    arima_pred = np.expm1(
        arima_pred_log
    )

    arima_metrics = calculate_metrics(
        actual_test,
        arima_pred,
    )

    print(
        "ARIMA:",
        arima_metrics,
    )

    # ========================================================
    # 2. SARIMA
    # ========================================================

    print("\nTraining SARIMA...")

    sarima_model = SARIMAX(
        train_log,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
    )

    sarima_fit = sarima_model.fit(
        method="powell",
        disp=False,
    )

    sarima_pred_log = (
        sarima_fit
        .forecast(
            steps=TEST_SIZE
        )
    )

    sarima_pred = np.expm1(
        sarima_pred_log
    )

    sarima_metrics = calculate_metrics(
        actual_test,
        sarima_pred,
    )

    print(
        "SARIMA:",
        sarima_metrics,
    )

    # ========================================================
    # 3. PROPHET
    # ========================================================

    print("\nTraining Prophet...")

    prophet_df = pd.DataFrame(
        {
            "ds": daily.index,
            "y": daily.values,
        }
    )

    prophet_df["is_weekend"] = (
        prophet_df["ds"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    prophet_train = (
        prophet_df.iloc[:-TEST_SIZE]
        .copy()
    )

    prophet_test = (
        prophet_df.iloc[-TEST_SIZE:]
        .copy()
    )

    from prophet.make_holidays import (
        make_holidays_df
    )

    holidays = make_holidays_df(
        year_list=[
            2012,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
        ],
        country="US",
    )

    prophet_model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.8,
        seasonality_prior_scale=20,
    )

    prophet_model.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5,
    )

    prophet_model.add_regressor(
        "is_weekend"
    )

    prophet_model.fit(
        prophet_train
    )

    future = prophet_model.make_future_dataframe(
        periods=TEST_SIZE,
        freq="D",
    )

    future["is_weekend"] = (
        future["ds"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    forecast = prophet_model.predict(
        future
    )

    prophet_pred = (
        forecast["yhat"]
        .tail(TEST_SIZE)
        .values
    )

    prophet_metrics = calculate_metrics(
        prophet_test["y"].values,
        prophet_pred,
    )

    print(
        "Prophet:",
        prophet_metrics,
    )

    # ========================================================
    # 4. XGBOOST
    # ========================================================

    print("\nTraining XGBoost...")

    xgb_df = create_xgb_dataset(
        daily
    )

    train_ml = xgb_df.iloc[:-TEST_SIZE]

    test_ml = xgb_df.iloc[-TEST_SIZE:]

    X_train = train_ml.drop(
        columns=["traffic_volume"]
    )

    y_train = train_ml[
        "traffic_volume"
    ]

    X_test = test_ml.drop(
        columns=["traffic_volume"]
    )

    y_test = test_ml[
        "traffic_volume"
    ]

    xgb_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train,
        y_train,
    )

    xgb_pred_log = (
        xgb_model.predict(X_test)
    )

    xgb_pred = np.expm1(
        xgb_pred_log
    )

    xgb_actual = np.expm1(
        y_test
    )

    xgb_metrics = calculate_metrics(
        xgb_actual,
        xgb_pred,
    )

    print(
        "XGBoost:",
        xgb_metrics,
    )

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions = pd.DataFrame(
        {
            "date": test_raw.index,
            "Actual": actual_test.values,
            "ARIMA": np.asarray(
                arima_pred
            ),
            "SARIMA": np.asarray(
                sarima_pred
            ),
            "Prophet": np.asarray(
                prophet_pred
            ),
            "XGBoost": np.asarray(
                xgb_pred
            ),
        }
    )

    predictions.to_csv(
        ARTIFACT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "ARIMA": arima_metrics,
        "SARIMA": sarima_metrics,
        "Prophet": prophet_metrics,
        "XGBoost": xgb_metrics,
    }

    with open(
        ARTIFACT_DIR / "metrics.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ========================================================
    # SAVE DAILY DATA
    # ========================================================

    daily_export = pd.DataFrame(
        {
            "traffic_volume": daily,
            "is_interpolated": was_missing.astype(int),
        }
    )

    daily_export.to_csv(
        ARTIFACT_DIR
        / "daily_traffic.csv"
    )

    # ========================================================
    # REFIT MODELS ON ALL DATA
    # ========================================================

    print(
        "\nRefitting models on complete dataset..."
    )

    # --------------------------------------------------------
    # ARIMA
    # --------------------------------------------------------

    final_arima = ARIMA(
        np.log1p(daily),
        order=ARIMA_ORDER,
    ).fit()

    # --------------------------------------------------------
    # SARIMA
    # --------------------------------------------------------

    final_sarima = SARIMAX(
        np.log1p(daily),
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
    ).fit(
        method="powell",
        disp=False,
    )

    # --------------------------------------------------------
    # Prophet
    # --------------------------------------------------------

    final_prophet_df = prophet_df.copy()

    final_prophet = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.8,
        seasonality_prior_scale=20,
    )

    final_prophet.add_seasonality(
        name="monthly",
        period=30.5,
        fourier_order=5,
    )

    final_prophet.add_regressor(
        "is_weekend"
    )

    final_prophet.fit(
        final_prophet_df
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    final_xgb_df = create_xgb_dataset(
        daily
    )

    final_X = final_xgb_df.drop(
        columns=["traffic_volume"]
    )

    final_y = final_xgb_df[
        "traffic_volume"
    ]

    final_xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
    )

    final_xgb.fit(
        final_X,
        final_y,
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    joblib.dump(
        final_arima,
        MODEL_DIR / "arima_model.joblib",
    )

    joblib.dump(
        final_sarima,
        MODEL_DIR / "sarima_model.joblib",
    )

    joblib.dump(
        final_prophet,
        MODEL_DIR / "prophet_model.joblib",
    )

    # Native XGBoost format instead of joblib/pickle: stable
    # across xgboost version upgrades (joblib.dump on the sklearn
    # wrapper pickles internal state that can break when the
    # library version changes on the serving machine).
    final_xgb.save_model(
        str(MODEL_DIR / "xgboost_model.json")
    )

    print("\n========================================")
    print("ALL MODELS SAVED")
    print("========================================")

    print(
        "ARIMA   → models/arima_model.joblib"
    )

    print(
        "SARIMA  → models/sarima_model.joblib"
    )

    print(
        "Prophet → models/prophet_model.joblib"
    )

    print(
        "XGBoost → models/xgboost_model.json"
    )

    print(
        "\nEvaluation artifacts saved to:"
    )

    print(
        "artifacts/metrics.json"
    )

    print(
        "artifacts/test_predictions.csv"
    )

    print(
        "artifacts/daily_traffic.csv"
    )


if __name__ == "__main__":
    train_all_models()