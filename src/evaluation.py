from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from src.all_models import (
    load_daily_data,
    create_xgb_dataset,
    TEST_SIZE,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_model.json"
)


# ============================================================
# EVALUATION
#
# This reuses the exact same feature pipeline used to train the
# deployed model (src.all_models.create_xgb_dataset), so results
# here always reflect the model that is actually being served.
# Do NOT reintroduce a separate/parallel feature-engineering
# module for this - that mismatch previously caused reported
# metrics to silently diverge from the deployed model.
#
# CAVEAT: the deployed model is refit on the FULL dataset at the
# end of training (see all_models.train_all_models), so the last
# TEST_SIZE days scored here were part of its own training data -
# this is an in-sample sanity check, not a true holdout score.
# The honest out-of-sample estimate is in artifacts/metrics.json,
# produced from a model trained with those days excluded.
# ============================================================

def evaluate_model():

    daily = load_daily_data()

    xgb_df = create_xgb_dataset(daily)

    test_ml = xgb_df.iloc[-TEST_SIZE:]

    X_test = test_ml.drop(columns=["traffic_volume"])
    y_test_log = test_ml["traffic_volume"]

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    predictions_log = model.predict(X_test)

    # Both y_test and predictions are in log1p space here
    # (create_xgb_dataset log1p-transforms the target) - convert
    # both back to real traffic volume before scoring.
    y_test = np.expm1(y_test_log)
    predictions = np.expm1(predictions_log)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    mape = np.mean(
        np.abs((y_test - predictions) / y_test)
    ) * 100

    print("\n========================================")
    print("XGBoost Model Evaluation (in-sample check)")
    print("See artifacts/metrics.json for the true holdout score.")
    print("========================================")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }


if __name__ == "__main__":
    evaluate_model()
