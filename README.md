# Traffic Intelligence — Metro Interstate Traffic Forecasting

Daily traffic-volume forecasting for the Minnesota I-94 corridor
(UCI "Metro Interstate Traffic Volume" dataset, 2012-2018, hourly,
resampled to daily). Four models are trained and compared: ARIMA,
SARIMA, Prophet, and XGBoost. A Streamlit dashboard visualizes
historical traffic, model comparisons, and a next-day forecast.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train / retrain all models

```bash
python -m src.all_models
```

This is the **only** training entry point - it regenerates:
- `models/*.json` / `models/*.joblib` - the four fitted models
- `artifacts/metrics.json` - MAE / RMSE / MAPE on a 30-day holdout
- `artifacts/test_predictions.csv` - per-day predictions vs. actuals on the holdout
- `artifacts/daily_traffic.csv` - the daily series used for training, with an
  `is_interpolated` column flagging days that were filled in rather than observed

## Evaluate the deployed model

```bash
python -m src.evaluation
```

Re-scores the currently saved XGBoost model using the same feature
pipeline it was trained with (`src.all_models.create_xgb_dataset`).

## Run the dashboard

```bash
streamlit run app/app.py
```

## Get a next-day forecast programmatically

```python
from src.predict import predict_all_models
predict_all_models()
```

## Project layout

```
src/
  data_loader.py   - raw CSV loading + validation checks
  all_models.py    - the single training pipeline for all 4 models (source of truth)
  predict.py       - next-day inference using the saved models
  evaluation.py    - re-scores the deployed XGBoost model
app/app.py         - Streamlit dashboard
models/            - saved model artifacts
artifacts/         - metrics, holdout predictions, daily series
data/raw/          - raw dataset
```

`src/all_models.py` is the single source of truth for feature
engineering and training. Don't add a second training script with
its own feature logic - a previous version of this project had one,
and it silently produced a model with a different feature scale than
`predict.py` was feeding it at inference time.

## Known data caveats

- The raw sensor data has real gaps, the largest being **2014-08-09
  to 2015-06-10 (~10 months)**. These are linearly interpolated so
  the models get a continuous daily series. That stretch is *not*
  observed data - it's flagged via `is_interpolated` in
  `artifacts/daily_traffic.csv` and shaded in the dashboard's
  history chart. Treat model behavior across that period with
  appropriate skepticism.
- `holiday` effects are modeled using the US holiday calendar in
  Prophet (this is Minnesota traffic data, confirmed by the raw
  `holiday` column values: Labor Day, MLK Day, Washington's
  Birthday, Minnesota State Fair, etc.).
- Reported holdout metrics reflect a single 30-day holdout at the
  end of the series, not cross-validated performance. Treat them as
  a rough guide, not a guarantee of production accuracy.
