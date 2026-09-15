# 🚦 Traffic Intelligence

> **Traffic Data Analysis, Time-Series Modeling & Next-Day Result
> Showcase**

Traffic Intelligence is an end-to-end data analysis project built using
the **Metro Interstate Traffic Volume dataset**. It studies historical
traffic behavior, prepares the time-series data, analyzes temporal and
weather patterns, applies multiple predefined models, compares their
results, and presents the findings through an interactive Streamlit
dashboard.

------------------------------------------------------------------------

## 📌 Project Overview

The project follows a complete workflow:

``` text
Raw Data → Preparation → Analysis → Models → Evaluation → Comparison → Results → Streamlit
```

### What it does

-   Loads and examines historical traffic data.
-   Converts and organizes timestamp information.
-   Aggregates hourly traffic into a daily time series.
-   Checks missing values and time gaps.
-   Performs exploratory data analysis.
-   Studies weekly and monthly traffic patterns.
-   Examines weather and traffic relationships.
-   Applies ARIMA, SARIMA, Prophet and XGBoost.
-   Compares model performance using evaluation metrics.
-   Generates a next-day result from the tested models.
-   Provides residual and error diagnostics.
-   Presents the complete analysis through Streamlit.

------------------------------------------------------------------------

## 🎯 Objectives

1.  Understand historical traffic behavior.
2.  Identify temporal and seasonal patterns.
3.  Prepare traffic data for time-series analysis.
4.  Compare different statistical and machine-learning approaches.
5.  Evaluate model results using MAE, RMSE and MAPE.
6.  Analyze model errors and residuals.
7.  Build an interactive application for presenting the results.

------------------------------------------------------------------------

## 📊 Dataset

### Metro Interstate Traffic Volume Dataset

The project uses historical traffic and weather observations from
**Interstate 94 in Minnesota**.

  Attribute      Details
  -------------- ---------------------------------
  Dataset        Metro Interstate Traffic Volume
  Location       I-94, Minnesota, USA
  Period         2012--2018
  Frequency      Hourly
  Observations   48,204
  Target         `traffic_volume`

### Main Features

-   `holiday` --- Holiday information
-   `temp` --- Temperature
-   `rain_1h` --- Rainfall
-   `snow_1h` --- Snowfall
-   `clouds_all` --- Cloud coverage
-   `weather_main` --- Main weather category
-   `weather_description` --- Weather description
-   `date_time` --- Timestamp
-   `traffic_volume` --- Traffic volume

------------------------------------------------------------------------

## 🧹 Data Preparation

The raw dataset contains hourly observations.

Main preparation steps:

1.  Convert `date_time` to datetime.
2.  Sort observations chronologically.
3.  Check missing values and time gaps.
4.  Aggregate hourly traffic into a daily series.
5.  Prepare a continuous series for analysis.
6.  Apply transformations required by individual models.

### Missing Data

Missing daily values in the prepared series are filled using **linear
interpolation** to maintain a continuous analysis series.

> **Limitation:** the original dataset contains a substantial historical
> gap. Interpolating across a long gap can create artificial
> observations and should be handled more carefully in a production
> pipeline.

------------------------------------------------------------------------

# 📈 Exploratory Data Analysis

The project analyzes traffic from multiple perspectives.

### Traffic Over Time

Daily traffic is visualized to reduce high-frequency noise and make
longer-term behavior easier to understand.

### Weekly Pattern

Average traffic is compared across:

``` text
Monday → Tuesday → Wednesday → Thursday → Friday → Saturday → Sunday
```

### Monthly Pattern

Average traffic is analyzed across the months of the year to identify
broader seasonal behavior.

### Weather & Traffic

Traffic is compared with:

-   Temperature
-   Rain
-   Snow
-   Cloud cover

These visualizations help identify possible relationships between
weather conditions and traffic volume.

------------------------------------------------------------------------

# 🤖 Models Tested

## 1. ARIMA

**AutoRegressive Integrated Moving Average**

A classical statistical model for time-series data.

Configuration used:

``` text
ARIMA(2, 1, 2)
```

------------------------------------------------------------------------

## 2. SARIMA

**Seasonal AutoRegressive Integrated Moving Average**

Extends ARIMA by incorporating seasonal behavior.

Configuration:

``` text
ARIMA order:    (2, 1, 2)
Seasonal order: (1, 1, 1, 7)
```

The seasonal period of 7 represents weekly behavior in the daily traffic
series.

------------------------------------------------------------------------

## 3. Prophet

Prophet is used to model trend and recurring seasonal patterns.

The project uses yearly, weekly and monthly seasonality along with a
weekend-related feature.

------------------------------------------------------------------------

## 4. XGBoost

**Extreme Gradient Boosting**

XGBoost is used as a supervised machine-learning approach using
historical time-series features such as:

-   Lag features
-   Rolling statistics
-   Calendar features
-   Time-based information

------------------------------------------------------------------------

# 📏 Model Evaluation

The project uses standard evaluation metrics.

### MAE

Mean Absolute Error measures the average absolute difference between
actual and model-generated values.

**Lower is better.**

### RMSE

Root Mean Squared Error gives greater weight to larger errors.

**Lower is better.**

### MAPE

Mean Absolute Percentage Error expresses error as a percentage.

**Lower is better.**

------------------------------------------------------------------------

# 📊 Initial Experimental Results

  Model          MAE     RMSE    MAPE
  --------- -------- -------- -------
  ARIMA       428.34   533.89     ---
  SARIMA      186.57   312.04     ---
  Prophet     174.30   287.02   5.94%
  XGBoost     106.31   173.33     ---

> These are initial experimental results. The XGBoost feature-generation
> workflow requires stricter leakage-safe handling before these numbers
> should be treated as final production benchmarks.

------------------------------------------------------------------------

# 🔬 Diagnostics

The application includes:

### Actual vs Model Results

Compares observed traffic against each model's results on the held-out
test period.

### Residual Analysis

``` text
Residual = Actual Traffic − Model Result
```

Residual plots show where models differ from observed traffic.

### Error Distribution

A histogram is used to examine the distribution of model errors.

### Residual Autocorrelation

Residual autocorrelation is examined to identify remaining
time-dependent structure.

> Current SARIMA diagnostics show remaining residual autocorrelation, so
> the model should not be considered perfect.

------------------------------------------------------------------------

# 🖥️ Streamlit Dashboard

The Streamlit application provides five main pages.

### 🏠 Overview

-   Project introduction
-   Dataset summary
-   Objectives
-   Data used
-   Models tested
-   Project workflow

### 📊 Data Analysis

-   Dataset snapshot
-   Data quality
-   Missing-data handling
-   Traffic over time
-   Weekly pattern
-   Monthly pattern
-   Weather relationships

### 🤖 Model Analysis

-   Model descriptions
-   Performance table
-   MAE comparison
-   Best stored result

### 🔮 Next-Day Result

-   Individual model results
-   Combined result
-   Model comparison chart

The next-day value is presented as an **output of the analysis and
modeling workflow**, rather than defining the entire project as a
standalone forecasting product.

### 🔬 Diagnostics

-   Actual vs model results
-   Residual analysis
-   Error distribution
-   Interpretation

------------------------------------------------------------------------

# 🗂️ Project Structure

``` text
traffic-forecasting/
│
├── app/
│   └── app.py
│
├── assets/
│   └── traffic_background.png
│
├── artifacts/
│   ├── daily_traffic.csv
│   ├── metrics.json
│   └── test_predictions.csv
│
├── data/
│   └── raw/
│       └── Metro_Interstate_Traffic_Volume.csv
│
├── models/
│   ├── arima_model.joblib
│   ├── prophet_model.joblib
│   ├── sarima_model.joblib
│   └── xgboost_model.joblib
│
├── notebooks/
├── src/
│   ├── all_models.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── features.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── runtime.txt
```

------------------------------------------------------------------------

# 🛠️ Tech Stack

  Area               Technologies
  ------------------ -----------------------------
  Language           Python
  Data Analysis      Pandas, NumPy
  Machine Learning   Scikit-learn, XGBoost
  Time Series        Statsmodels, Prophet
  Visualization      Plotly, Matplotlib, Seaborn
  Application        Streamlit
  Model Storage      Joblib
  Development        VS Code, Git, GitHub

------------------------------------------------------------------------

# ⚙️ Installation

Clone the repository:

``` bash
git clone <your-repository-url>
cd traffic-forecasting
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Run the Project

### Start Streamlit

From the project root:

``` bash
python -m streamlit run app/app.py
```

### Train and evaluate models

``` bash
python -m src.all_models
```

This generates the model and evaluation artifacts used by the dashboard.

------------------------------------------------------------------------

# ☁️ Deployment

The application can be deployed using **Streamlit Community Cloud**.

Make sure the repository contains:

``` text
app/
src/
models/
artifacts/
assets/
requirements.txt
runtime.txt
```

The Python version used for deployment should be compatible with the
environment used to create serialized model artifacts.

------------------------------------------------------------------------

# ⚠️ Limitations & Future Improvements

### Large Historical Gap

A substantial gap exists in the original dataset. Future preprocessing
should distinguish between short gaps and long missing periods instead
of interpolating long gaps.


### Training Pipeline

The project can be improved by consolidating preprocessing, feature
engineering, training, evaluation and prediction into one consistent
pipeline.

### Testing

The `tests/` directory can be expanded to test:

-   Data loading
-   Preprocessing
-   Feature generation
-   Metrics
-   Prediction outputs
-   Model artifacts

### Additional Improvements

-   Hyperparameter optimization
-   Leakage-safe cross-validation
-   LSTM/GRU models
-   Better weather integration
-   Real-time traffic data
-   Automated retraining
-   Model monitoring

------------------------------------------------------------------------

# 📚 Learning Outcomes

This project provides practical experience with:

-   Time-series data
-   Data preprocessing
-   Missing-data handling
-   Exploratory data analysis
-   Feature engineering
-   Stationarity
-   ARIMA and SARIMA
-   Prophet
-   XGBoost
-   Model evaluation
-   Residual diagnostics
-   Model serialization
-   Streamlit
-   Git/GitHub
-   Deployment
-   Production-oriented project structure

------------------------------------------------------------------------

# 👨‍💻 Conclusion

**Traffic Intelligence** combines data analysis, time-series techniques,
machine learning, model comparison and interactive visualization into
one complete workflow.

The main focus is understanding the traffic data, analyzing its
behavior, applying multiple predefined models, comparing their results,
and presenting the findings through a usable Streamlit application.

``` text
Understand Data
      ↓
Prepare Data
      ↓
Analyze Patterns
      ↓
Apply Models
      ↓
Compare Results
      ↓
Diagnose Errors
      ↓
Show Next-Day Result
      ↓
Streamlit Dashboard
```

> **Traffic Intelligence turns a raw traffic dataset into an interactive
> analysis and model-comparison experience.**
