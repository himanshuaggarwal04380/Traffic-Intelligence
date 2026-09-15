import pandas as pd


DATA_PATH = "data/raw/Metro_Interstate_Traffic_Volume.csv"

REQUIRED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


def load_data(path=DATA_PATH):
    """Load the raw traffic dataset."""
    return pd.read_csv(path)


def validate_columns(df):
    """Check whether all required columns are present."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def validate_data(df):
    """Validate important data-quality conditions."""

    # 1. Check required columns
    validate_columns(df)

    # 2. Check target column
    if df["traffic_volume"].isnull().any():
        raise ValueError("traffic_volume contains missing values.")

    # 3. Traffic volume cannot be negative
    if (df["traffic_volume"] < 0).any():
        raise ValueError("traffic_volume contains negative values.")

    # 4. Check datetime values
    parsed_dates = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    if parsed_dates.isnull().any():
        raise ValueError("date_time contains invalid values.")

    # 5. Check numeric columns
    numeric_columns = [
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "traffic_volume",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{column} must be numeric."
            )

    # 6. Check duplicate complete rows
    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate complete rows: {duplicate_rows}")

    print("All data validation checks passed.")

    return True


if __name__ == "__main__":

    df = load_data()

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    validate_data(df)