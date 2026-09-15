# TRAFFIC INTELLIGENCE
# Premium Streamlit Traffic Forecasting Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

from src.predict import predict_all_models


# PAGE CONFIG

st.set_page_config(
    page_title="Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# PATHS

ROOT_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = ROOT_DIR / "assets"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

METRICS_FILE = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_FILE = ARTIFACTS_DIR / "test_predictions.csv"
DAILY_TRAFFIC_FILE = ARTIFACTS_DIR / "daily_traffic.csv"

# Support both names currently possible in your project
BACKGROUND_IMAGE = ASSETS_DIR / "traffic_background.png"

if not BACKGROUND_IMAGE.exists():
    BACKGROUND_IMAGE = ASSETS_DIR / "traffic_background.png.png"


# HTML RENDER HELPER
#
# IMPORTANT:
# We use st.html() instead of st.markdown() for HTML.
# This prevents Streamlit from displaying HTML as code.

def render_html(content):
    st.html(content)


# BACKGROUND IMAGE

def get_background_base64():

    if not BACKGROUND_IMAGE.exists():
        return ""

    try:
        with open(BACKGROUND_IMAGE, "rb") as image_file:
            return base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    except Exception:
        return ""


background_base64 = get_background_base64()


# GLOBAL CSS

if background_base64:

    background_rule = f"""
        background-image:
            linear-gradient(
                rgba(2, 8, 20, 0.82),
                rgba(2, 8, 20, 0.94)
            ),
            url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """

else:

    background_rule = """
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 200, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 80%,
                rgba(100, 60, 255, 0.12),
                transparent 30%
            ),
            #020814;
    """


render_html(
    f"""
    <style>

    /* ======================================================
       MAIN APPLICATION
       ====================================================== */

    .stApp {{
        {background_rule}

        color: white;
    }}


    /* ======================================================
       REMOVE DEFAULT STREAMLIT ELEMENTS
       ====================================================== */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent !important;
    }}


    /* ======================================================
       MAIN CONTAINER
       ====================================================== */

    .block-container {{
        max-width: 1450px;

        padding-top: 2rem;
        padding-bottom: 4rem;
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {{
        background:
            linear-gradient(
                180deg,
                rgba(1, 12, 27, 0.98),
                rgba(1, 6, 17, 0.99)
            );

        border-right:
            1px solid rgba(255,255,255,0.08);
    }}


    /* ======================================================
       SIDEBAR BRAND
       ====================================================== */

    .sidebar-brand {{
        text-align: center;

        padding:
            15px 5px 20px 5px;
    }}

    .sidebar-logo {{
        font-size: 46px;

        line-height: 1;

        margin-bottom: 10px;

        filter:
            drop-shadow(
                0 0 14px
                rgba(80,220,255,0.45)
            );
    }}

    .sidebar-title {{
        color: white;

        font-size: 20px;

        font-weight: 850;

        letter-spacing: -0.5px;
    }}

    .sidebar-subtitle {{
        color:
            rgba(255,255,255,0.36);

        font-size: 9px;

        letter-spacing: 2px;

        margin-top: 5px;
    }}


    /* ======================================================
       HERO
       ====================================================== */

    .hero {{
        position: relative;

        overflow: hidden;

        padding: 48px;

        margin-bottom: 32px;

        border-radius: 28px;

        background:
            linear-gradient(
                135deg,
                rgba(5,27,51,0.82),
                rgba(3,11,29,0.72)
            );

        border:
            1px solid
            rgba(85,220,255,0.18);

        box-shadow:
            0 30px 90px
            rgba(0,0,0,0.42);

        backdrop-filter: blur(20px);
    }}

    .hero::before {{
        content: "";

        position: absolute;

        width: 420px;
        height: 420px;

        right: -170px;
        top: -230px;

        border-radius: 50%;

        background:
            rgba(0,200,255,0.16);

        filter: blur(80px);

        animation:
            floatingGlow 5s
            ease-in-out
            infinite alternate;
    }}

    .hero::after {{
        content: "";

        position: absolute;

        width: 300px;
        height: 300px;

        left: -180px;
        bottom: -230px;

        border-radius: 50%;

        background:
            rgba(100,60,255,0.12);

        filter: blur(80px);

        animation:
            floatingGlow 7s
            ease-in-out
            infinite alternate-reverse;
    }}

    @keyframes floatingGlow {{

        0% {{
            transform: scale(0.85);
            opacity: 0.35;
        }}

        100% {{
            transform: scale(1.15);
            opacity: 0.9;
        }}

    }}

    .hero-content {{
        position: relative;

        z-index: 2;
    }}

    .hero-eyebrow {{
        color: #5cddff;

        font-size: 11px;

        font-weight: 850;

        letter-spacing: 3px;

        margin-bottom: 14px;
    }}

    .hero-title {{
        color: white;

        font-size: 56px;

        font-weight: 900;

        letter-spacing: -2.5px;

        line-height: 1;

        margin: 0;
    }}

    .hero-subtitle {{
        max-width: 790px;

        color:
            rgba(255,255,255,0.58);

        font-size: 15px;

        line-height: 1.7;

        margin-top: 20px;
    }}

    .online-badge {{
        display: inline-flex;

        align-items: center;

        gap: 9px;

        margin-top: 22px;

        padding:
            8px 14px;

        border-radius: 999px;

        color: #61f3bd;

        background:
            rgba(0,230,160,0.07);

        border:
            1px solid
            rgba(0,240,180,0.20);

        font-size: 10px;

        font-weight: 850;

        letter-spacing: 1.2px;
    }}

    .online-dot {{
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #45f3b3;

        box-shadow:
            0 0 10px
            #45f3b3;

        animation:
            statusPulse 1.8s
            infinite;
    }}

    @keyframes statusPulse {{

        0% {{
            box-shadow:
                0 0 0 0
                rgba(69,243,179,0.5);
        }}

        70% {{
            box-shadow:
                0 0 0 9px
                rgba(69,243,179,0);
        }}

        100% {{
            box-shadow:
                0 0 0 0
                rgba(69,243,179,0);
        }}

    }}


    /* ======================================================
       SECTION
       ====================================================== */

    .section-title {{
        color: white;

        font-size: 26px;

        font-weight: 850;

        margin-top: 30px;

        margin-bottom: 4px;
    }}

    .section-description {{
        color:
            rgba(255,255,255,0.38);

        font-size: 12px;

        margin-bottom: 20px;
    }}


    /* ======================================================
       GLASS CARD
       ====================================================== */

    .glass-card {{
        background:
            linear-gradient(
                145deg,
                rgba(7,28,51,0.82),
                rgba(2,12,28,0.74)
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        border-radius: 20px;

        padding: 23px;

        backdrop-filter: blur(16px);

        box-shadow:
            0 15px 45px
            rgba(0,0,0,0.20);

        transition:
            transform 0.25s ease,
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }}

    .glass-card:hover {{
        transform:
            translateY(-5px);

        border-color:
            rgba(80,220,255,0.25);

        box-shadow:
            0 25px 65px
            rgba(0,0,0,0.34);
    }}


    /* ======================================================
       MODEL CARD
       ====================================================== */

    .model-card {{
        min-height: 155px;

        padding: 22px;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(8,31,55,0.85),
                rgba(3,13,29,0.75)
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        backdrop-filter: blur(15px);

        transition:
            all 0.3s ease;
    }}

    .model-card:hover {{
        transform:
            translateY(-7px)
            scale(1.015);

        border-color:
            rgba(75,220,255,0.35);

        box-shadow:
            0 20px 55px
            rgba(0,0,0,0.35);
    }}

    .model-name {{
        color:
            rgba(255,255,255,0.48);

        font-size: 10px;

        font-weight: 850;

        letter-spacing: 1.7px;

        text-transform: uppercase;
    }}

    .model-value {{
        color: white;

        font-size: 32px;

        font-weight: 900;

        margin-top: 13px;
    }}

    .model-description {{
        color:
            rgba(255,255,255,0.36);

        font-size: 10px;

        line-height: 1.5;

        margin-top: 6px;
    }}


    /* ======================================================
       BEST MODEL
       ====================================================== */

    .best-model {{
        position: relative;

        overflow: hidden;

        padding: 24px 28px;

        margin:
            25px 0 30px 0;

        border-radius: 21px;

        background:
            linear-gradient(
                135deg,
                rgba(0,210,160,0.12),
                rgba(0,130,255,0.08)
            );

        border:
            1px solid
            rgba(0,240,190,0.22);

        backdrop-filter: blur(15px);
    }}

    .best-model::after {{
        content: "";

        position: absolute;

        width: 160px;
        height: 160px;

        right: -80px;
        top: -80px;

        border-radius: 50%;

        background:
            rgba(0,240,180,0.10);

        filter: blur(25px);
    }}

    .best-label {{
        color: #5ef3bd;

        font-size: 10px;

        font-weight: 900;

        letter-spacing: 2px;
    }}

    .best-name {{
        color: white;

        font-size: 29px;

        font-weight: 900;

        margin-top: 5px;
    }}

    .best-description {{
        color:
            rgba(255,255,255,0.43);

        font-size: 12px;

        margin-top: 5px;
    }}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {{
        min-height: 48px;

        border-radius: 14px;

        color: white;

        background:
            linear-gradient(
                135deg,
                rgba(0,180,255,0.20),
                rgba(105,65,255,0.20)
            );

        border:
            1px solid
            rgba(75,215,255,0.28);

        font-weight: 850;

        transition:
            all 0.25s ease;
    }}

    .stButton > button:hover {{
        transform:
            translateY(-3px);

        border-color:
            rgba(80,225,255,0.60);

        box-shadow:
            0 12px 35px
            rgba(0,170,255,0.18);
    }}


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {{
        padding: 21px;

        border-radius: 19px;

        background:
            linear-gradient(
                145deg,
                rgba(8,29,51,0.82),
                rgba(3,13,28,0.74)
            );

        border:
            1px solid
            rgba(255,255,255,0.08);

        backdrop-filter:
            blur(14px);

        transition:
            transform 0.25s ease,
            border-color 0.25s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        transform:
            translateY(-4px);

        border-color:
            rgba(80,220,255,0.28);
    }}

    div[data-testid="stMetricLabel"] {{
        color:
            rgba(255,255,255,0.43)
            !important;

        font-size: 11px
            !important;

        font-weight: 750
            !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: white
            !important;

        font-weight: 900
            !important;
    }}


    /* ======================================================
       SELECTBOX
       ====================================================== */

    div[data-baseweb="select"] > div {{
        background:
            rgba(4,17,33,0.88)
            !important;

        border:
            1px solid
            rgba(255,255,255,0.10)
            !important;

        border-radius:
            12px
            !important;
    }}


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {{
        color:
            rgba(255,255,255,0.43);

        font-weight: 750;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #5edcff;
    }}


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {{
        border-radius: 15px;

        overflow: hidden;

        border:
            1px solid
            rgba(255,255,255,0.07);
    }}


    /* ======================================================
       INFO / SUCCESS / ERROR
       ====================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 14px;
    }}


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {{
        text-align: center;

        color:
            rgba(255,255,255,0.25);

        font-size: 10px;

        margin-top: 65px;

        padding-top: 25px;

        border-top:
            1px solid
            rgba(255,255,255,0.06);

        line-height: 1.8;
    }}

    </style>
    """
)


# DATA LOADING

@st.cache_data
def load_metrics():

    if not METRICS_FILE.exists():
        return {}

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


@st.cache_data
def load_predictions():

    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    try:

        return pd.read_csv(
            PREDICTIONS_FILE
        )

    except Exception:

        return pd.DataFrame()


@st.cache_data
def load_daily_data():

    if not DAILY_TRAFFIC_FILE.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            DAILY_TRAFFIC_FILE
        )

        for column in [
            "date_time",
            "date",
            "ds"
        ]:

            if column in df.columns:

                df[column] = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                break

        return df

    except Exception:

        return pd.DataFrame()


# HELPERS

def get_metric(
    metrics,
    model,
    metric
):

    try:

        value = metrics[
            model
        ][
            metric
        ]

        if value is None:
            return None

        return float(value)

    except Exception:

        return None


def format_number(value):

    if value is None:
        return "—"

    try:

        return f"{float(value):,.0f}"

    except Exception:

        return "—"


def get_date_column(df):

    for column in [
        "date_time",
        "date",
        "ds"
    ]:

        if column in df.columns:
            return column

    return None


def get_actual_column(df):

    for column in [
        "actual",
        "traffic_volume",
        "y",
        "Actual"
    ]:

        if column in df.columns:
            return column

    return None


# LOAD DATA

metrics = load_metrics()

test_predictions = load_predictions()

daily_data = load_daily_data()


MODELS = [
    "ARIMA",
    "SARIMA",
    "Prophet",
    "XGBoost",
]


available_models = [
    model
    for model in MODELS
    if model in metrics
]


# BEST MODEL

best_model = None
best_mae = None

for model in available_models:

    mae = get_metric(
        metrics,
        model,
        "MAE"
    )

    if mae is None:
        continue

    if (
        best_mae is None
        or mae < best_mae
    ):

        best_mae = mae
        best_model = model


# SIDEBAR

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🚦
            </div>

            <div class="sidebar-title">
                Traffic Intelligence
            </div>

            <div class="sidebar-subtitle">
                AI FORECASTING PLATFORM
            </div>

        </div>
        """
    )

    st.divider()

    st.markdown(
        "### ⚙️ Dashboard"
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Forecast",
            "Model Evaluation",
            "Diagnostics",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        "### 🧠 Models"
    )

    for model in MODELS:

        render_html(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:9px;
                margin:11px 0;
                color:rgba(255,255,255,0.48);
                font-size:12px;
            ">

                <span style="
                    width:7px;
                    height:7px;
                    border-radius:50%;
                    background:#55dcff;
                    box-shadow:0 0 9px #55dcff;
                    flex-shrink:0;
                "></span>

                {model}

            </div>
            """
        )

    st.divider()

    st.caption(
        "Multi-Model Time-Series Forecasting"
    )


# HERO

render_html(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="hero-eyebrow">
                INTELLIGENT MOBILITY ANALYTICS
            </div>

            <div class="hero-title">
                Traffic Intelligence
            </div>

            <div class="hero-subtitle">
                Predict traffic demand using advanced time-series
                and machine-learning models. Compare forecasts,
                evaluate model performance and understand traffic
                patterns through interactive analytics.
            </div>

            <div class="online-badge">

                <span class="online-dot"></span>

                FORECASTING SYSTEM ONLINE

            </div>

        </div>

    </div>
    """
)


# OVERVIEW

if page == "Overview":

    render_html(
        """
        <div class="section-title">
            📡 System Overview
        </div>

        <div class="section-description">
            A unified view of the traffic forecasting pipeline.
        </div>
        """
    )


    # TOP METRICS

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "🤖 Models",
            len(available_models)
        )


    with c2:

        st.metric(
            "📉 Best MAE",
            format_number(best_mae)
        )


    with c3:

        best_rmse = None

        if best_model:

            best_rmse = get_metric(
                metrics,
                best_model,
                "RMSE"
            )

        st.metric(
            "📊 Best RMSE",
            format_number(best_rmse)
        )


    with c4:

        historical_days = 0

        if not daily_data.empty:

            date_column = get_date_column(
                daily_data
            )

            if date_column:

                historical_days = (
                    daily_data[
                        date_column
                    ].dropna().nunique()
                )

        st.metric(
            "📅 Historical Days",
            f"{historical_days:,}"
        )


    # BEST MODEL

    if best_model:

        render_html(
            f"""
            <div class="best-model">

                <div class="best-label">
                    🏆 RECOMMENDED MODEL
                </div>

                <div class="best-name">
                    {best_model}
                </div>

                <div class="best-description">
                    Lowest MAE on the evaluation period
                    &nbsp;•&nbsp;
                    MAE: {best_mae:,.2f}
                </div>

            </div>
            """
        )


    # TRAFFIC HISTORY

    if not daily_data.empty:

        date_column = get_date_column(
            daily_data
        )

        if (
            date_column
            and
            "traffic_volume"
            in daily_data.columns
        ):

            has_interpolated_flag = (
                "is_interpolated" in daily_data.columns
            )

            render_html(
                """
                <div class="section-title">
                    📈 Traffic Through Time
                </div>

                <div class="section-description">
                    Historical daily traffic volume."""
                + (
                    " The shaded band marks a long stretch of "
                    "missing sensor data that was linearly "
                    "interpolated, not observed."
                    if has_interpolated_flag
                    and daily_data["is_interpolated"].sum() > 0
                    else ""
                )
                + """
                </div>
                """
            )


            fig = go.Figure()


            fig.add_trace(
                go.Scatter(
                    x=daily_data[
                        date_column
                    ],

                    y=daily_data[
                        "traffic_volume"
                    ],

                    mode="lines",

                    name="Traffic Volume",

                    line=dict(
                        color="#5bdcff",
                        width=2.2,
                    ),

                    fill="tozeroy",

                    fillcolor=
                        "rgba(91,220,255,0.07)",

                    hovertemplate=
                        "<b>%{x|%d %b %Y}</b>"
                        "<br>"
                        "Traffic: %{y:,.0f}"
                        "<extra></extra>",
                )
            )


            # Shade any interpolated (non-observed) stretches
            # instead of letting them look like real history.
            if (
                has_interpolated_flag
                and daily_data["is_interpolated"].sum() > 0
            ):

                flags = daily_data["is_interpolated"].astype(int).values
                dates = daily_data[date_column].values

                group_id = (
                    pd.Series(flags)
                    .diff()
                    .fillna(1)
                    .ne(0)
                    .cumsum()
                )

                span_df = pd.DataFrame(
                    {
                        "flag": flags,
                        "date": dates,
                        "group": group_id,
                    }
                )

                for _, span in span_df[
                    span_df["flag"] == 1
                ].groupby("group"):

                    fig.add_vrect(
                        x0=span["date"].min(),
                        x1=span["date"].max(),
                        fillcolor="rgba(255,180,80,0.15)",
                        line_width=0,
                    )


            fig.update_layout(
                height=440,

                margin=dict(
                    l=15,
                    r=15,
                    t=15,
                    b=15,
                ),

                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                font=dict(
                    color=
                        "rgba(255,255,255,0.65)"
                ),

                xaxis=dict(
                    showgrid=False,

                    zeroline=False,
                ),

                yaxis=dict(
                    showgrid=True,

                    gridcolor=
                        "rgba(255,255,255,0.05)",

                    zeroline=False,
                ),

                hovermode="x unified",
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


    # MODEL PERFORMANCE CARDS

    render_html(
        """
        <div class="section-title">
            🧠 Model Performance
        </div>

        <div class="section-description">
            Comparison of the four forecasting approaches.
        </div>
        """
    )


    if available_models:

        cols = st.columns(
            len(available_models)
        )


        descriptions = {

            "ARIMA":
                "Autoregressive integrated forecasting",

            "SARIMA":
                "Seasonal time-series forecasting",

            "Prophet":
                "Trend and seasonality forecasting",

            "XGBoost":
                "Gradient boosting with lag features",
        }


        for index, model in enumerate(
            available_models
        ):

            mae = get_metric(
                metrics,
                model,
                "MAE"
            )

            rmse = get_metric(
                metrics,
                model,
                "RMSE"
            )


            with cols[index]:

                render_html(
                    f"""
                    <div class="model-card">

                        <div class="model-name">
                            {model}
                        </div>

                        <div class="model-value">
                            {format_number(mae)}
                        </div>

                        <div class="model-description">
                            MAE
                            <br>
                            RMSE:
                            {format_number(rmse)}
                            <br>
                            {descriptions[model]}
                        </div>

                    </div>
                    """
                )


# FORECAST

elif page == "Forecast":

    render_html(
        """
        <div class="section-title">
            🔮 Traffic Forecast
        </div>

        <div class="section-description">
            Generate a next-day traffic prediction using every
            trained model.
        </div>
        """
    )


    left, right = st.columns(
        [3, 1]
    )


    with left:

        st.info(
            "ARIMA • SARIMA • Prophet • XGBoost "
            "will all generate an independent forecast."
        )


    with right:

        predict_clicked = st.button(
            "🚀 Predict Next Day",
            width="stretch",
        )


    # RUN PREDICTION

    if predict_clicked:

        with st.spinner(
            "Running all forecasting models..."
        ):

            try:

                predictions = (
                    predict_all_models()
                )

                st.session_state[
                    "predictions"
                ] = predictions

                st.success(
                    "All forecasts generated successfully."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


    # RESULTS

    if "predictions" in st.session_state:

        predictions = (
            st.session_state[
                "predictions"
            ]
        )


        render_html(
            """
            <div class="section-title">
                📊 Model Predictions
            </div>

            <div class="section-description">
                Estimated traffic volume for the next day.
            </div>
            """
        )


        forecast_values = {}


        descriptions = {

            "ARIMA":
                "Autoregressive forecasting",

            "SARIMA":
                "Seasonal forecasting",

            "Prophet":
                "Trend + seasonality",

            "XGBoost":
                "Gradient boosting",
        }


        cols = st.columns(4)


        for index, model in enumerate(
            MODELS
        ):

            value = predictions.get(
                model
            )


            # Support different prediction
            # dictionary formats.

            if isinstance(
                value,
                dict
            ):

                value = value.get(
                    "prediction",

                    value.get(
                        "forecast",
                        None
                    )
                )


            try:

                if value is not None:

                    value = float(value)

                    forecast_values[
                        model
                    ] = value

            except Exception:

                value = None


            with cols[index]:

                render_html(
                    f"""
                    <div class="model-card">

                        <div class="model-name">
                            {model}
                        </div>

                        <div class="model-value">
                            {format_number(value)}
                        </div>

                        <div class="model-description">
                            {descriptions[model]}
                        </div>

                    </div>
                    """
                )


        # FORECAST COMPARISON

        if forecast_values:

            render_html(
                """
                <div class="section-title">
                    📈 Forecast Comparison
                </div>

                <div class="section-description">
                    How the four models differ in their next-day
                    traffic estimate.
                </div>
                """
            )


            fig = go.Figure()


            fig.add_trace(
                go.Bar(

                    x=list(
                        forecast_values.keys()
                    ),

                    y=list(
                        forecast_values.values()
                    ),

                    text=[
                        format_number(value)

                        for value
                        in forecast_values.values()
                    ],

                    textposition="outside",

                    marker=dict(
                        color=[
                            "#45d8ff",
                            "#6d7cff",
                            "#b86cff",
                            "#48e0a4",
                        ]
                    ),

                    hovertemplate=
                        "<b>%{x}</b>"
                        "<br>"
                        "Prediction: %{y:,.0f}"
                        "<extra></extra>",
                )
            )


            fig.update_layout(
                height=440,

                margin=dict(
                    l=15,
                    r=15,
                    t=40,
                    b=15,
                ),

                paper_bgcolor=
                    "rgba(0,0,0,0)",

                plot_bgcolor=
                    "rgba(0,0,0,0)",

                font=dict(
                    color=
                        "rgba(255,255,255,0.65)"
                ),

                xaxis=dict(
                    showgrid=False
                ),

                yaxis=dict(
                    showgrid=True,

                    gridcolor=
                        "rgba(255,255,255,0.05)"
                ),

                showlegend=False,
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


            # FORECAST SUMMARY

            average_prediction = np.mean(
                list(
                    forecast_values.values()
                )
            )


            highest_model = max(
                forecast_values,
                key=forecast_values.get
            )


            lowest_model = min(
                forecast_values,
                key=forecast_values.get
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Average Forecast",
                    format_number(
                        average_prediction
                    )
                )


            with c2:

                st.metric(
                    "Highest Forecast",
                    highest_model
                )


            with c3:

                st.metric(
                    "Lowest Forecast",
                    lowest_model
                )


# MODEL EVALUATION

elif page == "Model Evaluation":

    render_html(
        """
        <div class="section-title">
            🧠 Model Evaluation
        </div>

        <div class="section-description">
            Performance on the held-out evaluation period.
        </div>
        """
    )


    # CREATE METRICS TABLE

    rows = []


    for model in MODELS:

        if model not in metrics:
            continue


        rows.append(
            {
                "Model": model,

                "MAE":
                    get_metric(
                        metrics,
                        model,
                        "MAE"
                    ),

                "RMSE":
                    get_metric(
                        metrics,
                        model,
                        "RMSE"
                    ),

                "MAPE":
                    get_metric(
                        metrics,
                        model,
                        "MAPE"
                    ),
            }
        )


    if rows:

        metrics_df = pd.DataFrame(
            rows
        )


        display_df = metrics_df.copy()


        display_df["MAE"] = (
            display_df["MAE"]
            .apply(
                lambda x:
                    f"{x:,.2f}"
                    if pd.notna(x)
                    else "—"
            )
        )


        display_df["RMSE"] = (
            display_df["RMSE"]
            .apply(
                lambda x:
                    f"{x:,.2f}"
                    if pd.notna(x)
                    else "—"
            )
        )


        display_df["MAPE"] = (
            display_df["MAPE"]
            .apply(
                lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "—"
            )
        )


        st.dataframe(
            display_df,

            width="stretch",

            hide_index=True,
        )


        # BAR CHART

        render_html(
            """
            <div class="section-title">
                🏆 Performance Comparison
            </div>

            <div class="section-description">
                Lower MAE and RMSE indicate better predictive
                performance.
            </div>
            """
        )


        fig = go.Figure()


        fig.add_trace(
            go.Bar(

                name="MAE",

                x=metrics_df[
                    "Model"
                ],

                y=metrics_df[
                    "MAE"
                ],

                marker_color="#52dfff",
            )
        )


        fig.add_trace(
            go.Bar(

                name="RMSE",

                x=metrics_df[
                    "Model"
                ],

                y=metrics_df[
                    "RMSE"
                ],

                marker_color="#7b6cff",
            )
        )


        fig.update_layout(
            barmode="group",

            height=450,

            margin=dict(
                l=15,
                r=15,
                t=25,
                b=15,
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color=
                    "rgba(255,255,255,0.65)"
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,

                gridcolor=
                    "rgba(255,255,255,0.05)"
            ),
        )


        st.plotly_chart(
            fig,
            width="stretch",
        )


    # ACTUAL VS PREDICTED

    if not test_predictions.empty:

        render_html(
            """
            <div class="section-title">
                📉 Actual vs Predicted
            </div>

            <div class="section-description">
                Compare model predictions with actual observed
                traffic during the evaluation period.
            </div>
            """
        )


        prediction_models = [

            model

            for model in MODELS

            if model
            in test_predictions.columns

        ]


        if prediction_models:

            selected_model = st.selectbox(
                "Select model",
                prediction_models,
                key="evaluation_model",
            )


            date_column = get_date_column(
                test_predictions
            )


            actual_column = get_actual_column(
                test_predictions
            )


            if (
                date_column
                and actual_column
            ):

                dates = pd.to_datetime(
                    test_predictions[
                        date_column
                    ],
                    errors="coerce",
                )


                actual = pd.to_numeric(
                    test_predictions[
                        actual_column
                    ],
                    errors="coerce",
                )


                predicted = pd.to_numeric(
                    test_predictions[
                        selected_model
                    ],
                    errors="coerce",
                )


                fig = go.Figure()


                fig.add_trace(
                    go.Scatter(

                        x=dates,

                        y=actual,

                        mode="lines",

                        name="Actual",

                        line=dict(
                            color="#ffffff",
                            width=2.5,
                        ),
                    )
                )


                fig.add_trace(
                    go.Scatter(

                        x=dates,

                        y=predicted,

                        mode="lines",

                        name=selected_model,

                        line=dict(
                            color="#54ddff",
                            width=2,
                            dash="dash",
                        ),
                    )
                )


                fig.update_layout(
                    height=470,

                    hovermode="x unified",

                    margin=dict(
                        l=15,
                        r=15,
                        t=20,
                        b=15,
                    ),

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    font=dict(
                        color=
                            "rgba(255,255,255,0.65)"
                    ),

                    xaxis=dict(
                        showgrid=False
                    ),

                    yaxis=dict(
                        showgrid=True,

                        gridcolor=
                            "rgba(255,255,255,0.05)"
                    ),
                )


                st.plotly_chart(
                    fig,
                    width="stretch",
                )


# DIAGNOSTICS

elif page == "Diagnostics":

    render_html(
        """
        <div class="section-title">
            🔬 Forecast Diagnostics
        </div>

        <div class="section-description">
            Inspect prediction errors and residual behaviour.
        </div>
        """
    )


    if not test_predictions.empty:

        prediction_models = [

            model

            for model in MODELS

            if model
            in test_predictions.columns

        ]


        if prediction_models:

            selected_model = st.selectbox(
                "Diagnostic model",
                prediction_models,
                key="diagnostic_model",
            )


            date_column = get_date_column(
                test_predictions
            )


            actual_column = get_actual_column(
                test_predictions
            )


            if (
                date_column
                and actual_column
            ):

                dates = pd.to_datetime(
                    test_predictions[
                        date_column
                    ],
                    errors="coerce",
                )


                actual = pd.to_numeric(
                    test_predictions[
                        actual_column
                    ],
                    errors="coerce",
                )


                predicted = pd.to_numeric(
                    test_predictions[
                        selected_model
                    ],
                    errors="coerce",
                )


                residuals = (
                    actual
                    - predicted
                )


                # ERROR SUMMARY

                c1, c2, c3, c4 = st.columns(4)


                with c1:

                    st.metric(
                        "Mean Error",
                        f"{residuals.mean():,.2f}"
                    )


                with c2:

                    st.metric(
                        "Std Error",
                        f"{residuals.std():,.2f}"
                    )


                with c3:

                    st.metric(
                        "Maximum Error",
                        f"{residuals.max():,.2f}"
                    )


                with c4:

                    st.metric(
                        "Minimum Error",
                        f"{residuals.min():,.2f}"
                    )


                # RESIDUAL GRAPH

                render_html(
                    """
                    <div class="section-title">
                        📉 Residual Behaviour
                    </div>

                    <div class="section-description">
                        Difference between actual and predicted
                        traffic.
                    </div>
                    """
                )


                fig = go.Figure()


                fig.add_trace(
                    go.Scatter(

                        x=dates,

                        y=residuals,

                        mode="lines+markers",

                        name="Residual",

                        line=dict(
                            color="#ff6b9d",
                            width=1.7,
                        ),

                        marker=dict(
                            size=4,
                        ),
                    )
                )


                fig.add_hline(
                    y=0,

                    line_dash="dash",

                    line_color=
                        "rgba(255,255,255,0.35)",
                )


                fig.update_layout(
                    height=410,

                    margin=dict(
                        l=15,
                        r=15,
                        t=20,
                        b=15,
                    ),

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    font=dict(
                        color=
                            "rgba(255,255,255,0.65)"
                    ),

                    xaxis=dict(
                        showgrid=False
                    ),

                    yaxis=dict(
                        showgrid=True,

                        gridcolor=
                            "rgba(255,255,255,0.05)"
                    ),
                )


                st.plotly_chart(
                    fig,
                    width="stretch",
                )


                # ERROR DISTRIBUTION

                render_html(
                    """
                    <div class="section-title">
                        📊 Error Distribution
                    </div>

                    <div class="section-description">
                        Distribution of prediction errors.
                    </div>
                    """
                )


                error_df = pd.DataFrame(
                    {
                        "Residual":
                            residuals.dropna()
                    }
                )


                fig = px.histogram(
                    error_df,

                    x="Residual",

                    nbins=30,

                    marginal="box",
                )


                fig.update_layout(
                    height=410,

                    margin=dict(
                        l=15,
                        r=15,
                        t=20,
                        b=15,
                    ),

                    paper_bgcolor=
                        "rgba(0,0,0,0)",

                    plot_bgcolor=
                        "rgba(0,0,0,0)",

                    font=dict(
                        color=
                            "rgba(255,255,255,0.65)"
                    ),

                    xaxis=dict(
                        showgrid=False
                    ),

                    yaxis=dict(
                        showgrid=True,

                        gridcolor=
                            "rgba(255,255,255,0.05)"
                    ),
                )


                st.plotly_chart(
                    fig,
                    width="stretch",
                )


                # ERROR TABLE

                with st.expander(
                    "🔍 View detailed prediction errors"
                ):

                    error_table = pd.DataFrame(
                        {
                            "Date":
                                dates,

                            "Actual":
                                actual,

                            "Predicted":
                                predicted,

                            "Error":
                                residuals,

                            "Absolute Error":
                                residuals.abs(),
                        }
                    )


                    st.dataframe(
                        error_table,

                        width="stretch",

                        hide_index=True,
                    )


    else:

        st.warning(
            "Evaluation prediction data is not available."
        )


    # PIPELINE

    render_html(
        """
        <div class="section-title">
            ⚙️ Forecasting Pipeline
        </div>

        <div class="section-description">
            End-to-end architecture of the deployed system.
        </div>
        """
    )


    pipeline = pd.DataFrame(
        {
            "Stage": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
            ],

            "Component": [
                "Raw Data",
                "Preprocessing",
                "Feature Engineering",
                "ML Models",
                "Evaluation",
                "Forecast",
            ],

            "Purpose": [
                "Traffic dataset",
                "Clean + resample",
                "Lag + calendar features",
                "ARIMA / SARIMA / Prophet / XGBoost",
                "MAE / RMSE / MAPE",
                "Next-day prediction",
            ],
        }
    )


    st.dataframe(
        pipeline,

        width="stretch",

        hide_index=True,
    )


# FOOTER

render_html(
    """
    <div class="footer">

        🚦 <b>Traffic Intelligence</b>

        &nbsp; • &nbsp;

        Multi-Model Time-Series Forecasting

        <br>

        ARIMA
        &nbsp;•&nbsp;
        SARIMA
        &nbsp;•&nbsp;
        Prophet
        &nbsp;•&nbsp;
        XGBoost

        <br>

        Intelligent Mobility Analytics

    </div>
    """
)
