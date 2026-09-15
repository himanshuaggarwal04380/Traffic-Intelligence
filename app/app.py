from pathlib import Path
import base64
import mimetypes
import json
import traceback
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Traffic Intelligence", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# Resolve the project root reliably, even when Streamlit is launched from the app folder.
HERE = Path(__file__).resolve().parent
PROJECT_CANDIDATES = [HERE, HERE.parent, Path.cwd(), Path.cwd().parent]
ROOT = next((p for p in PROJECT_CANDIDATES if (p / "data" / "raw").exists()), HERE.parent)

# Make the project root importable even when Streamlit is launched from app/.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DAILY_PATH = ROOT / "artifacts" / "daily_traffic.csv"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
PREDICTIONS_PATH = ROOT / "artifacts" / "test_predictions.csv"

# The app accepts the old filename too. It also searches the assets folder so
# the background keeps working whether the app is started from the project
# root or with `streamlit run app.py` from inside the app folder.
def find_background():
    assets = ROOT / "assets"
    if not assets.exists():
        return None

    preferred = [
        "traffic_background.png",
        "traffic_background.png.png",
        "traffic_background.jpg",
        "traffic_background.jpeg",
        "traffic_background.webp",
    ]
    for name in preferred:
        candidate = assets / name
        if candidate.is_file():
            return candidate

    for candidate in sorted(assets.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return candidate
    return None

BG_PATH = find_background()

st.html("""
<style>
.stApp{background:#0b0d10;color:#f5f7fa}.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#0b0d10!important;border-right:1px solid rgba(255,255,255,.07)!important}
[data-testid="stSidebar"]>div:first-child{padding:.75rem .65rem}
[data-testid="stSidebarContent"]{padding:0!important}
.sidebar-brand{padding:1.15rem .55rem 2.1rem}
.sidebar-logo{
    display:block;
    color:#fff;
    font-size:1.8rem;
    font-weight:800;
    letter-spacing:-.02em;
    line-height:1.15
}
.sidebar-logo-mark{display:none}
.sidebar-sub{display:none}
[data-testid="stSidebar"] .stRadio>label{display:none}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:3px}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label{position:relative;border-radius:10px;padding:.72rem .75rem;color:#aeb5c0;min-height:43px;display:flex;align-items:center;font-size:.88rem;font-weight:520;transition:color .15s ease,opacity .15s ease;background:transparent!important;box-shadow:none!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label>div:first-child{display:none}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label:hover{background:transparent!important;color:#fff}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label:has(input:checked){background:transparent!important;color:#fff;font-weight:700;box-shadow:none!important}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label:has(input:checked):before{content:"";position:absolute;left:0;top:50%;transform:translateY(-50%);width:3px;height:20px;border-radius:3px;background:#fff}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] p{margin:0!important;padding-left:0!important}

.stButton>button{border-radius:9px;background:#171b21;border:1px solid rgba(255,255,255,.12);color:#fff;min-height:42px}
[data-testid="stMetric"]{background:#11151a;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:1rem}
.section-title{font-size:1.2rem;font-weight:650;margin:1.6rem 0 .25rem}.muted{color:#929ba8;font-size:.88rem;line-height:1.5}
.hero{position:relative;min-height:355px;display:flex;align-items:flex-end;overflow:hidden;border:1px solid rgba(255,255,255,.08);border-radius:18px;margin-bottom:1.5rem;background:#090b0e}
.hero-bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;display:block;z-index:0}
.hero-overlay{position:absolute;inset:0;z-index:1;background:linear-gradient(90deg,rgba(7,9,12,.60),rgba(7,9,12,.35) 45%,rgba(7,9,12,.06)),linear-gradient(0deg,rgba(7,9,12,.50),transparent 70%)}
.hero-content{position:relative;z-index:2;padding:2.2rem;max-width:700px}.eyebrow{color:#a9b4c3;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700;margin-bottom:.7rem}
.hero h1{font-size:clamp(2.3rem,5vw,4.2rem);line-height:1;letter-spacing:-.045em;margin:0 0 1rem;color:#fff}.hero p{color:#c0c7d1;font-size:1rem;line-height:1.6;margin:0}
.pill{display:inline-flex;align-items:center;gap:.45rem;margin-top:1.2rem;padding:.45rem .7rem;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.06);border-radius:999px;color:#d7dde5;font-size:.78rem}.dot{width:7px;height:7px;border-radius:50%;background:#6ee7a0;box-shadow:0 0 12px rgba(110,231,160,.6)}
.card{background:#11151a;border:1px solid rgba(255,255,255,.07);border-radius:13px;padding:1.1rem;height:100%}.card .title{font-weight:650;color:#fff;margin:.35rem 0}.card .text{color:#919ba8;font-size:.86rem;line-height:1.45}
.result{background:linear-gradient(135deg,#141920,#101318);border:1px solid rgba(255,255,255,.09);border-radius:16px;padding:1.5rem;text-align:center}.result-label{color:#8f99a7;font-size:.75rem;text-transform:uppercase;letter-spacing:.12em}.result-number{color:#fff;font-size:3rem;font-weight:750;letter-spacing:-.04em;margin:.35rem 0}.result-sub{color:#9ca6b3;font-size:.85rem}
.flow{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}.flow span{padding:.5rem .75rem;border-radius:8px;background:#14181e;border:1px solid rgba(255,255,255,.07);font-size:.82rem}.flow b{color:#687382;font-weight:400}
#MainMenu,footer{visibility:hidden}header{background:transparent!important}
</style>
""")


def card(icon, title, text):
    st.html(f'<div class="card"><div>{icon}</div><div class="title">{title}</div><div class="text">{text}</div></div>')


def model_card(name, kind, text):
    st.html(f'<div class="card"><div class="title">{name}</div><div style="color:#8f99a7;font-size:.8rem">{kind}</div><div class="text" style="margin-top:.65rem">{text}</div></div>')


@st.cache_data(show_spinner=False)
def raw_data():
    df = pd.read_csv(DATA_PATH)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def daily_data():
    if DAILY_PATH.exists():
        df = pd.read_csv(DAILY_PATH)
        dc = next((c for c in df.columns if c.lower() in {"date", "date_time", "ds"}), None)
        if dc:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            if dc != "date": df = df.rename(columns={dc: "date"})
        return df
    raw = raw_data()
    return (raw.dropna(subset=["date_time"]).set_index("date_time")["traffic_volume"].resample("D").mean().rename("traffic_volume").reset_index().rename(columns={"date_time":"date"}))


@st.cache_data(show_spinner=False)
def metrics():
    if not METRICS_PATH.exists(): return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception: return {}


@st.cache_data(show_spinner=False)
def test_predictions():
    return pd.read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else None


def col(df, names):
    if df is None: return None
    lookup = {c.lower(): c for c in df.columns}
    return next((lookup[n.lower()] for n in names if n.lower() in lookup), None)


def score(metrics_obj, model, metric):
    for key in (model, model.lower(), model.upper()):
        obj = metrics_obj.get(key, {}) if isinstance(metrics_obj, dict) else {}
        if isinstance(obj, dict):
            for k in (metric, metric.lower(), metric.upper()):
                if k in obj:
                    try: return float(obj[k])
                    except Exception: pass
    return None


def normalize_predictions(result):
    names = ["ARIMA", "SARIMA", "Prophet", "XGBoost"]
    out = {}
    if isinstance(result, dict):
        for name in names:
            value = None
            for key in (name, name.lower(), name.upper()):
                if key in result:
                    value = result[key]; break
            if isinstance(value, dict):
                for k in ("prediction", "forecast", "value", "yhat"):
                    if k in value: value = value[k]; break
            try:
                if hasattr(value, "iloc"): value = value.iloc[-1]
                elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)): value = value[-1]
                out[name] = float(value)
            except Exception: pass
    elif isinstance(result, pd.DataFrame):
        mc, vc = col(result,["model"]), col(result,["prediction","forecast","value","yhat"])
        if mc and vc:
            for _, r in result.iterrows():
                for name in names:
                    if str(r[mc]).lower() == name.lower():
                        try: out[name] = float(r[vc])
                        except Exception: pass
    return out


# Sidebar — minimal navigation.
with st.sidebar:
    st.html("""
        <div class="sidebar-brand">
            <div class="sidebar-logo">Traffic<br>Intelligence</div>
        </div>
    """)

    page = st.radio(
        "Navigation",
        ["Overview", "Data Analysis", "Model Analysis", "Next-Day Result", "Diagnostics"],
        label_visibility="collapsed",
    )


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    # Use a real <img> element for the local traffic image. This is more
    # reliable than a CSS background in Streamlit and also works on Cloud.
    hero_image = ""
    if BG_PATH and BG_PATH.exists():
        encoded = base64.b64encode(BG_PATH.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(BG_PATH.name)[0] or "image/png"
        hero_image = (
            f'<img class="hero-bg" src="data:{mime};base64,{encoded}" '
            f'alt="Traffic background" />'
        )

    st.html(
        f'''<div class="hero">
            {hero_image}
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="eyebrow">Traffic data analysis</div>
                <h1>Traffic<br>Intelligence</h1>
                <p>Explore historical traffic behavior, analyze patterns, test different models, and view the resulting next-day estimate.</p>
                <div class="pill"><span class="dot"></span>Analysis system online</div>
            </div>
        </div>'''
    )

    st.markdown("### About the Data")
    st.markdown('<div class="muted">Historical traffic and weather observations from the Metro Interstate Highway dataset.</div>', unsafe_allow_html=True)
    try:
        raw = raw_data(); start, end = raw.date_time.min(), raw.date_time.max()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Observations", f"{len(raw):,}"); c2.metric("Features", len(raw.columns)); c3.metric("Start", start.strftime("%b %Y")); c4.metric("End", end.strftime("%b %Y"))
    except Exception:
        pass

    st.markdown("### What We Do")
    cs=st.columns(4)
    for c, item in zip(cs, [("01","Prepare","Clean and organize the raw data."),("02","Analyze","Study traffic trends and patterns."),("03","Compare","Test four predefined models."),("04","Estimate","Show the resulting next-day value.")]):
        with c: card(*item)

    st.markdown("### Data We Use")

    cs = st.columns(5)

    for c, item in zip(cs, [
        ("", "Traffic", "Traffic volume recorded over time."),
        ("", "Temperature", "Temperature at each observation."),
        ("", "Weather", "Rain, snow and cloud information."),
        ("", "Date & Time", "Used to study time-based patterns."),
        ("", "Holiday", "Holiday information in the dataset.")
    ]):
        with c:
            card(*item)
        
        
    st.markdown("### Models Tested")
    cs=st.columns(4)
    for c,item in zip(cs,[("ARIMA","Statistical","Captures time-based behavior."),("SARIMA","Seasonal","Captures time and seasonal behavior."),("Prophet","Trend & seasonality","Models trend and recurring patterns."),("XGBoost","Machine learning","Learns from historical features.")]):
        with c: model_card(*item)

    st.markdown("### Project Flow")
    st.html('<div class="flow"><span>Data</span><b>→</b><span>Preparation</span><b>→</b><span>Analysis</span><b>→</b><span>Models</span><b>→</b><span>Results</span></div>')


# ============================================================
# DATA ANALYSIS
# ============================================================
elif page == "Data Analysis":
    st.title("Data Analysis")
    st.caption("See how traffic changes across time and conditions.")
    try: raw = raw_data()
    except Exception as e: st.error(f"Could not load the dataset: {e}"); st.stop()

    st.markdown("### Dataset Snapshot")
    c1,c2,c3,c4=st.columns(4); c1.metric("Rows",f"{len(raw):,}"); c2.metric("Columns",len(raw.columns)); c3.metric("Missing Cells",f"{int(raw.isna().sum().sum()):,}"); c4.metric("Original Frequency","Hourly")

    st.markdown("### Data Quality")
    missing=raw.isna().sum().sort_values(ascending=False); missing=missing[missing>0]
    if missing.empty: st.success("No missing values found in the raw dataset.")
    else:
        md=missing.reset_index(); md.columns=["Feature","Missing Values"]; md["Percentage"]=(md["Missing Values"]/len(raw)*100).round(2); st.dataframe(md,hide_index=True,width="stretch")

    st.markdown("### Missing Data Handling")

    st.markdown(
        "**We first check the raw observations for missing values and time gaps.** "
        "Traffic is aggregated from hourly observations to a daily series. "
        "Missing daily values in the prepared series are filled using linear "
        "interpolation to keep the analysis series continuous."
    )

    try:
        daily=daily_data(); dc=col(daily,["date","date_time","ds"]); tc=col(daily,["traffic_volume","traffic","y","value"])
        daily=daily.dropna(subset=[dc,tc]).sort_values(dc).copy(); daily[dc]=pd.to_datetime(daily[dc],errors="coerce")
        st.markdown("### Traffic Over Time"); st.caption("Daily traffic volume across the prepared dataset.")
        fig=px.line(daily,x=dc,y=tc,template="plotly_dark"); fig.update_layout(height=430,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Traffic Volume"); st.plotly_chart(fig,width="stretch")
        daily["day_of_week"]=daily[dc].dt.day_name(); daily["month"]=daily[dc].dt.month_name()

        st.markdown("### Weekly Pattern"); st.caption("Average traffic by day of the week.")
        order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly=daily.groupby("day_of_week")[tc].mean().reindex(order).reset_index(); fig=px.bar(weekly,x="day_of_week",y=tc,template="plotly_dark"); fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Average Traffic"); st.plotly_chart(fig,width="stretch")

        st.markdown("### Monthly Pattern"); st.caption("Average traffic by month.")
        mo=["January","February","March","April","May","June","July","August","September","October","November","December"]
        monthly=daily.groupby("month")[tc].mean().reindex(mo).reset_index(); fig=px.bar(monthly,x="month",y=tc,template="plotly_dark"); fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Average Traffic"); st.plotly_chart(fig,width="stretch")
    except Exception as e: st.warning(f"Could not build daily charts: {e}")

    st.markdown("### Weather & Traffic")
    st.caption("Simple relationships between traffic and selected weather features.")
    available=[("temp","Temperature"),("rain_1h","Rain"),("snow_1h","Snow"),("clouds_all","Cloud Cover")]
    available=[x for x in available if x[0] in raw.columns]
    cs=st.columns(2)
    for i,(feature,label) in enumerate(available):
        with cs[i%2]:
            sample=raw[[feature,"traffic_volume"]].dropna(); sample=sample.sample(min(5000,len(sample)),random_state=42)
            fig=px.scatter(sample,x=feature,y="traffic_volume",opacity=.35,template="plotly_dark"); fig.update_layout(title=label,height=330,margin=dict(l=10,r=10,t=45,b=10),xaxis_title=label,yaxis_title="Traffic Volume"); st.plotly_chart(fig,width="stretch")


# ============================================================
# MODEL ANALYSIS
# ============================================================
elif page == "Model Analysis":
    st.title("Model Analysis")
    st.caption("Four predefined models tested on the prepared traffic data.")
    cs=st.columns(4)
    for c,item in zip(cs,[("ARIMA","Statistical","Captures time-based behavior."),("SARIMA","Seasonal","Captures time and seasonal behavior."),("Prophet","Trend & seasonality","Models trend and recurring patterns."),("XGBoost","Machine learning","Learns from historical features.")]):
        with c: model_card(*item)

    m=metrics(); rows=[]
    for model in ["ARIMA","SARIMA","Prophet","XGBoost"]: rows.append({"Model":model,"MAE":score(m,model,"MAE"),"RMSE":score(m,model,"RMSE"),"MAPE":score(m,model,"MAPE")})
    df=pd.DataFrame(rows)
    st.markdown("### Model Performance"); st.caption("Compare results on the held-out test period.")
    display=df.copy()
    for c in ["MAE","RMSE","MAPE"]: display[c]=display[c].map(lambda x:round(x,2) if pd.notna(x) else "—")
    st.dataframe(display,hide_index=True,width="stretch")
    if df["MAE"].notna().any():
        chart=df.dropna(subset=["MAE"]); fig=px.bar(chart,x="Model",y="MAE",template="plotly_dark",text_auto=".1f"); fig.update_layout(height=370,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Mean Absolute Error"); st.plotly_chart(fig,width="stretch")
        best=chart.loc[chart.MAE.idxmin()]; st.html(f'<div class="result"><div class="result-label">Lowest MAE in stored evaluation</div><div class="result-number">{best.Model}</div><div class="result-sub">MAE: {best.MAE:.2f}</div></div>')


# ============================================================
# NEXT-DAY RESULT
# ============================================================
elif page == "Next-Day Result":
    st.title("Next-Day Result")
    st.caption("View the next-day estimate produced by each tested model.")
    st.markdown('<div class="muted">The values are model-generated results based on the available historical data.</div>',unsafe_allow_html=True)
    st.write("")
    if st.button("Generate Next-Day Result", type="primary"):
        with st.spinner("Running the tested models..."):
            try:
                from src.predict import predict_all_models
                result=predict_all_models(); pred=normalize_predictions(result)
                if not pred: st.error("The models ran, but the returned result format could not be read by the interface.")
                else: st.session_state["next_day_predictions"]=pred
            except ModuleNotFoundError as e:
                st.error("The project source folder could not be imported. This usually happens when Streamlit is started from the wrong folder. The app now adds the project root automatically; restart Streamlit after replacing app.py.")
            except Exception as e:
                st.error(f"Could not generate the next-day result: {type(e).__name__}: {e}")
    pred=st.session_state.get("next_day_predictions")
    if pred:
        avg=sum(pred.values())/len(pred)
        c1,c2=st.columns([1.1,1])
        with c1: st.html(f'<div class="result"><div class="result-label">Combined view</div><div class="result-number">{avg:,.0f}</div><div class="result-sub">Average of available model results</div></div>')
        with c2: st.dataframe(pd.DataFrame({"Model":list(pred),"Estimated Traffic":[round(v) for v in pred.values()]}),hide_index=True,width="stretch")
        st.markdown("### Model Results")
        rdf=pd.DataFrame({"Model":list(pred),"Estimated Traffic":list(pred.values())}); fig=px.bar(rdf,x="Model",y="Estimated Traffic",template="plotly_dark",text_auto=".0f"); fig.update_layout(height=400,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Traffic Volume"); st.plotly_chart(fig,width="stretch")
    else:
        st.caption("Click the button above to run all four tested models and display their individual results.")


# ============================================================
# DIAGNOSTICS
# ============================================================
elif page == "Diagnostics":
    st.title("Diagnostics")
    st.caption("See how model results behave against observed traffic.")
    p=test_predictions()
    if p is None or p.empty: st.warning("No test prediction artifact was found in artifacts/test_predictions.csv.")
    else:
        actual=col(p,["actual","traffic_volume","y","actual_traffic"]); date=col(p,["date","date_time","ds","timestamp"]); models=[c for c in p.columns if c.lower() in {"arima","sarima","prophet","xgboost"}]
        if actual and models:
            x=date or "Index"
            if not date: p=p.copy(); p["Index"]=range(len(p))
            if date: p=p.copy(); p[date]=pd.to_datetime(p[date],errors="coerce")
            fig=go.Figure(); fig.add_trace(go.Scatter(x=p[x],y=p[actual],mode="lines",name="Actual"))
            for model in models: fig.add_trace(go.Scatter(x=p[x],y=p[model],mode="lines",name=model))
            fig.update_layout(template="plotly_dark",height=470,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Traffic Volume"); st.plotly_chart(fig,width="stretch")
            st.markdown("### Residuals"); st.caption("Residual = actual traffic − model result.")
            chosen=st.selectbox("Choose a model",models); residual=pd.to_numeric(p[actual],errors="coerce")-pd.to_numeric(p[chosen],errors="coerce"); rd=pd.DataFrame({"Index":range(len(residual)),"Residual":residual}).dropna()
            fig=px.line(rd,x="Index",y="Residual",template="plotly_dark"); fig.add_hline(y=0,line_dash="dash"); fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=10),xaxis_title=None,yaxis_title="Residual"); st.plotly_chart(fig,width="stretch")
            st.markdown("### Error Distribution"); fig=px.histogram(rd,x="Residual",nbins=40,template="plotly_dark"); fig.update_layout(height=340,margin=dict(l=10,r=10,t=10,b=10),xaxis_title="Residual",yaxis_title="Count"); st.plotly_chart(fig,width="stretch")
        else: st.dataframe(p.head(20),hide_index=True,width="stretch")
    st.markdown("### Interpretation")
    st.markdown("Residuals show where model results differ from observed traffic. Patterns remaining in the residuals can indicate information that the model did not capture.")
