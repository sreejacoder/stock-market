import os
import warnings
from datetime import date, timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st

from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI-Powered Stock Market Analysis Dashboard")
st.caption(
    "Historical stock analysis and machine-learning demonstration "
    "for educational purposes."
)

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_START = date.today() - timedelta(days=730)
DEFAULT_END = date.today()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📊 Stock Input")

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL"
).upper().strip()

start_date = st.sidebar.date_input(
    "Start Date",
    value=DEFAULT_START
)

end_date = st.sidebar.date_input(
    "End Date",
    value=DEFAULT_END
)

prediction_days = st.sidebar.slider(
    "Prediction Days",
    min_value=5,
    max_value=60,
    value=30
)

analyze = st.sidebar.button(
    "🔍 Analyze Stock",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    "For learning purposes only. "
    "This application is not financial or trading advice."
)

# ============================================================
# VALIDATION
# ============================================================

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

if not symbol:
    st.warning("Please enter a stock symbol.")
    st.stop()


# ============================================================
# FETCH HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch historical stock data using Stooq's public CSV endpoint.
    Falls back to generated demo data if the request fails.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    url = (
        f"https://stooq.com/q/d/l/"
        f"?s={symbol.lower()}.us"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
    )

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        from io import StringIO

        data = pd.read_csv(StringIO(response.text))

        if data.empty or "Close" not in data.columns:
            raise ValueError("No valid stock data returned.")

        data["Date"] = pd.to_datetime(data["Date"])

        data = data.sort_values("Date").reset_index(drop=True)

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in numeric_columns:
            if column in data.columns:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce"
                )

        data = data.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        data["Source"] = "Live Historical Data"

        if len(data) >= 50:
            return data

    except Exception:
        pass

    # --------------------------------------------------------
    # DEMO DATA FALLBACK
    # --------------------------------------------------------

    np.random.seed(42)

    dates = pd.bdate_range(
        start=start,
        end=end
    )

    if len(dates) < 50:
        dates = pd.bdate_range(
            end=end,
            periods=250
        )

    base_price = 150.0

    returns = np.random.normal(
        loc=0.0004,
        scale=0.018,
        size=len(dates)
    )

    close_prices = (
        base_price *
        np.exp(np.cumsum(returns))
    )

    open_prices = (
        close_prices *
        (1 + np.random.normal(
            0,
            0.005,
            len(dates)
        ))
    )

    high_prices = np.maximum(
        open_prices,
        close_prices
    ) * (
        1 + np.random.uniform(
            0,
            0.015,
            len(dates)
        )
    )

    low_prices = np.minimum(
        open_prices,
        close_prices
    ) * (
        1 - np.random.uniform(
            0,
            0.015,
            len(dates)
        )
    )

    volume = np.random.randint(
        500000,
        5000000,
        len(dates)
    )

    data = pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume,
        "Source": "Demo Data"
    })

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    df = data.copy()

    df["Daily_Return"] = df["Close"].pct_change()

    df["MA_5"] = (
        df["Close"]
        .rolling(5)
        .mean()
    )

    df["MA_10"] = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    df["MA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["Volatility_10"] = (
        df["Daily_Return"]
        .rolling(10)
        .std()
    )

    df["Momentum_5"] = (
        df["Close"] -
        df["Close"].shift(5)
    )

    df["Momentum_10"] = (
        df["Close"] -
        df["Close"].shift(10)
    )

    df["Target"] = (
        df["Close"].shift(-1)
    )

    return df


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

def train_model(data):

    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA_5",
        "MA_10",
        "MA_20",
        "MA_50",
        "Volatility_10",
        "Momentum_5",
        "Momentum_10"
    ]

    model_data = data[
        feature_columns + ["Target"]
    ].dropna()

    if len(model_data) < 100:
        return None

    X = model_data[feature_columns]
    y = model_data["Target"]

    split_index = int(
        len(model_data) * 0.8
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    model_path = os.path.join(
        MODEL_DIR,
        "stock_xgboost_model.joblib"
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "stock_scaler.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    return {
        "model": model,
        "scaler": scaler,
        "features": feature_columns,
        "mae": mae,
        "rmse": rmse,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions
    }


# ============================================================
# FUTURE PREDICTION
# ============================================================

def predict_future(
    data,
    model,
    scaler,
    feature_columns,
    days
):

    df = data.copy()

    predictions = []

    for _ in range(days):

        temp = create_features(df)

        latest = temp.iloc[-1]

        values = []

        for feature in feature_columns:
            values.append(
                latest[feature]
            )

        X_future = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        X_future = X_future.fillna(
            X_future.median()
        )

        X_scaled = scaler.transform(
            X_future
        )

        prediction = float(
            model.predict(
                X_scaled
            )[0]
        )

        next_date = (
            pd.Timestamp(
                df["Date"].iloc[-1]
            ) +
            pd.offsets.BDay(1)
        )

        new_row = {
            "Date": next_date,
            "Open": prediction,
            "High": prediction,
            "Low": prediction,
            "Close": prediction,
            "Volume": df["Volume"].iloc[-1]
        }

        df = pd.concat(
            [
                df,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        predictions.append({
            "Date": next_date,
            "Predicted_Close": prediction
        })

    return pd.DataFrame(predictions)


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading historical stock data..."):

    stock_data = fetch_stock_data(
        symbol,
        start_date,
        end_date
    )

if stock_data.empty:
    st.error(
        "No stock data available for the selected period."
    )
    st.stop()

features = create_features(
    stock_data
)

# ============================================================
# DATA SOURCE MESSAGE
# ============================================================

if "Source" in stock_data.columns:

    source = stock_data["Source"].iloc[0]

    if source == "Demo Data":
        st.warning(
            "⚠️ Live market data could not be retrieved. "
            "The dashboard is currently displaying generated "
            "demo data for learning/testing."
        )
    else:
        st.success(
            "✅ Historical stock data loaded successfully."
        )


# ============================================================
# KEY METRICS
# ============================================================

latest_close = stock_data["Close"].iloc[-1]

first_close = stock_data["Close"].iloc[0]

change = (
    latest_close -
    first_close
)

change_percent = (
    change /
    first_close *
    100
)

average_close = (
    stock_data["Close"]
    .mean()
)

maximum_close = (
    stock_data["Close"]
    .max()
)

minimum_close = (
    stock_data["Close"]
    .min()
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Latest Close",
    f"${latest_close:,.2f}"
)

col2.metric(
    "Period Change",
    f"{change_percent:.2f}%"
)

col3.metric(
    "Average Close",
    f"${average_close:,.2f}"
)

col4.metric(
    "Maximum",
    f"${maximum_close:,.2f}"
)

col5.metric(
    "Minimum",
    f"${minimum_close:,.2f}"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Historical Analysis",
        "📊 Technical Analysis",
        "🤖 ML Prediction",
        "📋 Data",
        "ℹ️ About"
    ]
)


# ============================================================
# TAB 1 - HISTORICAL ANALYSIS
# ============================================================

with tab1:

    st.subheader(
        f"{symbol} Historical Price"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock_data["Date"],
            y=stock_data["Close"],
            mode="lines",
            name="Closing Price"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Trading Volume")

    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=stock_data["Date"],
            y=stock_data["Volume"],
            name="Volume"
        )
    )

    volume_fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Volume"
    )

    st.plotly_chart(
        volume_fig,
        use_container_width=True
    )


# ============================================================
# TAB 2 - TECHNICAL ANALYSIS
# ============================================================

with tab2:

    st.subheader(
        "Moving Average Analysis"
    )

    ma_fig = go.Figure()

    ma_fig.add_trace(
        go.Scatter(
            x=features["Date"],
            y=features["Close"],
            name="Close"
        )
    )

    ma_fig.add_trace(
        go.Scatter(
            x=features["Date"],
            y=features["MA_20"],
            name="20-Day MA"
        )
    )

    ma_fig.add_trace(
        go.Scatter(
            x=features["Date"],
            y=features["MA_50"],
            name="50-Day MA"
        )
    )

    ma_fig.update_layout(
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(
        ma_fig,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Daily Returns"
        )

        return_fig = go.Figure()

        return_fig.add_trace(
            go.Scatter(
                x=features["Date"],
                y=features["Daily_Return"],
                mode="lines",
                name="Daily Return"
            )
        )

        return_fig.update_layout(
            yaxis_title="Return"
        )

        st.plotly_chart(
            return_fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Return Distribution"
        )

        returns = (
            features["Daily_Return"]
            .dropna()
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.histplot(
            returns,
            kde=True,
            ax=ax
        )

        ax.set_xlabel(
            "Daily Return"
        )

        ax.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig)

    st.subheader(
        "Statistical Summary"
    )

    stats = stock_data[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ].describe()

    st.dataframe(
        stats,
        use_container_width=True
    )


# ============================================================
# TAB 3 - MACHINE LEARNING
# ============================================================

with tab3:

    st.subheader(
        "🤖 XGBoost Stock Prediction"
    )

    st.write(
        "The model uses historical price, volume, "
        "moving-average, volatility and momentum "
        "features for an educational prediction demo."
    )

    with st.spinner(
        "Training XGBoost model..."
    ):

        result = train_model(
            features
        )

    if result is None:

        st.error(
            "Not enough historical data to train the model."
        )

    else:

        col1, col2 = st.columns(2)

        col1.metric(
            "MAE",
            f"{result['mae']:.4f}"
        )

        col2.metric(
            "RMSE",
            f"{result['rmse']:.4f}"
        )

        st.subheader(
            "Actual vs Predicted"
        )

        prediction_fig = go.Figure()

        prediction_fig.add_trace(
            go.Scatter(
                x=result["X_test"].index,
                y=result["y_test"],
                mode="lines",
                name="Actual"
            )
        )

        prediction_fig.add_trace(
            go.Scatter(
                x=result["X_test"].index,
                y=result["predictions"],
                mode="lines",
                name="Predicted"
            )
        )

        prediction_fig.update_layout(
            height=500,
            xaxis_title="Test Sample",
            yaxis_title="Closing Price"
        )

        st.plotly_chart(
            prediction_fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.subheader(
            "Feature Importance"
        )

        importance = pd.DataFrame({
            "Feature": result["features"],
            "Importance": result["model"].feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        importance_fig = go.Figure()

        importance_fig.add_trace(
            go.Bar(
                x=importance["Importance"],
                y=importance["Feature"],
                orientation="h"
            )
        )

        importance_fig.update_layout(
            height=500
        )

        st.plotly_chart(
            importance_fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # FUTURE FORECAST
        # ----------------------------------------------------

        st.subheader(
            f"📅 {prediction_days}-Day Demonstration Forecast"
        )

        future = predict_future(
            features,
            result["model"],
            result["scaler"],
            result["features"],
            prediction_days
        )

        st.dataframe(
            future,
            use_container_width=True
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=stock_data["Date"].tail(90),
                y=stock_data["Close"].tail(90),
                mode="lines",
                name="Historical"
            )
        )

        forecast_fig.add_trace(
            go.Scatter(
                x=future["Date"],
                y=future["Predicted_Close"],
                mode="lines+markers",
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )


# ============================================================
# TAB 4 - RAW DATA
# ============================================================

with tab4:

    st.subheader(
        "Historical Stock Data"
    )

    st.dataframe(
        stock_data.sort_values(
            "Date",
            ascending=False
        ),
        use_container_width=True,
        height=500
    )

    csv = stock_data.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=f"{symbol}_historical_data.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 5 - ABOUT
# ============================================================

with tab5:

    st.subheader(
        "About the Project"
    )

    st.markdown(
        """
        ### AI-Powered Stock Market Analysis Dashboard

        **Domain:** Stock Market

        **Purpose:**
        This project demonstrates how historical stock-market
        data can be collected, analyzed, visualized and used
        with a machine-learning model.

        ### Technologies Used

        - Python 3.11.9
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - XGBoost
        - Matplotlib
        - Seaborn
        - Plotly
        - Statsmodels
        - Requests
        - Joblib
        - python-dotenv

        ### Machine Learning

        The project uses an XGBoost regression model to
        demonstrate next-price prediction using historical
        features.

        ### Disclaimer

        **This project is for educational and learning
        purposes only. It is NOT financial or trading advice.
        Predictions are not guaranteed to be accurate and
        should not be used to make investment decisions.**
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Powered Stock Market Analysis Dashboard | "
    "Educational Project | Not Trading Advice"
)