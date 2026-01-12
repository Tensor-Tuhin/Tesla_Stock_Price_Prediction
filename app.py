# Importing the libraries
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# Importing the functions from timeseries_utils
from utils.timeseries_utils import (
    plot_adj_close,
    plot_open,
    plot_vol,
    plot_high_low,
    plot_overlay,
    plot_clo_vol,
    plot_price_change,
    plot_daily_returns,
    inverse_transform_adj_close)


# App configuration
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide")

st.title("📈 Tesla Stock Price Prediction & Analysis")

LOOKBACK = 60
N_FEATURES = 2
TARGET_COL = 0

# Load artifacts (cached)
@st.cache_resource
def load_artifacts():
    df = joblib.load("../data/artifacts/df.joblib")
    df_model = joblib.load("../data/artifacts/df_model.joblib")
    scaler = joblib.load("../data/artifacts/scaler.joblib")
    return df, df_model, scaler

df, df_model, scaler = load_artifacts()

# Sidebar controls
st.sidebar.header("🔧 Controls")

section = st.sidebar.radio(
    "Choose Section",
    ["EDA Analytics", "Future Prediction"])

# EDA Section
if section == "EDA Analytics":

    st.subheader("Exploratory Data Analysis")

    eda_option = st.selectbox(
        "Select Analysis",
        [
            "Adjusted Close",
            "Open Price",
            "Volume",
            "High vs Low",
            "Open vs Adj Close (Overlay)",
            "Adj Close vs Volume",
            "Daily Price Change",
            "Daily Returns"
        ]
    )

    if eda_option == "Adjusted Close":
        st.pyplot(plot_adj_close(df))

    elif eda_option == "Open Price":
        st.pyplot(plot_open(df))

    elif eda_option == "Volume":
        st.pyplot(plot_vol(df))

    elif eda_option == "High vs Low":
        st.pyplot(plot_high_low(df))

    elif eda_option == "Open vs Adj Close (Overlay)":
        st.pyplot(plot_overlay(df))

    elif eda_option == "Adj Close vs Volume":
        st.pyplot(plot_clo_vol(df))

    elif eda_option == "Daily Price Change":
        st.pyplot(plot_price_change(df))

    elif eda_option == "Daily Returns":
        st.pyplot(plot_daily_returns(df))


# Prediction Section
else:

    st.subheader("Future Stock Price Prediction")

    model_type = st.selectbox(
        "Select Model",
        ["SimpleRNN", "LSTM (Best model)"])

    horizon = st.selectbox(
        "Prediction Horizon",
        ["1 Day", "5 Days", "10 Days"])

    horizon_map = {
        "1 Day": 1,
        "5 Days": 5,
        "10 Days": 10}

    horizon_days = horizon_map[horizon]

    if model_type == "LSTM (Best model)":
        model_path = f"../data/models/lstm/lstm_{horizon_days}day.h5"
    else:
        model_path = f"../data/models/simplernn/rnn_{horizon_days}day.h5"

    model = load_model(model_path)

    st.markdown("### 🔍 Prediction Logic")
    st.markdown(
        """
        - Uses *last 60 days* of data  
        - No retraining  
        - Fully inference-based  
        - Scaled → Model → Inverse scaled  
        """)

    if st.button("Predict Future Price 🚀"):

        # Taking last 60 rows
        last_sequence = df_model.values[-LOOKBACK:]

        # Scaling
        last_sequence_scaled = scaler.transform(last_sequence)

        # Reshaping for model: (1, 60, 2)
        X_input = last_sequence_scaled.reshape(1, LOOKBACK, N_FEATURES)

        # Predict
        pred_scaled = model.predict(X_input)

        # Inverse transform the prediction
        pred_price = inverse_transform_adj_close(
            pred_scaled,
            scaler,
            N_FEATURES,
            TARGET_COL)

        st.success(f"📌 Predicted Adjusted Close after {horizon}:")
        st.metric(label = "Predicted Price :-",
                  value = f"${pred_price[0]:.2f}")

        st.info("⚠️ This is a model-based forecast and not financial advice.")