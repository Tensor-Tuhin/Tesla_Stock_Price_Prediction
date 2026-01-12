# Tesla_Stock_Price_Prediction
Predicting the price of Tesla's Stock (TSLA) based on historical data using SimpleRNN and LSTM models.

A time-series forecasting project that predicts Tesla’s Adjusted Closing Price using SimpleRNN and LSTM models for 1-day, 5-day, and 10-day horizons, with an interactive Streamlit dashboard for analysis and prediction.

*This project is only for educational purposes and not actual financial advice.*

Key Features :-

- Deep Learning models: SimpleRNN (baseline) and LSTM
- Lookback window: 60 days
- Features used: Adj Close, Volume
- Inference-only Streamlit app (no retraining)
- Multiple prediction horizons


Important Files :-

- TSLA.csv - Raw Tesla stock price data
- TSLA_1.ipynb - EDA, feature engineering, preprocessing, SimpleRNN training and evaluation
- TSLA_2.ipynb - LSTM training, evaluation, comparison between the two models
- app.py - Streamlit app for analytics & predictions
- df.joblib - Full dataframe for EDA plots
- df_model.joblib - Modeling dataframe (Adj Close, Volume)
- scaler.joblib - Fitted MinMaxScaler
- lstm_*.h5 - Trained LSTM models (1, 5, 10 days)
- rnn_*.h5 - Trained SimpleRNN models (1, 5, 10 days)
- timeseries_utils.py - Reusable plotting & time-series utilities

Run the app :-

- pip install -r requirements.txt
- streamlit run app.py
