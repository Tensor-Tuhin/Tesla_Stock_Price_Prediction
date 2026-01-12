import numpy as np
import matplotlib.pyplot as plt

def plot_adj_close(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df['Adj Close'], color='green')
    ax.set_title('Adjusted closing price over the years')
    ax.grid(color='y')
    ax.set_ylabel('Adjusted Closing Price')
    ax.set_xlabel('Years')
    return fig

def plot_open(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index,df['Open'],color='darkgreen')
    ax.set_title('Open prices across the years')
    ax.set_xlabel('Years')
    ax.set_ylabel('Open Prices')
    ax.grid(color='y')
    return fig

def plot_vol(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index,df['Volume'],color='blue')
    ax.set_title('Volume of stocks traded over the years')
    ax.set_xlabel('Years')
    ax.set_ylabel('Volume')
    ax.grid(color='y')
    return fig

def plot_high_low(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index,df['High'],color='dodgerblue',label='High')
    ax.plot(df.index,df['Low'],color='crimson',label='Low')
    ax.set_title('Visualising the highs and lows of the stock over the years')
    ax.grid(color='lightgreen')
    ax.set_xlabel('Years')
    ax.legend()
    return fig

def plot_overlay(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df['Open'], color='red', label='Open')
    ax.plot(df.index, df['Adj Close'], color='green', label='Adj Close')
    ax.set_title('Visualising Price Consistency within a day')
    ax.set_xlabel('Years')
    ax.grid(color='darkgray')
    ax.legend()
    return fig

def plot_clo_vol(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.scatter(df['Adj Close'], df['Volume'],color='darkcyan', alpha=0.5)
    ax.set_title('Adjusted Close vs Volume')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel('Volume')
    return fig

def plot_price_change(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df['price_change'], color='midnightblue')
    ax.set_title('Daily Price Change')
    ax.set_xlabel('Years')
    ax.set_ylabel('Difference between opening and closing')
    ax.grid(color='gold')
    return fig

def plot_daily_returns(df):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df.index, df['daily_returns'], color='goldenrod')
    ax.set_title('Daily Return over the years')
    ax.set_xlabel('Years')
    ax.set_ylabel('daily returns')
    ax.grid(color='lightsteelblue')
    return fig

def plot_test_pred(y_pred, y_test):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(range(len(y_test)), y_test, color='blue', linestyle='dotted', label='Test')
    ax.plot(range(len(y_pred)), y_pred, color='green', label='Prediction')
    ax.set_title('Prediction vs Test values')
    ax.set_ylabel('values')
    ax.legend()
    ax.grid(color='y')
    return fig

def create_sequences(data, lookback, horizon):
    x, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        # Append input sequence
        x.append(data[i:i + lookback])                      # Creates on input window per iteration

        # Append target value
        y.append(data[i + lookback + horizon - 1, 0])       # Pairs each input sequence with single target value
        
    # Return converted arrays
    return np.array(x), np.array(y)

def inverse_transform_adj_close(scaled_values, scaler, n_features=2, target_col=0):
    dummy = np.zeros((len(scaled_values), n_features))              # Creates fake data of only zeros that matches training shape
    dummy[:,target_col] = scaled_values.reshape(-1)                 # Put real predictions in target column, leaving others as 0
    return scaler.inverse_transform(dummy)[:,target_col]            # Gives back original-scale predictions

