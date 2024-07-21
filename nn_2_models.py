from darts import TimeSeries
from darts.models import NBEATSModel, TransformerModel
from error_metrics_weights import *

import pandas as pd
import numpy as np
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense
from sklearn.metrics import r2_score

df_train = pd.read_csv("Hourly_wdates.csv", parse_dates=True, index_col=0)
df_test = pd.read_csv("Hourly-test.csv", parse_dates=True, index_col=0)

def nbeats_model(df_train, df_test, horizons):
    train_series_ts = TimeSeries.from_series(df_train)
    df_test_series = TimeSeries.from_series(df_test)

    # Initialize and train the N-BEATS model
    nbeats_model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=24,
        n_epochs=15,
        num_stacks=3,
        num_blocks=3,
        num_layers=3,
        layer_widths=512,
        batch_size=32,
        random_state=42
    )

    nbeats_model.fit(train_series_ts)

    # Predict the next 'horizons' values
    predictions = nbeats_model.predict(n=horizons).values().flatten()

    # Create a Pandas Series with the correct index for predictions
    predictions_series = pd.Series(predictions, index=df_test.index[:horizons])

    return predictions_series

def gru_model(df_train, df_test, horizons):
    seq_length = 24  

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(df_train.values, seq_length)

    model = Sequential([
        GRU(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    forecast = []
    current_batch = df_train.iloc[-seq_length:].values.reshape((1, seq_length, 1))

    for i in range(horizons):
        current_pred = model.predict(current_batch)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    forecast_index = pd.date_range(start=df_test.index[0], periods=horizons, freq='H')
    forecast_series = pd.Series(np.array(forecast).flatten(), index=forecast_index)

    return forecast_series

def lstm_model(df_train, df_test, horizons):
    # Sequence length for each input (hours in a day)
    seq_length = 24  

    # Function to create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    # Create sequences for LSTM from training series
    X_train, y_train = create_sequences(df_train.values, seq_length)

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train LSTM model
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # Initialize the forecasting process
    forecast = []
    current_batch = df_train.iloc[-seq_length:].values.reshape((1, seq_length, 1))

    # Forecasting loop for each horizon
    for i in range(horizons):
        current_pred = model.predict(current_batch)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    # Create a Pandas Series with the correct index for forecasts
    forecast_index = pd.date_range(start=df_test.index[0], periods=horizons, freq='H')
    forecast_series = pd.Series(np.array(forecast).flatten(), index=forecast_index)

    return forecast_series
