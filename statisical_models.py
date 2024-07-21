import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from itertools import product
import statsmodels.api as sm 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from error_metrics_weights import *
from tabulate import tabulate


df_train = pd.read_csv("Hourly_wdates.csv", parse_dates=True, index_col=0)
df_test = pd.read_csv("Hourly-test.csv", parse_dates=True, index_col=0)


# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

def arima(df_train, df_test, horizons):
    # ARIMA Modeling
    arima_model = ARIMA(df_train, order=(2, 1, 5))
    arima_results = arima_model.fit()

    # Make predictions for the specified horizon in the test set
    start_index = df_test.index[0]
    end_index = df_test.index[min(horizons-1, len(df_test)-1)]
    
    predictions = arima_results.predict(start=start_index, end=end_index, typ='levels')

    return predictions



def bats(df_train, df_test, horizons):
    ### BATS Modeling ###
    bats_model = ExponentialSmoothing(df_train, trend='add', damped_trend=True, seasonal='add', seasonal_periods=24)
    bats_results = bats_model.fit()

    # Make predictions for the specified horizon in the test set
    start_index = df_test.index[0]
    end_index = df_test.index[min(horizons - 1, len(df_test) - 1)]

    predictions = bats_results.predict(start=start_index, end=end_index)

    return predictions



def sarimax(df_train, df_test, horizons):


    # Fit the SARIMAX model
    sarimax_model = SARIMAX(df_train, order=(2, 1, 4), seasonal_order=(1, 0, 1, 24))
    sarimax_results = sarimax_model.fit()

    # Make predictions for the specified horizon in the test set
    start_index = df_test.index[0]
    end_index = df_test.index[min(horizons - 1, len(df_test) - 1)]

    predictions = sarimax_results.predict(start=start_index, end=end_index, typ='levels')

    return predictions
