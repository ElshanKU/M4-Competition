import pandas as pd
from darts import TimeSeries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from darts.models import NaiveSeasonal

#####
# Models
#####

##### 
# Holt-Winters Method
def holt_winters(train_df, test_df, alpha=0.1, beta=0.1, gamma=0.1, seasonal_periods=24, horizons = 48):
    """
    Generates forecasts using Exponential Smoothing method.

    Parameters:
    train_df (pd.Series): Training data as a pandas Series.
    val_df (pd.Series): Validation data as a pandas Series.
    alpha (float): Smoothing parameter for level.
    beta (float): Smoothing parameter for trend.
    gamma (float): Smoothing parameter for seasonal component.
    seasonal_periods (int): The number of observations per seasonal cycle.
    horizons(int) : The number of desired horizons

    Returns:
    pd.Series: Forecasted values as a pandas Series.
    """

   # Fitting the model
    model = ExponentialSmoothing(train_df, trend='add', 
                                 damped_trend=True,
                                  seasonal='add', 
                                  seasonal_periods=seasonal_periods).fit(
                                  smoothing_level=alpha, 
                                  smoothing_trend=beta, 
                                  smoothing_seasonal=gamma)

    # Forecast
    forecast = model.forecast(steps= horizons)

    # Get actual and predicted data
    return forecast


##### 
# Seasonal Averaging Method

def seasonal_averaging_forecast(train_df, test_df, seasonal_periods=24):
    """

    Generates forecasts using Seasonal Averaging Forecast method,
    Parameters:

    train_df: DataFrame containing training data
    val_df: DataFrame containing validation data
    seasonal_periods: Number of periods in a season (e.g., 24 for hourly data)

    Returns:
    Returns:
    forecast: Series containing the forecasted values
    """
    # Calculate seasonal averages
    seasonal_averages = train_df.groupby(train_df.index.hour).mean()
    # Forecast using seasonal averages
    forecast = pd.Series(seasonal_averages[test_df.index.hour].values, index=test_df.index)
    
    return forecast

##### 
# Seasonal Naive Forecast

def seasonal_naive_forecast(train_df, test_df, seasonal_periods=24, horizons = 48):
    """
    Generates forecasts using a seasonal naive model.

    Parameters:
    train_df (pd.Series): Training data as a pandas Series.
    val_df (pd.Series): Validation data as a pandas Series.
    seasonal_periods (int, optional): The number of observations per seasonal cycle. Default is 24 for hourly data.
    horizon (int, optional): The forecast horizon, i.e., the number of future time steps to forecast. Default is 48.

    Returns:
    pd.Series: Forecasted values as a pandas Series.
    """

    ts_joined = pd.concat([train_df, test_df[:horizons]])

    # Seasonal Naive Model
    seasonal_naive_model = ts_joined.shift(periods=seasonal_periods)

    # Forecast
    forecast = seasonal_naive_model[-horizons:]

    return forecast


