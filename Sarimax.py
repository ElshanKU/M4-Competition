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
from error_metrics import smape, mase
from tabulate import tabulate

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing the trained and tested values
df_train = pd.read_csv("Hourly_wdates.csv", parse_dates=True, index_col=0)
df_test = pd.read_csv("Hourly-test.csv", parse_dates=True, index_col=0)

# Defining n as the Time Series example 
n = 1

# Dropping NA values and using only positive values
df = df_train.iloc[n - 1, 2:].dropna() 
df = df[df > 0]
df = pd.to_numeric(df, errors="coerce")

# use initial value and set it ti the first observation 
initial_date = df_train.iloc[n - 1, 0]
index = pd.date_range(start=initial_date, periods=len(df), freq="H")
df.index = index 

# Doing the same for tested values, just using the last date of the trained set 
tested = df_test.iloc[n - 1, 1:].dropna()
tested = tested[tested > 0]
tested = pd.to_numeric(tested, errors="coerce")

last_date_train = df.index[-1]
index_test = pd.date_range(start=last_date_train, periods=len(tested), freq="H")
tested.index = index_test

# Splitting into training and validation sets 
split_index = int(len(df) * 0.90)
train_df = df.iloc[: split_index]
val_df = df.iloc[split_index :]

# Plotting original data
plt.plot(train_df.index, train_df, label="Trained Values", color="blue")
plt.plot(val_df.index, val_df, label="Validation Values", color="red")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Checking Stationarity

result = adfuller(df)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')

# Interpret the results

if result[1] <= 0.05:
    print("Reject the null hypothesis. The data is stationary.")
else:
    print("Fail to reject the null hypothesis. The data is not stationary.")


# Making series stationary 

row_series = pd.Series(df)

# Perform differencing
differenced_data = row_series.diff().dropna()

# Plot the differenced data to visualize the result
plt.plot(differenced_data)
plt.xlabel('Index')
plt.ylabel('Differenced Value')
plt.title('Differenced Data')
plt.xticks(rotation=45)
plt.show()

# Check again if data is stationary 
result = adfuller(differenced_data)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')

# Interpret the results
if result[1] <= 0.05:
    print("Reject the null hypothesis. The data is stationary.")
else:
    print("Fail to reject the null hypothesis. The data is not stationary.")


# Seasonal Decomposition 

window_size = 24  

# Perform seasonal decomposition
decomposition = seasonal_decompose(differenced_data, model='additive', period=window_size)

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(differenced_data, label='Original Time Series')
plt.title('Original Time Series')
plt.xlabel('Hour')
plt.ylabel('Input')
plt.legend()
plt.show()

# Plot the trend component
plt.figure(figsize=(12, 6))
plt.plot(decomposition.trend, label='Trend')
plt.title('Trend Component')
plt.xlabel('Hour')
plt.ylabel('Input')
plt.legend()
plt.show()

# Plot the seasonal component
plt.figure(figsize=(12, 6))
plt.plot(decomposition.seasonal, label='Seasonal')
plt.title('Seasonal Component')
plt.xlabel('Hour')
plt.ylabel('Input')
plt.legend()
plt.show()

# Plot the residual component
plt.figure(figsize=(12, 6))
plt.plot(decomposition.resid, label='Residual')
plt.title('Residual Component')
plt.xlabel('Hour')
plt.ylabel('Input')
plt.legend()
plt.show()

# Plot ACF
plt.figure(figsize=(12, 6))
plot_acf(differenced_data, lags=30, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

# Plot PACF
plt.figure(figsize=(12, 6))
plot_pacf(differenced_data, lags=30, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

# Fit the SARIMAX model
sarimax_model = SARIMAX(train_df, order=(2, 1, 4), seasonal_order=(1, 0, 1, 24))
sarimax_results = sarimax_model.fit()

# Make predictions for the validation period
start_index = val_df.index[0]
end_index = val_df.index[-1]
predictions = sarimax_results.predict(start=start_index, end=end_index, typ='levels')

# Evaluate the performance of the model
mse = mean_squared_error(val_df, predictions)
print("Mean Squared Error (MSE) on validation data:", mse)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(val_df.index, val_df, label='Actual', color='blue')
plt.plot(predictions.index, predictions, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values (Validation Period)')
plt.legend()
plt.show()

# Calculating sMAPE and MASE
smape_value = round(smape(val_df, predictions), 2)
mase_value = round(mase(train_df,val_df, predictions[:-2], 24), 2)

# Table of comparassion
table = [["Symmetric mean absolute percentage error (sMAPE)", f"{smape_value} %"],
         ["Mean Absolut Scaled Error                (MASE)", mase_value]]

# Print table
print(tabulate(table, headers=["Metric", "Value"], tablefmt="simple")) 
