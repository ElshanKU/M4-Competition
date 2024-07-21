import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from error_metrics_weights import *
from tabulate import tabulate

## Data Preparation Functon

def prepare_data(n, train_file, test_file):
    # Importing the trained and tested values
    df_train = train_file
    df_test = test_file

    # Dropping NA values and using only positive values
    df = df_train.iloc[n - 1, 2:].dropna()
    df = df[df > 0]
    df = pd.to_numeric(df, errors="coerce")

    # Use the initial value and set it to the first observation, then add an hour to next observations
    initial_date = df_train.iloc[n - 1, 0]
    index = pd.date_range(start=initial_date, periods=len(df), freq="h")
    df.index = index

    # Prepare the tested values
    tested = df_test.iloc[n - 1, :].dropna()
    tested = tested[tested > 0]
    tested = pd.to_numeric(tested, errors="coerce")

    last_date_train = df.index[-1]
    index_test = pd.date_range(start=last_date_train + pd.Timedelta(hours=1), periods=len(tested), freq="h")
    tested.index = index_test
    
    return df, tested


## Visualization Original Data Function

def plot_original_data(train, test, n, h):
    plt.figure(figsize=(20, 6))
    plt.plot(train.index, train, color="black", label="Trained Values", lw=3)
    plt.plot(test[:h].index, test[:h], label="Tested Values", color="red", lw=3)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(40))
    plt.xticks(rotation=75)
    plt.title(f"Hourly Time Series No. {n}")
    plt.xlabel("Date")
    plt.ylabel("Date Observation Value")
    plt.legend()
    plt.show()


## Visualization Predictions Data Function

def plot_forecast(train, test, model_1, m_1_name, model_2, m_2_name, model_3, m_3_name, n, h,type):
    # Normal plot
    plt.figure(figsize=(18, 4))
    plt.plot(train.index, train, color="black", label="Training Values", lw=3)
    plt.plot(test[:h].index, test[:h], color="red", label="Tested values", lw=3)

    # Model 1 Prediction
    plt.plot(model_1.index, model_1, color="green", label=m_1_name, linestyle="--")
    # Model 2 Prediction
    plt.plot(model_2.index, model_2, color="blue", label=m_2_name, linestyle="--")
    # Model 3 Prediction
    plt.plot(model_3.index, model_3, color="purple", label=m_3_name, linestyle="--")

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.xticks(rotation=45)

    plt.title(f"Hourly Time Series No. {n}. {type} Models")
    plt.xlabel("Date")
    plt.ylabel("Observation Value")

    plt.legend()
    plt.show()

    # Closer plot
    plt.figure(figsize=(20, 6))

    plt.plot(train.index[-h:], train[-h:], color="black", label=f"Last {h} values from Training Set", lw=3)
    plt.plot(test[:h].index, test[:h], color="red", label="Testes values", lw=3)

    # Model 1 Prediction
    plt.plot(model_1.index, model_1, color="green", label=m_1_name, linestyle="--")
    # Model 2 Prediction
    plt.plot(model_2.index, model_2, color="blue", label=m_2_name, linestyle="--")
    # Model 3 Prediction
    plt.plot(model_3.index, model_3, color="purple", label=m_3_name, linestyle="--")

    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.xticks(rotation=45)

    plt.title(f"Hourly Time Series No. {n}. {type} Models")
    plt.xlabel("Date")
    plt.ylabel("Observation Value")

    plt.legend()
    plt.show()


# Function to evaluate error metrics for the 9 forecasts
def evaluate_forecasts_9(train, test, forecasts, h):
    metrics = {}
    for model_name, forecast in forecasts.items():
        smape_val = round(smape(test[:h], forecast), 2)
        mase_val = round(mase(train, test[:h], forecast, 24), 2)
        metrics[model_name] = {"sMAPE": smape_val, "MASE": mase_val}
    
    # Combine forecasts if needed and calculate combined metrics
    combined_forecast = np.mean(list(forecasts.values()), axis=0)
    combined_smape = round(smape(test[:h], combined_forecast), 2)
    combined_mase = round(mase(train, test[:h], combined_forecast, 24), 2)
    
    return metrics, combined_forecast, combined_smape, combined_mase



# Function to plot the 9 forecasts
def plot_forecasts_9(train, test, forecasts, combined_forecast,horizons, n, type):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode="lines", name="Training Values", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=test.index[:horizons], y=test[:horizons], mode="lines", name="Tested values", line=dict(color="red", width=3)))

    for model_name, forecast in forecasts.items():
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name=model_name, line=dict(dash="dash")))

    fig.add_trace(go.Scatter(x=test.index[:horizons], y=combined_forecast[:horizons], mode='lines+markers', name='Combined Forecast', line=dict(color='orange', width=3, dash='dash')))
    fig.update_layout(
        title=f"Hourly Time Series No. {n}. {type} Models",
        xaxis_title="Date",
        yaxis_title="Observation Value",
        legend_title="Legend",
        xaxis=dict(nticks=20),
        template="plotly_white"
    )
    fig.show()

    fig_closer = go.Figure()
    fig_closer.add_trace(go.Scatter(x=train.index[-len(test):], y=train[-len(test):], mode="lines", name=f"Last {len(test)} values from Training Set", line=dict(color="black", width=3)))
    fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=test[:horizons], mode="lines", name="Tested values", line=dict(color="red", width=3)))

    for model_name, forecast in forecasts.items():
        fig_closer.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name=model_name, line=dict(dash="dash")))

    fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=combined_forecast[:horizons], mode='lines+markers', name='Combined Forecast', line=dict(color='orange', width=3, dash='dash')))
    fig_closer.update_layout(
        title=f"Hourly Time Series No. {n}. {type} Models",
        xaxis_title="Date",
        yaxis_title="Observation Value",
        legend_title="Legend",
        xaxis=dict(nticks=6),
        template="plotly_white"
    )
    fig_closer.show()

# Function that uses evaluate_forecasts_9 and plot_forecasts_9
def evaluate_and_plot_forecasts_9(train, test, forecasts, h, n, type):
    # Step 1: Evaluate forecasts
    metrics, combined_forecast, combined_smape, combined_mase = evaluate_forecasts_9(train, test, forecasts, h)
    
    # Step 2: Print Metrics and Weights in a Table
    num_models = len(forecasts)
    weights = estimateEW(num_models)
    
    table = [["Model", "sMAPE (%)", "MASE", "Weight"]]
    for i, model_name in enumerate(forecasts.keys()):
        table.append([model_name, metrics[model_name]["sMAPE"], metrics[model_name]["MASE"], round(weights[i], 2)])
    table.append(["Combined", combined_smape, combined_mase, "-"])
    print(tabulate(table, headers="firstrow", tablefmt="heavy_grid"))
    
    # Step 3: Plot forecasts
    plot_forecasts_9(train, test, forecasts, combined_forecast, h, n, type)


# Function to evaluate forecasts and return metrics
def evaluate_forecasts_3(train, test, forecasts, h):
    metrics = {}
    for model_name, forecast in forecasts.items():
        smape_val = round(smape(test[:h], forecast), 2)
        mase_val = round(mase(train, test[:h], forecast, 24), 2)
        metrics[model_name] = {'sMAPE': smape_val, 'MASE': mase_val}
    return metrics, None, None, None



# Function to evaluate 9 best models and select the best one from each category
def evaluate_models_3(train, test, forecasts_heuristic, forecasts_statistical, forecasts_nn):
    # Heuristic Models
    metrics_heuristic, _, _, _ = evaluate_forecasts_3(train, test, forecasts_heuristic, len(test))
    best_heuristic_model = min(metrics_heuristic, key=lambda x: metrics_heuristic[x]['sMAPE'] + metrics_heuristic[x]['MASE'])

    # Statistical Models
    metrics_statistical, _, _, _ = evaluate_forecasts_3(train, test, forecasts_statistical, len(test))
    best_statistical_model = min(metrics_statistical, key=lambda x: metrics_statistical[x]['sMAPE'] + metrics_statistical[x]['MASE'])

    # Neural Network Models
    metrics_nn, _, _, _ = evaluate_forecasts_3(train, test, forecasts_nn, len(test))
    best_nn_model = min(metrics_nn, key=lambda x: metrics_nn[x]['sMAPE'] + metrics_nn[x]['MASE'])

    return (best_heuristic_model, metrics_heuristic), (best_statistical_model, metrics_statistical), (best_nn_model, metrics_nn)


# Function that uses evaluate_forecasts_3 and plot_forecasts_3 to plot the Weights and Plots

def evaluate_and_plot_forecasts_3(train, test,horizons, forecasts_heuristic, forecasts_statistical, forecasts_nn):
    # Step 1: Evaluate models and select the best one from each category
    (best_heuristic_model, metrics_heuristic), (best_statistical_model, metrics_statistical), (best_nn_model, metrics_nn) = evaluate_models_3(train, test, forecasts_heuristic, forecasts_statistical, forecasts_nn)
    
    # Print metrics table
    table = [["Model Category", "Best Model", "sMAPE (%)", "MASE", "Weight"]]
    table.append(["Heuristic Models", best_heuristic_model, metrics_heuristic[best_heuristic_model]['sMAPE'], metrics_heuristic[best_heuristic_model]['MASE'], "N/A"])
    table.append(["Statistical Models", best_statistical_model, metrics_statistical[best_statistical_model]['sMAPE'], metrics_statistical[best_statistical_model]['MASE'], "N/A"])
    table.append(["Neural Network Models", best_nn_model, metrics_nn[best_nn_model]['sMAPE'], metrics_nn[best_nn_model]['MASE'], "N/A"])
    #print(tabulate(table, headers="firstrow", tablefmt="heavy_grid"))

    # Step 2: Combine forecasts using optimal weights
    ErrorData = np.array([[metrics_heuristic[best_heuristic_model]['sMAPE'], metrics_heuristic[best_heuristic_model]['MASE']],
                          [metrics_statistical[best_statistical_model]['sMAPE'], metrics_statistical[best_statistical_model]['MASE']],
                          [metrics_nn[best_nn_model]['sMAPE'], metrics_nn[best_nn_model]['MASE']]])
    
    optimal_weights = estimateInverseErrorWeights(ErrorData)

    # Update weights in the table
    table[1][4] = round(optimal_weights[0], 3)  # Heuristic model weight
    table[2][4] = round(optimal_weights[1], 3)  # Statistical model weight
    table[3][4] = round(optimal_weights[2], 3)  # Neural network model weight

    #print(tabulate(table, headers="firstrow", tablefmt="heavy_grid"))
    # Step 3: Plot combined forecast
    combined_forecast = np.zeros_like(test)  # Placeholder for combined forecast

    # Combine forecasts using optimal weights
    if best_heuristic_model in forecasts_heuristic:
        combined_forecast += optimal_weights[0] * forecasts_heuristic[best_heuristic_model]

    if best_statistical_model in forecasts_statistical:
        combined_forecast += optimal_weights[1] * forecasts_statistical[best_statistical_model]

    if best_nn_model in forecasts_nn:
        combined_forecast += optimal_weights[2] * forecasts_nn[best_nn_model]

    # Calculate combined model metrics
    combined_smape = round(smape(test[:horizons], combined_forecast), 2)
    combined_mase = round(mase(train, test[:horizons], combined_forecast, 24), 2)

    # Add combined model metrics to the table
    table.append(["Combined Model", "Combined Forecast", combined_smape, combined_mase, "N/A"])

    print(tabulate(table, headers="firstrow", tablefmt="heavy_grid"))

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Values', line=dict(color='black', width=3)))
    fig.add_trace(go.Scatter(x=test.index[:horizons], y=test[:horizons], mode='lines', name='Tested values', line=dict(color='red', width=3)))
    
    # Plot individual forecasts
    if best_heuristic_model in forecasts_heuristic:
        fig.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_heuristic[best_heuristic_model], mode='lines', name=best_heuristic_model, line=dict(dash='dash')))
    
    if best_statistical_model in forecasts_statistical:
        fig.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_statistical[best_statistical_model], mode='lines', name=best_statistical_model, line=dict(dash='dash')))
    
    if best_nn_model in forecasts_nn:
        fig.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_nn[best_nn_model], mode='lines', name=best_nn_model, line=dict(dash='dash')))
    
    # Plot combined forecast
    fig.add_trace(go.Scatter(x=test.index[:horizons], y=combined_forecast, mode='lines', name='Combined Forecast', line=dict(color='orange', width=3, dash='dash')))
    
    fig.update_layout(
        title="Combined Forecast using Best Models with Inverse ErrorWeights",
        xaxis_title="Date",
        yaxis_title="Observation Value",
        legend_title="Legend",
        xaxis=dict(nticks=20),
        template="plotly_white"
    )
    fig.show()

    # Closer observation plot
    fig_closer = go.Figure()
    fig_closer.add_trace(go.Scatter(x=train.index[-len(test):], y=train[-len(test):], mode='lines', name=f'Last {len(test)} values from Training Set', line=dict(color='black', width=3)))
    fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=test[:horizons], mode='lines', name='Tested values', line=dict(color='red', width=3)))
    
    if best_heuristic_model in forecasts_heuristic:
        fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_heuristic[best_heuristic_model], mode='lines', name=best_heuristic_model, line=dict(dash='dash')))
    
    if best_statistical_model in forecasts_statistical:
        fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_statistical[best_statistical_model], mode='lines', name=best_statistical_model, line=dict(dash='dash')))
    
    if best_nn_model in forecasts_nn:
        fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=forecasts_nn[best_nn_model], mode='lines', name=best_nn_model, line=dict(dash='dash')))
    
    fig_closer.add_trace(go.Scatter(x=test.index[:horizons], y=combined_forecast, mode='lines', name='Combined Forecast', line=dict(color='orange', width=3, dash='dash')))
    
    fig_closer.update_layout(
        title=f"Closer Observation: Last {len(test[:horizons])} Values from Training Set and Model Forecasts Inverse ErrorWeights Combination",
        xaxis_title="Date",
        yaxis_title="Observation Value",
        legend_title="Legend",
        xaxis=dict(nticks=6),
        template="plotly_white"
    )
    fig_closer.show()