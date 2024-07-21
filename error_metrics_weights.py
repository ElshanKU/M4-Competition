import numpy as np
import pandas as pd
from scipy.optimize import minimize


def smape(a, b):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE (clipped between 0% and 100%)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    smape_value = np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b)))
    
    # Clip sMAPE values to ensure they are within the range of 0% to 100%
    smape_value = max(0, min(smape_value, 1)) * 100
    
    return smape_value

def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample.iloc[i - freq])

    masep = np.mean(np.abs(np.array(insample.iloc[freq:]) - np.array(y_hat_naive)))

    #print(mean_abs_error)
    #print(masep)

    return np.mean(np.abs(y_test - y_hat_test)) / masep



# Function to calculate equal weights
# Used for 9 models
def estimateEW(num_models):
    return np.full(num_models, 1/num_models)

# Function to calculate Optimal Weights
# Used for the best 3 models
def estimateInverseErrorWeights(ErrorData):
    # Normalize each error metric to the range [0, 1]
    normalized_errors = ErrorData / np.max(ErrorData, axis=0)
    
    # Calculate the average normalized error for each forecast
    avg_errors = np.mean(normalized_errors, axis=1)
    
    # Calculate the inverse error weights
    weights = 1 / avg_errors
    
    # Normalize the weights so they sum to 1
    weights = weights / np.sum(weights)
    
    return weights