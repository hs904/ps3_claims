import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred, exposure, sample_weight=None):
    """
    Evaluates the predictions based on various metrics.

    Parameters:
    - y_true: np.array or pd.Series, true outcome values.
    - y_pred: np.array or pd.Series, predicted values.
    - exposure: np.array or pd.Series, exposure values for weighting.
    - sample_weight: np.array or pd.Series, additional weights for metrics (optional).

    Returns:
    - metrics_df: pd.DataFrame, with metrics as index and values.
    """
    # Compute metrics
    weighted_mean_true = np.average(y_true, weights=exposure)
    weighted_mean_pred = np.average(y_pred, weights=exposure)

    bias = weighted_mean_pred - weighted_mean_true

    deviance = np.mean(
        exposure * ((y_true - y_pred) ** 2) / weighted_mean_true
    )  # Exposure-adjusted mean squared error
    
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    
    # Gini coefficient calculation
    def gini_coefficient(true, pred, weights):
        order = np.argsort(pred)
        ordered_true = true[order]
        ordered_weights = weights[order]
        cum_claims = np.cumsum(ordered_true * ordered_weights)
        cum_claims /= cum_claims[-1]
        cum_weights = np.cumsum(ordered_weights)
        cum_weights /= cum_weights[-1]
        gini = 1 - 2 * np.trapz(cum_claims, x=cum_weights)
        return gini

    gini = gini_coefficient(y_true, y_pred, exposure)

    # Create a dataframe for metrics
    metrics_df = pd.DataFrame(
        {
            "Bias": [bias],
            "Deviance": [deviance],
            "MAE": [mae],
            "RMSE": [rmse],
            "Gini": [gini],
        }
    )

    return metrics_df
