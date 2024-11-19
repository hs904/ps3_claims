import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Initialize the Winsorizer with default quantile values.

        Parameters:
        lower_quantile (float): Lower quantile threshold (default is 5th percentile).
        upper_quantile (float): Upper quantile threshold (default is 95th percentile).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the quantiles for clipping and store them.

        Parameters:
        X (array-like): The input data to calculate quantiles.
        y: Ignored, not used in this transformer.

        Returns:
        self: Fitted transformer.
        """
        X = np.asarray(X)
        self.lower_quantile_ = np.percentile(X, self.lower_quantile * 100, axis=0)
        self.upper_quantile_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        """
        Clip the data using the computed quantiles.

        Parameters:
        X (array-like): The input data to transform.

        Returns:
        X_clipped (array-like): The transformed data with values clipped to quantile range.
        """
        check_is_fitted(self, ['lower_quantile_', 'upper_quantile_'])
        X = np.asarray(X)
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X_clipped
