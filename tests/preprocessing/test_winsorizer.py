import numpy as np
import pytest
from ps3.preprocessing import Winsorizer

# Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", 
    [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    # Generate random normal data
    X = np.random.normal(0, 1, 1000)
    
    # Initialize Winsorizer
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    
    # Fit the Winsorizer
    winsorizer.fit(X)
    
    # Transform the data
    X_transformed = winsorizer.transform(X)
    
    # Assert that no value is below the lower quantile threshold
    if lower_quantile > 0:
        assert np.all(X_transformed >= np.percentile(X, lower_quantile * 100)), \
            "Values below lower quantile not clipped!"
    
    # Assert that no value is above the upper quantile threshold
    if upper_quantile < 1:
        assert np.all(X_transformed <= np.percentile(X, upper_quantile * 100)), \
            "Values above upper quantile not clipped!"
