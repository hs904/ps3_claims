import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.
    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Define a helper function to hash the ID and get an integer representation
    #If the ID column is a string, it uses the hashlib library to create a hash, ensuring a stable and deterministic integer value.
    #If the ID column is numeric, it uses modulo 100 directly.
    def hash_id(value):
        if isinstance(value, str):
            return int(hashlib.sha256(value.encode()).hexdigest(), 16) % 100
        else:
            return value % 100
    # Apply the hash function to the specified ID column to get a split value
    df['split_value'] = df[id_column].apply(hash_id)
    # Assign 'train' or 'test' based on the split value and the training fraction
    df['sample'] = df['split_value'].apply(
        lambda x: 'train' if x < training_frac * 100 else 'test')
    # Drop the temporary 'split_value' column
    df = df.drop(columns=['split_value'])
    return df
