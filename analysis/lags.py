import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


def lag_dataset(df: pd.DataFrame, lags: int = 5, cols: list = None) -> pd.DataFrame:
    """
    Create lagged features for a given DataFrame.

    Args:
        df (pd.DataFrame): Pandas dataframe to create lags for.
        lags (int, optional): Number of lags to create. Defaults to 5.
        cols (list, optional): Columns aimed to be lagged. Defaults to None implies all columns are lagged.

    Returns:
        pd.DataFrame: DataFrame containing the lagged features.
    """
    if cols is None:        
        cols = list(df.columns)
    
    X = pd.DataFrame(index=df.index)
    for i in range(1, lags + 1):
        for col_name in cols:
            X[f"{col_name}_lag_{i}"] = df[col_name].shift(i)
    X.dropna(inplace=True)
    return X

def build_ar_lag_matrix(series: np.ndarray, p: int):
    """
    Build lagged design matrix X and target y.
    """
    N = len(series)
    if N <= p:
        raise ValueError("Series too short for the requested number of lags.")
    X = np.column_stack([series[p-k-1:N-k-1] for k in range(p)])
    y = series[p:]
    return X, y

def recursive_forecast(model: LinearRegression, X_train: pd.DataFrame, lags: int = 5) -> pd.Series:
    last_feats = [X_train.iloc[-i].values.reshape(1, -1) for i in lags]