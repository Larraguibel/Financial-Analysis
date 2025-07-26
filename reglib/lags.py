import pandas as pd


def lag_dataset(df: pd.DataFrame, window: int = 5, cols: list = None) -> pd.DataFrame:
    """
    Create lagged features for a given DataFrame.

    Args:
        df (pd.DataFrame): Pandas dataframe to create lags for.
        window (int, optional): Number of lags to create. Defaults to 5.
        cols (list, optional): Columns aimed to be lagged. Defaults to None implies all columns are lagged.

    Returns:
        pd.DataFrame: DataFrame containing the lagged features.
    """
    if cols is None:        
        cols = list(df.columns)
    
    X = pd.DataFrame(index=df.index)
    for i in range(1, window+1):
        for col_name in cols:
            X[f"{col_name}_lag_{i}"] = df[col_name].shift(i)
    X.dropna(inplace=True)
    return X