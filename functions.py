import pandas as pd
import numpy as np

def fill_NaN_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills empty columns with the values of the next non-empty column in the DataFrame.

    Args: 
        df: The DataFrame to fill

    Returns:
        Filled DataFrame
    """
    for index, column in enumerate(df.columns): 
        if df[column].isnull().all().all():
            i = 1
            ## find the next non-empty column
            while df.iloc[:,index+i].isnull().all():
                i += 1
            ## then copy
            df[column] = df.iloc[:,index+i]

    return df

def MSE(pred: np.ndarray, y: np.ndarray) -> float: 
    assert len(pred) == len(y)
    return (1/len(y)) * ((pred - y).T @ (pred - y))
