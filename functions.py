import pandas as pd
import numpy as np

def MSE(pred: np.ndarray, y: np.ndarray) -> float: 
    assert len(pred) == len(y)
    return (1/len(y)) * ((pred - y).T @ (pred - y))


## Defining a KNN imputer on entire columns:
def KNN(K: int, Sensor_ID: str, data_no_nan: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame: 
    """
    Returns a column with the average values from the K geometrically closest sensors.

    Args: 
        K: Number of neighbors to consider.
        Sensor_ID: The sensor we want to impute.
        data_no_nan: a copy of the data but where NaN values were removed
        coords: The dataframe containing the coordinate informations.

    Returns: 
        A column with the average values from the K geometrically closest sensors
    """

    ## making a disctionnary to easily change from sensor names to indices
    sensor_dic = {i:j for i,j in enumerate(coords["id"])}
    ## Selecting the row of the sensor we impute
    ## Caution: the columns have to be already renamed for "id" to be found !
    point = coords[coords["id"] == Sensor_ID]
    ## has to be numpy array for broadcasting
    point = point[["x","y","z"]].to_numpy() 

    ## taking the coordinates of all the other sensors
    all_others = coords[["x","y","z"]]

    ## computing the distance of the sensor to all others
    distances = np.sum((all_others - point)**2, axis = 1)

    ## sorting in ascending order
    distances = distances.sort_values()

    sensors = []
    i = 0
    while  len(sensors) < K:

        ## selecting the closest sensor (index 0 is the sensor itself at a distance 0)
        distance = distances.copy().to_list()[i+1] 
        
        ## Converting sensor index back to string
        ## the length of distances.copy().to_list() match with distances.shape[0] so the index is the right one
        assert len(distances.copy().to_list()) == distances.shape[0]
        sensor = sensor_dic[distances[distances == distance].index[0]]

        ## making sure we copy a sensor that has values
        if sensor in data_no_nan.columns: 
            sensors.append(sensor)

        i += 1

    ## selecting the columns associated to those sensors
    values = data_no_nan[sensors]
    
    return  1/K * np.sum(values, axis = 1) # return the average

def fill_NaN_columns(K: int, df: pd.DataFrame, imputer , coords: pd.DataFrame) -> pd.DataFrame:
    """
    Fills empty columns with the values of the average of the K geometrically closest sensors in the DataFrame.

    Args: 
        K: number of neighbors to consider
        Sensor_ID: The sensor to impute
        df: The DataFrame with the empty column to fill
        data_no_nan: a copy of the data but where NaN values were removed
        coords: The DataFrame containing coordinates of the sensors

    Returns:
        Filled DataFrame
    """
    data_no_nan = imputer.fit_transform(df)

    for index, sensor in enumerate(df.columns[1:]): 
        if df[sensor].isnull().all():
            df[sensor] = KNN(
                K = K,
                Sensor_ID = sensor,
                data_no_nan = data_no_nan, 
                coords = coords
            )

    return df

def get_batch_indices(n_samples, batch_size):
    """Get a list of tuples for each batch."""
    return [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]