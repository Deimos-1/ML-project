import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.ensemble import GradientBoostingRegressor

## --- FUNCTIONS ---
# In this file you will find the definition of important
# functions, this was done to improve the legibility of
# the notebook. 




def MSE(pred: np.ndarray, y: np.ndarray) -> float: 
    """
    Returns the mean squared error between two vectors.   
    WARNING: Do not use this function on large arrays !
    """
    assert len(pred) == len(y), f"Dimension mismatch between pred: {len(pred)} and y: {len(y)}"
    return (1/len(y)) * ((pred - y).T @ (pred - y))




## Defining a KNN imputer on entire columns:
def KNN(K: int, Sensor_ID: str, data_no_nan: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame: 
    """
    Returns a column with the average values from the K geometrically closest sensors.

    Parameters 
    ----------
    K : Number of neighbors to consider.  
    Sensor_ID : The sensor we want to impute.  
    data_no_nan : A copy of the data but where NaN values were removed.  
    coords : The dataframe containing the coordinate informations. Columns have to be renamed.  

    Returns
    -------
    A column with the average values from the K geometrically closest sensors
    """

    ## making a dictionnary to easily change from sensor names to indices
    sensor_dic = {coords.index[coords["id"] == i][0] :i for i in coords["id"]}

    ## Selecting the row of the sensor we impute
    ## Caution: the columns have to be already renamed for "id" to be found !
    point = coords[coords["id"] == Sensor_ID]

    ## making it numpy array for broadcasting
    point = point[["x","y","z"]].to_numpy() 

    ## taking the coordinates of all the other sensors
    all_others = coords[["x","y","z"]]

    ## computing the distance of the sensor to all others
    distances = np.sum((all_others - point)**2, axis = 1)

    ## sorting in ascending order, top values are the closest
    distances = distances.sort_values()

    sensors = []
    i = 0

    ## Looping until we have selected the K nearest neighbors
    while  len(sensors) < K:

        ## selecting the closest sensor (index 0 is the sensor itself at a distance 0)
        distance = distances.copy().to_list()[i+1] 
        
        ## Converting sensor index back to string
        sensor = sensor_dic[distances[distances == distance].index[0]]

        ## making sure we copy a sensor that has values
        if sensor in data_no_nan.columns: 
            sensors.append(sensor)

        i += 1

    ## selecting the columns associated to those sensors
    values = data_no_nan[sensors]
    
    return  1/K * np.sum(values, axis = 1)




def fill_NaN_columns(K: int, df: pd.DataFrame, imputer , coords: pd.DataFrame) -> pd.DataFrame:
    """
    Fills empty columns with the values of the average of the K geometrically closest sensors in the DataFrame.

    Parameters
    ----------
    K : Number of neighbors to consider.
    df : The DataFrame with the empty columns to fill.
    imputer : The imputer used to remove empty columns.
    coords : The DataFrame containing coordinates of the sensors.

    Returns
    -------
    Filled DataFrame
    """
    data_no_nan = imputer.fit_transform(df)

    for sensor in df.columns[1:]: 
        if df[sensor].isnull().all():
            df[sensor] = KNN(
                K = K,
                Sensor_ID = sensor,
                data_no_nan = data_no_nan, 
                coords = coords
            )

    return df




def get_batch_indices(n_samples, batch_size) -> List[Tuple[int, int]]:
    """Get a list of tuples for each batch."""
    return [(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]




def get_train_data(
        pressure_train: pd.DataFrame,
        humidity_train: pd.DataFrame,
        temperature_train: pd.DataFrame,
        coordinates_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group all the data into one array that can be fed into the model.

    Parameters
    ---------
    pressure_train : The training data for the pressure.
    humidity_train : The training data for the humidity.
    temperature_train : The temperature labels for training.
    coordinates_train : The coordinates and materials for the training data.

    Returns
    -------
    X : Array of examples with each column being a feature.
    y : Labels, rows matching with the ones of X.
    """
    time = pressure_train["M.Time[d]"].to_list()

    ## We will create a dictionary with each timestamp and then stack each entry into one array
    X_t: dict = {}
    y_t: dict = {}

    for i, t in enumerate(pressure_train['M.Time[d]']): 
        
        ## First, select the pressure data at the time:
        P = pressure_train.iloc[i, 1:]

        ## Then the humidity:
        H = humidity_train.iloc[i,1:]

        ## Then transpose those as we need the sensors as rows, not columns:
        P = P.T
        H = H.T

        ## We need to reshape them to concatenate well (n_sensors, 1):   
        P = np.reshape(P, (P.shape[0], 1)) 
        H = np.reshape(H, (H.shape[0], 1))

        ## Take coordinates and material features:
        coords = coordinates_train.iloc[:,1:].to_numpy()

        ## Adding the time as a feature in the last row:
        X_i = np.concatenate([P, H, coords, np.ones((P.shape[0], 1))*t], axis = 1)

        ## Add it to the dictionnary
        X_t[t] = X_i

    for i, t in enumerate(time): 
        ## for each time we have the solution of the 900 sensors in a vector
        ## selecting row i corresponding to time t
        y = temperature_train.iloc[i, 1:]
        y_t[t] = y.to_numpy()

    ## grouping all the data in one array: 
    X = np.vstack([matrix for matrix in X_t.values()])
    y = np.vstack([vector.reshape((vector.shape[0], 1)) for vector in y_t.values()])

    assert X.shape[0] == y.shape[0], f"Mismatch in the number of samples... \n- size of X: {X.shape}\n- size of y: {y.shape}"

    ## --- OLD CODE FOR THE NEURAL NET ---

    ## Shuffling the rows so the data is well-mixed
    ## Else the model could have an unrepresentative timestamp in one epoch and mess the learning.
    ## After some testing, it improves our results and make the results more stable.
    ## First, generate a permutation of indices:
    # permutation = np.random.permutation(X.shape[0])
    ## Then shuffle both X and y using the same permutation, unfortunately we cannot impose a random_state on this method.
    # X = X[permutation]
    # y = y[permutation]

    ## --- NO NEED TO SHUFFLE WHEN USING TRAIN_TEST_SPLIT() ---

    return X, y




def predict(
        model: GradientBoostingRegressor,
        pressure_pred: pd.DataFrame,
        humidity_pred: pd.DataFrame,
        coordinates_pred: pd.DataFrame
) -> pd.DataFrame:
    """
    Make a prediction on the prediction dataset. Returns a DataFrame that is also saved to a csv file:  
    relative path: /results/submission.csv

    Parameters
    ----------
    model : Trained XGboost model.  
    pressure_pred : Pressure data for the prediction.  
    humidity_pred : Humidity data for the prediction.  
    coordinates_pred : Coordinates of the sensors and materials.  

    Returns
    -------
    pred : The prediction as a DataFrame. 
    """ 
    X_t_pred = {}

    ## Initialize the submission DataFrame:
    sensor_IDs = pressure_pred.columns[1:]
    pred = pd.DataFrame(data=sensor_IDs, columns=["id"])

    for i, t in enumerate(pressure_pred['M.Time[d]']): 
        # Extract pressure and humidity data at the specific time `t`
        P = pressure_pred.iloc[i, 1:]
        H = humidity_pred.iloc[i, 1:]

        # Reshape to (n_sensors, 1) to maintain consistency
        P = np.reshape(P.values, (P.shape[0], 1))
        H = np.reshape(H.values, (H.shape[0], 1))

        # Extract coordinates and material features
        coords = coordinates_pred.iloc[:, 1:].to_numpy()

        # Combine all features including time:
        X = np.concatenate([P, H, coords, np.ones((P.shape[0], 1)) * t], axis=1)
        
        # Store the data for the given timestamp
        X_t_pred[t] = X

    # Iterate through each timestamp to predict and store the results
    for t in pressure_pred["M.Time[d]"]:
        X = X_t_pred[t]

        # Predict using the trained Gradient Boosting Regressor model
        y_pred = model.predict(X)

        # Add predictions to the dataframe
        pred[str(int(t))] = y_pred

    ## Save the prediction in a .csv file that can be submitted on Kaggle.
    pred.to_csv("results\\submission.csv", index=False)
    print("Submission file saved as 'submission.csv'")

    return pred