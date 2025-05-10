# ML-project

**Group 20: [Stefan Mignerey](mailto:stefan.mignerey@epfl.ch) & [Giona Ghezzi](mailto:giona.ghezzi@epfl.ch)**

## Introduction

The goal of this project is to predict the temperature in time around a nuclear waste canister. To run to notebooks, we provide our conda environment as environment.txt. You can install it with the following command in the anaconda prompt: 

```conda create -n <environment-name> --file /<path>/environment.txt```
 
## Outline of the repository  

* backups: This folder contains a copy of the notebooks while the project evolved. They contain the date in the name with the following format: MMDD.  
* data: Contains the .csv files: the data and the submission.  
* environment.txt: Installation dependencies for the conda environment.  
* functions.py: Definition of some generic function to make the notebook more readable.  
* LICENSE: An MIT License in case we make our code public.
* logs_deimos.ipynb: Notebook used by Stefan to work on the project.
* logs_giona.ipynb: Notebook used by Giona to work on the project.
* logs.ipynb: Final notebook for submission.
* metrics.py: Loss metric copied from the [exercises](https://github.com/vita-epfl/IntroML-2025/blob/main/exercises/07-convnets/metrics.py).

## Progression of the project

1) Preparing the data  
   We started by extracting features from the data and to handle outliers. The extracted features can be found in the notebook with justifications. To handle the outliers we decided to look at the data distribution and delete values above/below a certain threshold visually determined. After this, to avoid deleting too many samples of data, we implemented two KNN imputers: 
    
    1) The first one imputing values for sensors with no values at all:
    It uses the average of the 5 spatially closest neighbors. 

    2) The second (...)


## Hyperparameter search

| Architecture | Activation | weight_decay | dropout | val. loss |
|--------------|------------|--------------|---------|-----------|
| [13,18,8,1]  | LReLU(0.1) | 1e-2         | 0.1     | 0.31      |
| [13,18,13,1] | LReLU(0.1) | 1e-2         | 0.0     | 0.30      |
| [13,18,13,1] | LReLU(0.1) | 5e-3         | 0.0     | 0.23      |          
| [13,18,13,1] | LReLU(0.1) | 2e-3         | 0.0     | 0.21      |
| [13,18,13,1] | LReLU(0.1) | 1e-3         | 0.0     | 0.22      |
| [13,18,13,1] | LReLU(0.1) | 8e-4         | 0.0     | 0.27      |
| [13,18,13,1] | LReLU(0.1) | 7e-4         | 0.0     | 0.33      |
| [13,18,13,1] | LReLU(0.1) | 5e-5         | 0.0     | 0.20      |
| [13,18,13,1] | LReLU(0.1) | 3e-4         | 0.0     | 0.41      |
| [13,18,13,1] | LReLU(0.1) | 1e-4         | 0.0     | 0.24      |

## To Do: 

⬜ Augment the data w.r.t time as high values are under-represented in the dataset.  
⬜ Making sure the model can overfitt small parts of dataset.   
✅ Choose reasonable thresholds for clipping.  
⬜ Use an L1 regularization to see if some features are useless (1st layer of weights).   
⬜ Use batchnorm between fully connected layers and activation functions.   
✅ Implement both KNN algorithms.  
✅ Search for a good weight_decay.  
⬜ Optimize the Neural Net's architecture.  
✅ Look for dropout.  (not improving results currently...)  
⬜ Search for the best number of neighbors in the KNN.  
✅ Search for a good learning rate.  
✅ Implement cross-validation.  
