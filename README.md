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

1)  Data preprocessing:  

    1) Feature selection: we only got rid of the index column which was useless. We then 1-hot encoded material (categorical feature) to have a suitable representation for the model.  
    Our final features are {pressure, humidity, x, y, z, r, materials [1-hot], time [d]}.  

    2) Handling outliers:. To handle the outliers we decided to look at the data distribution and delete values above/below a certain threshold visually determined. After this, to avoid deleting too many samples of data, we implemented two KNN imputers: 
    
       * The first one imputing values for sensors with no values at all:
         It uses the average of the 5 spatially closest neighbors. 

       * The second one imputing sensors with partially missing values:  
         At a specific time, it takes the average between the next and prior time with data. We think that taking the average between two sensors will preserve the time patterns more. 

         We later replaced this method by the linear interpolation method of the pandas library as it was mork suited to our time dependant evolution. It is slower to run but shows slightly better, and more coherent results.  

        We then applied low and high filters to cut data that is too distant from the center of the distribution, this was done visually with values that seemed good. We also noticed that there was a distribution mismatch between the pressure validation/training datasets: different mean and variance which can make poor submissions as the model would have seen more different datas. We would try to adress that later.

    3) Data normalization: For the weights to be accordingly punished and the learning to converge faster we used a z-score scaling on our data. It was not optimal as the data was not following a normal distribution (it was more like a spike with long tails) but we thought it was better than a min-max scaling or a log scaling, which would have been more complicated to implement as we have negative data, but could have helped reduce the length of the tails in the data distribution.  


2)  Choosing the model:  
    We decided to go for a neural network as we thought it would be more suited to learn the complicated distribution of heat.  

    We also did some testing on a KNN (unsupervised) model, which performed well (scored 92.6 on Kaggle) with 5 neighbors. But we were not satisfied as unsupervised models required to make submissions to learn and submissions were limitted to 15/day. It would also learn patterns specifically related to this dataset and maybe not generalize to totally new data, for example times beyond 10'000. With a neural network we would hopefully not have that first problem as we could rely on the validation/test loss.  

    For the architecture, we initially set a 2 hidden-layers NN with LeakyReLU activation functions but it had bad performances. By having a 4 hidden-layers NN and slightly larger layers (~18 neurons), it improved, but not that much. Finally with a 4 hidden-layers net, with ReLU activation functions, a greater size ([14,80,30,1]) and some hyperparameter search we were able to have low losses: ~0.05 on the training set and ~0.13 on the validation set, which was 3 times better as our initial results.  

3)  Training:  
    With our relatively small dataset we decided to implement a cross validation. We noticed that the validation loss was varying a lot between each execution of the notebook. With a 3-fold cross-validation we could have a more precise view on the validation loss for our hyperparameter search. 

4)  Addressing the distribution mismatch:  
    When training our model, we noticed good performance on the trainning and validation set but when using our model for a prediction it gives poor results. We though that our training data under represented what the model would face in the prediction set so we wanted to fix that. We tried data augmentation so the high values of time would be more represented, but it accentuated the issue. We then tried removing 50% of the over-reprensented pressure values and their corresponding sensors. No success either. 

    Disabling both of those two things, scaling down the architecture to [14,20,20,1], with a dropout of 40% and relaxing the clipping values in the outliers handling (0 pressure, 100 temperature), we get back to reasonable scores of 120.  

    If re-enabling data augmentation, we score 70 and 120... There is a high variance between each iteration. It might mean that in the 7 out of 8 fold the model is trained on, there are some important data missing

    If we do more data augmentation (x2), we would replicate useful data and hopefuly minimize the noise even more. And increase the batch size to 10 (so it's faster), we get a worse score of 160. We can deduce that the data augmentation reiforced the noise instead of minimizing it. 

    I tried clipping pressure values below -1000 




## Some extra learning outcomes: 
*   By trying to use GPUs we learned that GPUs did not compute forward passes faster but could compute it in parallel on a single batch. It was useful only with large enough batch sizes.


## Hyperparameter search

| Architecture | Activation | weight_decay | dropout | val. loss |
|--------------|------------|--------------|---------|-----------|
| [14,18,8,1]  | LReLU(0.1) | 1e-2         | 0.1     | 0.31      |
| [14,18,13,1] | LReLU(0.1) | 1e-2         | 0.0     | 0.30      |
| [14,18,13,1] | LReLU(0.1) | 5e-3         | 0.0     | 0.23      |
| [14,18,13,1] | LReLU(0.1) | 2e-3         | 0.0     | 0.21      |
| [14,18,13,1] | LReLU(0.1) | 1e-3         | 0.0     | 0.22      |
| [14,18,13,1] | LReLU(0.1) | 8e-4         | 0.0     | 0.27      |
| [14,18,13,1] | LReLU(0.1) | 7e-4         | 0.0     | 0.33      |
| [14,18,13,1] | LReLU(0.1) | 5e-5         | 0.0     | 0.20 (pb) |
| [14,18,13,1] | LReLU(0.1) | 3e-4         | 0.0     | 0.41      |
| [14,18,13,1] | LReLU(0.1) | 1e-4         | 0.0     | 0.24      |
| [14,100,50,1]| ReLU       | 2e-4         | 0.0     | 0.14      |
| [14,80,30,1] | ReLU       | 2e-4         | 0.0     | 0.15      | (more stable)


Dropout in the range [0,30%] did not have any effect on small networks. 

## To Do: 

✅ Augment the data w.r.t time as high values are under-represented in the dataset.  
✅ Making sure the model can overfitt small parts of dataset. 
✅ Choose reasonable thresholds for clipping.  
⬜ Use an L1 regularization to see if some features are useless (1st layer of weights).   
✅ Use batchnorm between fully connected layers and activation functions.   
✅ Implement both KNN algorithms.  
✅ Search for a good weight_decay.  
⬜ Optimize the Neural Net's architecture.  
⬜ Look for dropout (again as the network size changed).  
⬜ Search for the best number of neighbors in the KNN (not very useful I think).  
✅ Search for a good learning rate.  
✅ Implement cross-validation.  
✅ Test bigger batch sizes (does well with 10).
<<<<<<< Updated upstream
⬜ See if the results improve when not touching the prediction set.
⬜ Find a solution for pressure_train and pressure_pred being very different. 
=======
✅ See if the results improve when not touching the prediction set. (NO)
⬜ Find a solution for pressure_train and pressure_pred being very different. 
⬜ Type of regularization. (maybe not important ?)  
✅ Find a good learning rate.   
⬜ Hyperparameter search in log scale. (???)  
✅ Impute with pandas linear method instead of 2-NN (doesnt work well as it doesnt know what to do on edges)  
⬜ Log scaling data  
⬜ Make the initial data into only one array instead of doing a dictionnary with time keys.  
>>>>>>> Stashed changes
