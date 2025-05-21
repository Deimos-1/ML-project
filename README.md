**Group 20: [Stefan Mignerey](mailto:stefan.mignerey@epfl.ch) (379891) & [Giona Ghezzi](mailto:giona.ghezzi@epfl.ch) (380057)**

# Predicting Temperature of Nuclear Waste Canisters

## 1. Introduction

In this project, we predict the transient temperatures around a nuclear waste canister. All notebooks run using the provided Conda environment. 


## 2. Environment Setup

You can install the Conda environment with the following command in the anaconda prompt: 

```bash
conda create -n <environment-name> --file /<path>/environment.txt
conda activate <environment-name>
```
 
## 3. Repository Structure

* **data/**: Contains the dataset in `.csv` files and an example of submission. 
* **imgs/**: Some images we saved.  
* **other\_models/**: This folder contains a copy of the notebooks while the project evolved. They contain the date in the name with the following format: MMDD. For them to work, you need to take them out of the folder, where train.ipynb is, or else the relative paths won't be correct. 
    * `logs_deimos_xxxx.ipynb`: Notebook used by Stefan to work on the project.  
    * `logs_giona_xxxx.ipynb`: Notebook used by Giona to work on the project.  
* **results/**: Results of the hyperparameter-search for the XGboost model and the Kaggle submission.
* **environment.txt**: Installation dependencies for the Conda environment.  
* **functions.py**: Definition of some generic functions to make the notebooks more legible.  
* **LICENSE**: An MIT License in case we make our code public.
* **metrics.py**: Loss metric copied from the [exercises](https://github.com/vita-epfl/IntroML-2025/blob/main/exercises/07-convnets/metrics.py).  
* **train.ipynb**: Final submission notebook.  
* **XGboost\_RandomSearch.ipynb**: Randomized hyperparameter search for the XGboost model. 

## **Progression of the project**


### 4. **Neural Network**

Our first idea was to use a neural network to predict the temperature. With various materials, pressure and humidity, we thought the temperature would be non-linear in those parameters. This is where the neural network shines, it can learn non-linear patterns. We thought it would be good for generalizing to the prediction data, but also on new timeframes, beyond the submission.  

#### 4.1. Data preprocessing

**Feature selection**: 
We dropped the index column, which was useless. We then 1-hot encoded material types (categorical feature) to have a suitable representation for the model.  
Our final features are `{pressure, humidity, x, y, z, r, materials [1-hot], time}`.  

**Handling outliers**: 
To handle the outliers we used clipping and deleted values above/below a certain threshold, visually determined, both on the training set and validation set. This is because we didn't want to predict on values that seemed obviously wrong and rather predict on imputed values.  
With the neural network, it was more effective because large inputs would more likely lead to incoherent results.  
But for the XGboost model, we later kept high values in the prediction set which improved our score to 7.1 after the Kaggle deadline.

**Handling missing values**:
After clipping, 10% of the sensors had no data in the training dataset. To avoid deleting the sensors that have partially missing data and to have some regularization, we implemented two KNN imputers: 

1.  A KNN imputer, using the average of the 5 spatially closest   neighbors to impute sensors with no data at all. 

2. The second one imputing sensors with partially missing values:  

At a specific time, it takes the average between the next and prior time with data.  
We later replaced this method by the linear interpolation method from the pandas library as it was more suited to our time dependant evolution. It is slower to run but shows slightly better, and more coherent results.  

### 4.2. Distribution mismatch

We supposed there was a distribution mismatch between the validation and training datasets of pressure: The data distribution of the training set doesn't represent well the validation data. We supposed that the model will not learn well this part of the data and instead be more influenced by training values of pressure close to 1'000 as they are dominant. We will try to verify that during the developement of the neural network. 
   
<br />  

![Figure 1](imgs\\pressure_train.png "Fig. 1") ![Figure 2](imgs\\pressure_pred.png "Fig. 2")


### 4.3. Normalization  
**Z-score scaling**: For the weights to be accordingly punished and the learning to converge faster (preconditioning), we used a z-score scaling on our data. It was not optimal as the data was not following a normal distribution, but we thought it was the best option.  
With a  min-max scaling, the axis would be too scaled up by the heavy tailed data and with a log scaling, it would have been more complicated as we have negative data, but it could have helped reduce the length of the tails in the data distribution.  

### 4.4. Architecture  
For the architecture, we initially set a 2 hidden-layers NN with LeakyReLU activation functions but it had bad performances. By having a 3 hidden-layers neural network and slightly larger layers (~18 neurons), it improved, but not that much.  
Finally with a 3 hidden-layers net, wider, with layer normalization and some hyperparameter search (dropout, learning_rate, L2 regularization strengh) we were able to have low losses: ~0.05 on the training set and ~0.13 on the validation set, which was 3 times better as our initial results.  

*see `/other_models/neural_net_0516.ipynb` for more details*

The layer normalization was helpful to avoid vanishing gradients or exploding gradients and contributed to better scores.  

We check that the model was capable of learning the data by training it on small batches. It reached very low losses: ~1e-3, so we assumed it was perfomant enough, we just had to be careful about ovefitting.  


### 4.3. Training  

**K-fold cross-validation**: (k = 8)  
With our relatively small dataset we decided to implement a k-fold cross validation to better evaluate the model's performance.  

We noticed that the validation loss was varying a lot between each execution of the notebook, here are some theories we explored: 

1. **Small batch size**:  
We initially had a stochastic gradient descent: batch size of 1. The data was not shuffled initially and every 32 samples the model trained on originated from the same timestamp<sup>[1](#footnote1)</sup>. The model could have adapted itself to each new batch without learning the bigger picture. Increasing the batch size to about 30, on shuffled data, made the model learning more stable. This was done in the lates stages of the model and reduced our scores in the 60-70 range on Kaggle.  

2. **Unrepresentative folds**:  
Another idea was that in the 7 folds the model trained on, the late timestamps were not represented and thus not learned. To address this we implemented a medium data augmentation, which helped. More data augmentation led to poorer results, increasing the noise instead of minimizing it.

<a name="footnote1">1</a>: *We initially had our data in a dictionnary as it was easier to code. Each key was a timestamp and we passed the keys and their corresponding data one by one into the model. We eventually grouped everything into one array*. 

### 4.4. Tuning the model

**Bad generalization**:  
When training our model, we noticed good performance on the trainning and validation set but bad scores around 130 on Kaggle. Here are some ideas we considered: 

1. **Distribution mismatch**:   
    As previously mentioned, we though that our training data under-represented what the model would face in the prediction set. We tried more data augmentation with respect to time, but it accentuated the issue. We then tried removing 50% of the over-reprensented pressure values and their corresponding sensors, but it did not improve the results. We concluded that the distribution mismatch was either not a problem, or that we incorrectly addressed it.  

2. **Overfitting**:   
    We also thought it could be some overfitting. But, with a batch size of 1, the validation loss kept decreasing during the training and stayed only 2 to 3 times higher than the validation loss. With a batch size of 30, both losses were almost identical, which eliminated the possibility of overfitting for us. This is also coherent as we had many regularization techniques: Data augmentation, dropout & L2 weight penalty.  

    Scaling down the architecture to [14,20,20,1], with a dropout of 40% and relaxing the clipping values in the outliers handling (0 pressure, 100 temperature), we got to scores of ~120 and sometimes lower, still a high variance between each execution of the notebook.  

3. **Bad imputation**:  
    Maybe our imputation changed the data too much, so we tried disabling the imputation and dropped all sensors containing NaN values. It led to low losses but a score of 400+ on Kaggle. We concluded that the imputation was great for regularization and we kept it.  

**Final NN Configuration**: 

* Layers: `[14,30,25,32,1]`   
* Dropout: [40%](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) 
* Layer norm + ReLU  
* Medium augmentation, 30-sample batch
* KNN imputer + linear interpolation imputation
* Outlier clipping: pressure <–1500; temperature >100°C
* Data normalization  
* Data shuffling  
* K-fold cross-validation

## 5. XGboost  
We were not satisfied of the neural network so we decided to test another model. After some discussion with the assistants in the exercise sessions, we were told that XGboost is often used in the industry and is very powerful, so we decided to test it. We implemented an XGboost model that got us a score of 23.05 with the following parameters:  

$$
\begin{cases}
    \text{estimators} & = 500\\
    \text{learning rate} & = 0.1\\  
    \text{max depth} & = 6\\
    \text{subsample} & = 0.8\\  
    \text{random state} & = 42\\ 
\end{cases}
$$

XGboost is generally better on small or medium, unbalanced, datasets like ours, which might explain this sudden improvement in the score.

We removed the data normalization as it is not needed here as the trees will find decision boundaries not affected by the scale of the data., it didn't change the results, logically, but reduced the execution time.  

### 5.1. Tuning  
We dropped the data augmentation as it showed better scores. We also improved the cplipping thresholds depending on the Kaggle scores we had. The pressure threshold was changed to -2'000 and the temperature threshold to 150°C. With a randomized hyperparameter search on about 150 combinations we ended up scoring 7.28 on Kaggle, our best score. 

## 6. KNN  
We also did some testing on a KNN (unsupervised) model while developing the neural net. It scored 81.5 on Kaggle with 7 neighbors, which was a little higher than the neural net's score.  
Were not satisfied as unsupervised models required to make submissions to learn, capped to 15/day. It would also learn patterns specifically related to this dataset and maybe not generalize to totally new data, for example times beyond 10'000. With a neural network we would hopefully not have that first problem as we could rely on the validation/test loss.  

## 7. Some lessons learned: 
*   By trying to use GPUs we learned that GPUs did not compute faster but could compute in parallel. It was useful only with large batch sizes that would be faster to process in parallel than sequentially. In our case, we couldn't benefit from GPUs.  

*   It would have been better to start off with different model types and to train them in parrallel, then select the best ones for a better hyperparameter search. We would have used XGboost sooner that way.  

*   Group all the data together in one array after feature engineering and rely on libraries if possible to handle the data afterwards. We would have had better results from the beginning and be able to easily implement different models.  

*   Write down every change, every test, and scores. The documentation about our neural net is foggy, with a lot of parameters and ways of approaching each aspect of the problem, is was overwhelming. We headed a bit in every direction and had difficulties finding what was important to work on and what wasn't. Eventually towards the end of the neural net, we realized this and it became easier to work on the project.

---

*More in-depth explanations can be found in the notebooks. Especially in **`train.ipynb`** and **`XGboost_RandomSearch.ipynb`***