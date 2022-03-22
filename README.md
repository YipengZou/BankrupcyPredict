# BankrupcyPredict

## Background

  This project comes from CUHKSZ FTE4560. The purpose of this project is to construct different machine learning models to predict whether a company will go bankruptcy or not.

## Model Summary

​	The performance of our model is relatively good. Precision / Recall / F1-score are calculated using weighted average, from `sklearn.metrics.classficiation_report`. AUC here is calculated using `sklearn.metrics.roc_auc_rate`. The result is:

| Model           | Accuracy | Precision | Recall  | AUC        | F1-score | Training Time | Developer   |
| --------------- | -------- | --------- | ------- | ---------- | -------- | ------------- | ----------- |
| **KNN**         | 75%      | 75%       | 75%     | 72.00%     | 0.72     | NA            | Ziqi Liao   |
| **LS**          | 72%      | 76%       | 72%     | 79.67%     | 0.66     | 1.2s          | Wenzhong Xu |
| **LS Gaussian** | 79%      | **80%**   | 73%     | 76.37%     | 0.73     | 8.6s          | Yipeng Zou  |
| **LDA**         | 73%      | 76%       | 75%     | 76.00%     | 0.76     | 0.002s        | Ziqi Liao   |
| **Logistic**    | 74%      | 79%       | 74%     | 79.61%     | 0.69     | 9.1s          | Wenzhong Xu |
| **Softmax**     | 75%      | 76%       | 63%     | 74.31%     | 0.69     | 17m37s        | Yipeng Zou  |
| **Tree**        | 79%      | 78%       | **79%** | 81.01%     | **0.78** | **0.06s**     | Ziqi Liao   |
| **RF**          | **80%**  | 78%       | 77%     | **87.17%** | **0.78** | 12m11s        | Yipeng Zou  |
| NN              | 77%      | 77%       | 77%     | 80.37%     | 0.77     | 2.2s          | Wenzhong Xu |

## Data resource

​	Our data is about polish bankruptcies. Here are the data sources. 

[	**Polish companies bankruptcy data Data Set**](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

​	We used only a subset of the total data. We have 64 features, All features are continuous . And its labels are 0 and 1, with 1 representing bankrupt and 0 representing unbankrupt. Data has been shred into training datasets (949 samples) and test datasets (151 samples). The origin data and cleaned data we used in this project is [here](https://github.com/YipengZou/BankrupcyPredict/tree/main/data). 

## Code

​	In the preprocessing process, we use `R` code to do some data cleaning work. For the machine learning model part, we use `python` to do the training. All the codes are written in `Jupyter Notebook`. The running environment is `3.8.12`. 

------

# Content

- [BankrupcyPredict](#bankrupcypredict)
  * [Background](#background)
  * [Model Summary](#model-summary)
  * [Data resource](#data-resource)
  * [Code](#code)
- [Content](#content)
  * [Data Exploring](#data-exploring)
    + [Unbalanced data](#unbalanced-data)
    + [Features Distribution](#features-distribution)
    + [Correlation of features](#correlation-of-features)
    + [Missing Values](#missing-values)
  * [Data Preprocess](#data-preprocess)
    + [Handling Missing Values](#handling-missing-values)
    + [Adjusting Data](#adjusting-data)
      - [Exclude Outliers](#exclude-outliers)
      - [Data Scale Down](#data-scale-down)
      - [PCA Dimension Reduction](#pca-dimension-reduction)
      - [Result of preprocess](#result-of-preprocess)
  * [Model Summary](#model-summary-1)
    + [Performance Summary](#performance-summary)
  * [Build Classification Model](#build-classification-model)
    + [KNN Classifier Model](#knn-classifier-model)
      - [Overview of KNN Classifier Model](#overview-of-knn-classifier-model)
      - [Result Analysis](#result-analysis)
    + [Least Square Model](#least-square-model)
      - [Least square without regularization](#least-square-without-regularization)
      - [Least square with L2 regularization](#least-square-with-l2-regularization)
      - [The predictions on the training set: without and with $\lambda = 2.1$](#the-predictions-on-the-training-set--without-and-with---lambda---21-)
    + [LS with Gaussian Basis Model](#ls-with-gaussian-basis-model)
      - [How to Choose Appropriate $\mu$ and $\sigma^2$](#how-to-choose-appropriate---mu--and---sigma-2-)
        * [K-means Clustering](#k-means-clustering)
        * [Random Generate](#random-generate)
        * [Choose an appropriate $\sigma^2$](#choose-an-appropriate---sigma-2-)
      - [Using Gaussian Basis Functions to Transform Data](#using-gaussian-basis-functions-to-transform-data)
      - [Calculate Parameters](#calculate-parameters)
      - [Predict Result](#predict-result)
      - [Result Analysis](#result-analysis-1)
    + [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis--lda-)
      - [Overview of LDA](#overview-of-lda)
      - [Result Analysis](#result-analysis-2)
    + [Logistic classification with/without regularization (L2).](#logistic-classification-with-without-regularization--l2-)
      - [Logistic classification without regularization](#logistic-classification-without-regularization)
      - [Logistic classification with regularization](#logistic-classification-with-regularization)
    + [SoftMax Classification Model](#softmax-classification-model)
      - [Overview of SoftMax Model](#overview-of-softmax-model)
      - [Use Gradient Descent to Find Parameters:](#use-gradient-descent-to-find-parameters-)
      - [Result Analysis](#result-analysis-3)
      - [Code of SoftMax method](#code-of-softmax-method)
    + [Decision Tree](#decision-tree)
      - [Overview of Decision Tree](#overview-of-decision-tree)
      - [Result Analysis](#result-analysis-4)
    + [Random Forest Classification Model](#random-forest-classification-model)
      - [Overview of Random Forest Model](#overview-of-random-forest-model)
      - [Choose Hyper-parameters](#choose-hyper-parameters)
        * [Determine maximum feature used in each tree](#determine-maximum-feature-used-in-each-tree)
        * [Choose Criterion](#choose-criterion)
        * [Choose number of maximum decision trees in the model](#choose-number-of-maximum-decision-trees-in-the-model)
        * [Choose the maximum depth of each decision tree](#choose-the-maximum-depth-of-each-decision-tree)
        * [Choose of threshold](#choose-of-threshold)
        * [Choose of Hyper-parameters](#choose-of-hyper-parameters)
      - [Result Evaluation](#result-evaluation)
    + [Neural Network](#neural-network)
      - [selection of layers and units](#selection-of-layers-and-units)
      - [Selection of optimizer](#selection-of-optimizer)
      - [Select the callbacks, validation split rate and activation function](#select-the-callbacks--validation-split-rate-and-activation-function)
      - [Select the final parameters](#select-the-final-parameters)
      - [Add regularization (set seed 80)](#add-regularization--set-seed-80-)
      - [L1 and L2 regularization](#l1-and-l2-regularization)
      - [Dropout](#dropout)
  * [Conclusion](#conclusion)
    + [Comparison of Models](#comparison-of-models)
    + [Pros and Cons of each Model](#pros-and-cons-of-each-model)
  * [Appendix](#appendix)
    + [References:](#references-)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Data Exploring

### Unbalanced data

​	The pie chart shows that the data is very uneven, and it is worth noting that the training set and test set are not evenly distributed.  

<img src="https://s2.loli.net/2022/03/22/Yt9XHqlUKfxWEBg.png" alt="proportion" style="zoom: 50%;" />

### Features Distribution

​	All our features are continuous, so we can make histogram to check the training set distribution of each feature, as shown below:

![distribution before](https://s2.loli.net/2022/03/22/9ODTNJadh46YrWP.png)

​	From the histogram, we can find some information:

- The distribution of most features is extremely uneven and there are a lot of outliers.
- The magnitude of the features are inconsistent.  This needs to be standardized or normalized.

​	These problems will be solved in the pre-processing stage.

### Correlation of features

​	The extreme correlation of features leads to multicollinearity, which reduces the effect of some models.  We used a thermal map to check the correlation of features(training set):

![heatmap](https://s2.loli.net/2022/03/22/9N5atXfcsLwOgRP.png)

​	Only pairs with Pearson correlation coefficients greater than 0.7 (highly correlated) are shown here, and we can see that the data has serious correlation problems.  

​	This will be resolved during data preprocessing.

### Missing Values

​	In the given dataset, the missing values are represented as "?". Thus, we need to replace this symbol with `nan` so that the computer can recognize the missing values. After that, we count the number of the missing values for each parameter. Here are the results of the Training set.

![missing value](https://s2.loli.net/2022/03/22/GpiB5lUbTIYsROL.png)

​	From the above histogram(left), we found that most attributes (N=61) have less than 5% missing values. Only `Attr37`, `Attr27`, `Attr45` have more than 5% missing values. 

​	The figure on the right shows the missing pattern. The yellow features in each row represent the patterns that these features are missing at the same time, and the corresponding histogram to the right of each row represents the proportion of this pattern in our data

​	Therefore, the problem of missing values is also serious, and we will deal with the missing values in the pre-processing stage.

## Data Preprocess

### Handling Missing Values

​		The first step to clean the data is to handle the missing values. Due to the high proportion of partial features missing, it is no longer suitable to use median or average to fill our data. We tried **MICE** algorithm, which has the characteristics of universality and rapidity, and has a good effect for all kinds of missing value ratio. 

The following are the core principles of the MICE algorithm:

![MICE](https://s2.loli.net/2022/03/22/qx8ab3cdyE6BLiP.png)

This process continues until all specified variables are interpolated.

We use the MICE package in R to implement the algorithm on the feature data of training set. With these three features(`Attr37`, `Attr27`, `Attr45`) with the highest proportion, we can view the filling results(training set).

![MICE result1](https://s2.loli.net/2022/03/22/2mo1EPqBcfpZ7Uv.png)

![MICE result2](https://s2.loli.net/2022/03/22/8cYoZSPMm1iACGl.png)

We can see filling results and perfect matching of raw data, which gives us more confidence in MICE algorithm.

Finally, we take the filling results in the training set as the real value, then combine the training set and test set, and fill the training set with the whole feature data.

### Adjusting Data

#### Exclude Outliers

​	By the law of large number, we assume the values of our attributes follow the normal distribution. Thus, the range of the value of each attributes should be in $[\mu-3\sigma,\ \mu+3\sigma]$ by the **68-95-99.7 Rule**. 

<img src="https://s2.loli.net/2022/03/22/FSMqR6hK7WrLfCT.png" style="zoom: 33%;" />

Thus, we will denote the data point as outlier if it is out of the range $[\mu-3\sigma,\ \mu+3\sigma]$, and set the value of the outlier as the threshold $\mu-3\sigma$ or $\mu+3\sigma$. The code to exclude outliers is:

```python
def outsideAdj(self):
    mu = self.train_x.mean()
    sigma = np.sqrt(self.train_x.var())
    upper_bound = mu + 1 * sigma
    lower_bound = mu - 1 * sigma
    for attr in lower_bound.index:
        self.train_x[attr][self.train_x[attr] > upper_bound[attr]] = upper_bound[attr]
        self.test_x[attr][self.test_x[attr] > upper_bound[attr]] = upper_bound[attr]
        self.train_x[attr][self.train_x[attr] < lower_bound[attr]] = lower_bound[attr]
        self.test_x[attr][self.test_x[attr] < lower_bound[attr]] = lower_bound[attr]
    self.train_x = (self.train_x - self.train_x.min()) / (self.train_x.max() - self.train_x.min())*10
    self.test_x = (self.test_x - self.test_x.min()) / (self.test_x.max() - self.test_x.min())*10
```

#### Data Scale Down

​	In some models like `Least Square with Gaussian Basis` or `SoftMax`, we will do exponential transformation to the original data. If we just put the original data to do the transformation, we may encounter the problem that the transformation results are too large or too small. Therefore, we have to scale down the origin data to scale down it and reduce the variance. We will do two steps of the transformation:

1. Take square root of each data.
2. Do the Min-Max transformation.

#### PCA Dimension Reduction

​	Due to the strong correlation of the data, we considered using Principal Component Analysis(PCA) to reduce the dimension of the data. 

​	Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.

​	We implemented PCA using Python's Sklearn package, which preserved 97% of the information and ultimately reduced the dimensions of the 64-dimensional features to 24-dimensional. 

​	The figure below shows the proportion of interpretation information corresponding to the top ten new features:

<img src="https://s2.loli.net/2022/03/22/UgmcrKw2tqLGSXv.png" alt="PCA expained_ratio" style="zoom: 50%;" />

#### Result of preprocess

As shown in the figure below, after data preprocessing, we solved outlier, magnitude and correlation problems (because each feature generated by PCA has no correlation).

![image-20220321204017404](https://s2.loli.net/2022/03/22/MOHKC196PSkymXJ.png)

## Model Summary

### Performance Summary

We use weighted average to calculate accuracy, precision, recall and F1-score.

| Model           | Accuracy | Precision | Recall  | AUC        | F1-score | Training Time | Developer   |
| --------------- | -------- | --------- | ------- | ---------- | -------- | ------------- | ----------- |
| **KNN**         | 75%      | 75%       | 75%     | 72.00%     | 0.72     | NA            | Ziqi Liao   |
| **LS**          | 72%      | 76%       | 72%     | 79.67%     | 0.66     | 1.2s          | Wenzhong Xu |
| **LS Gaussian** | 79%      | **80%**   | 73%     | 76.37%     | 0.73     | 8.6s          | Yipeng Zou  |
| **LDA**         | 73%      | 76%       | 75%     | 76.00%     | 0.76     | 0.002s        | Ziqi Liao   |
| **Logistic**    | 74%      | 79%       | 74%     | 79.61%     | 0.69     | 9.1s          | Wenzhong Xu |
| **Softmax**     | 75%      | 76%       | 63%     | 74.31%     | 0.69     | 17m37s        | Yipeng Zou  |
| **Tree**        | 79%      | 78%       | **79%** | 81.01%     | **0.78** | **0.06s**     | Ziqi Liao   |
| **RF**          | **80%**  | 78%       | 77%     | **87.17%** | **0.78** | 12m11s        | Yipeng Zou  |
| NN              | 77%      | 77%       | 77%     | 80.37%     | 0.77     | 2.2s          | Wenzhong Xu |

## Build Classification Model

### KNN Classifier Model

#### Overview of KNN Classifier Model

​	K-Nearest Neighbour(KNN) is one of the simplest Machine Learning algorithms based on Supervised Learning technique. It assumes the similarity between the new data and available data and put the new data into the class that is most similar to the available class.

​	When we build a KNN model, we need save the train features and train labels. It should be noted that, since Euclidean distance is calculated during KNN model prediction, our data need to be standardized. When we get a new data we need to do the following steps:

- **Step-1**: Select the number K of the neighbors and threshold.
- **Step-2**: Calculate the Euclidean distance between features of new data and all train features.
- **Step-3**: Take the K nearest neighbors as per the calculated Euclidean distance.
- **Step-4**: Calculate the proportion of k samples that are class 2, which is score of class 2 for this data
- **Step-5**: If its score is greater than the threshold, we classify it as class 2

<img src="https://s2.loli.net/2022/03/22/iJxwO7CuTYsyPgj.png" alt="image-20220321204349486" style="zoom: 50%;" />

The implementation of KNN is not difficult. Here is our code:

```python
class knn:
    def __init__(self,k):
        assert 1<=k
        self.k = k
        self.X = None
        self.y = None
    
    def fit(self,train_X,train_y):
        self.X = train_X
        self.y = train_y
        
    def predict(self,test_X):
        n = test_X.shape[0]
        result = np.ones(n)
        for i in range(n):
            new_X = test_X.iloc[i,:]
            nn = (np.argsort(((self.X - new_X)**2).sum(1)))[:self.k]
            result[i] = self.y.iloc[nn,].mean()
        return result
```

#### Result Analysis

Now, we use KNN in our pre-processed data. Due to unbalanced data, we take AUC, precision rate, recall rate and f1-score as the evaluation criteria.  In our data, the following is the change of AUC, precision rate, recall rate and f1-score in test data when K goes from 1 to 30 (threshold = 0.3 for reasons that will be explained later).

<img src="https://s2.loli.net/2022/03/22/q1bTsNg7cAPu26n.png" alt="knn compa" style="zoom:50%;" />

We find that AUC reaches its maximum 0.7224 and precision rate, recall rate and f1-score were also close to the highest values when k is 16. So we chose k=16 as the parameter of the final model. The following is the result of KNN on the test set when K =16 threshold = 0.5.

| Testing threshold = 0.5 | Precision | Recall | F1-score | Support |
| ----------------------- | --------- | ------ | -------- | ------- |
| Class 1                 | 0.70      | 0.98   | 0.81     | 100     |
| Class 2                 | 0.80      | 0.16   | 0.26     | 51      |
| Accuracy                |           |        | 0.70     | 151     |
| Macro Avg               | 0.75      | 0.57   | 0.54     | 151     |
| Weighted Avg            | 0.73      | 0.70   | 0.63     | 151     |
| AUC                     |           |        | 0.72     | 151     |

We found that the results were mediocre, especially for class 1, whose recall rate was low at 0.16, and whose AUC was relatively high, so we might be able to change the threshold to improve the predicted performance. The following is the result of KNN on the test set when K =16 threshold = 0.3.

| Testing threshold = 0.3 | Precision | Recall | F1-score | Support |
| ----------------------- | --------- | ------ | -------- | ------- |
| Class 1                 | 0.80      | 0.82   | 0.81     | 100     |
| Class 2                 | 0.63      | 0.61   | 0.62     | 51      |
| Accuracy                |           |        | 0.75     | 151     |
| Macro Avg               | 0.72      | 0.71   | 0.72     | 151     |
| Weighted Avg            | 0.75      | 0.75   | 0.65     | 151     |
| AUC                     |           |        | 0.72     | 151     |

It can be found that there is a great improvement in the model. In addition to the slightly reduced precision of Class 2 and recall of Class 1, other indicators, especially the recall of Class 2, have been greatly improved.

Let's look at the performance of the model on the training set when K =16 threshold = 0.3：

| Training threshold = 0.3 | Precision | Recall | F1-score | Support |
| ------------------------ | --------- | ------ | -------- | ------- |
| Class 1                  | 0.80      | 0.90   | 0.85     | 100     |
| Class 2                  | 0.74      | 0.55   | 0.63     | 51      |
| Accuracy                 |           |        | 0.78     | 151     |
| Macro Avg                | 0.77      | 0.72   | 0.74     | 151     |
| Weighted Avg             | 0.78      | 0.78   | 0.77     | 151     |
| AUC                      |           |        | 0.79     | 151     |

Here is the ROC curve：

<img src="https://s2.loli.net/2022/03/22/z2lSmaPyINdUCOG.png" alt="knn ROC" style="zoom: 67%;" />

In order to check the influence of different k on model performance, we also use k=1,3,5,10,15 as comparison values. Their results are as follows (a good threshold is selected for different k values).

| K =1 threshold is invalid | Precision | Recall | F1-score | Support |
| ------------------------- | --------- | ------ | -------- | ------- |
| Class 1                   | 0.68      | 0.77   | 0.72     | 100     |
| Class 2                   | 0.39      | 0.29   | 0.34     | 51      |
| Accuracy                  |           |        | 0.61     | 151     |
| Macro Avg                 | 0.54      | 0.53   | 0.53     | 151     |
| Weighted Avg              | 0.58      | 0.61   | 0.59     | 151     |
| AUC                       |           |        | 0.53     | 151     |

| K =3 threshold = 0.33 | Precision | Recall | F1-score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Class 1               | 0.76      | 0.60   | 0.67     | 100     |
| Class 2               | 0.44      | 0.63   | 0.52     | 51      |
| Accuracy              |           |        | 0.61     | 151     |
| Macro Avg             | 0.60      | 0.61   | 0.60     | 151     |
| Weighted Avg          | 0.65      | 0.61   | 0.62     | 151     |
| AUC                   |           |        | 0.63     | 151     |

| K =5 threshold = 0.3 | Precision | Recall | F1-score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| Class 1              | 0.74      | 0.74   | 0.74     | 100     |
| Class 2              | 0.49      | 0.49   | 0.49     | 51      |
| Accuracy             |           |        | 0.66     | 151     |
| Macro Avg            | 0.62      | 0.62   | 0.62     | 151     |
| Weighted Avg         | 0.65      | 0.66   | 0.66     | 151     |
| AUC                  |           |        | 0.64     | 151     |

| K =10 threshold = 0.3 | Precision | Recall | F1-score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Class 1               | 0.76      | 0.83   | 0.79     | 100     |
| Class 2               | 0.60      | 0.49   | 0.54     | 51      |
| Accuracy              |           |        | 0.72     | 151     |
| Macro Avg             | 0.68      | 0.66   | 0.67     | 151     |
| Weighted Avg          | 0.71      | 0.72   | 0.71     | 151     |
| AUC                   |           |        | 0.70     | 151     |

| K =15 threshold = 0.33 | Precision | Recall | F1-score | Support |
| ---------------------- | --------- | ------ | -------- | ------- |
| Class 1                | 0.79      | 0.83   | 0.81     | 100     |
| Class 2                | 0.63      | 0.57   | 0.60     | 51      |
| Accuracy               |           |        | 0.74     | 151     |
| Macro Avg              | 0.71      | 0.70   | 0.70     | 151     |
| Weighted Avg           | 0.74      | 0.74   | 0.74     | 151     |
| AUC                    |           |        | 0.72     | 151     |

| K =17 threshold = 0.33 | Precision | Recall | F1-score | Support |
| ---------------------- | --------- | ------ | -------- | ------- |
| Class 1                | 0.78      | 0.88   | 0.83     | 100     |
| Class 2                | 0.68      | 0.51   | 0.58     | 51      |
| Accuracy               |           |        | 0.75     | 151     |
| Macro Avg              | 0.73      | 0.69   | 0.71     | 151     |
| Weighted Avg           | 0.75      | 0.75   | 0.74     | 151     |
| AUC                    |           |        | 0.72     | 151     |

| K =20 threshold = 0.33 | Precision | Recall | F1-score | Support |
| ---------------------- | --------- | ------ | -------- | ------- |
| Class 1                | 0.77      | 0.89   | 0.82     | 100     |
| Class 2                | 0.69      | 0.47   | 0.56     | 51      |
| Accuracy               |           |        | 0.75     | 151     |
| Macro Avg              | 0.73      | 0.68   | 0.69     | 151     |
| Weighted Avg           | 0.74      | 0.75   | 0.73     | 151     |
| AUC                    |           |        | 0.71     | 151     |

It is not hard to see that as K gets closer and closer to 16, the overall performance of the model gets better and better, which is consistent with our expectations. 

In addition, it is worth noting that the better threshold value is basically 0.33, which may be because our data is very unbalanced, that is to say:
$$
\frac{\#Class1}{\#Class2} \approx \frac{2}{1}
$$
So it is likely that the ratio near the boundary point still be close to 2:1, so threshold is a good choice.

### Least Square Model

#### 	Least square without regularization

​	The method of **least squares** is a standard approach in [regression analysis](https://en.wikipedia.org/wiki/Regression_analysis) to approximate the solution of [overdetermined systems](https://en.wikipedia.org/wiki/Overdetermined_system) (sets of equations in which there are more equations than unknowns) by minimizing the sum of the squares of the residuals.
$$
residual_i = y_i - f(x_i,\beta) \\
S = \sum_i(residual_i^2) = (\bold y - \bold X \beta)^T(\bold y - \bold X \beta)\\
\beta_{LS}=(\bold X^T\bold X)^{-1}\bold X^T\bold y
$$

Here, the dimantion of y_i is (1,2)  and the meaning of it is the score for getting 0 and the score for getting 1.

```python
def trainning_least_square(trainning_set):
    y_vec = trainning_set['class']
    x_matrix = trainning_set.drop(['class'], axis = 1)
    # you can use this to standardize the data if no preprocessing
    # x_matrix = x_matrix.apply(lambda t:(t - np.min(t)) / (np.max(t) - np.min(t)))
    # x_matrix = x_matrix.apply(lambda t:((t - np.mean(t)) / np.std(t)))
    x_matrix['one'] = 1
    y_vec = classOneHot(y_vec)
    x_matrix = np.array(x_matrix)
    x_matrix = x_matrix.T
    first_part = np.dot(y_vec, x_matrix.T)
    second_part = np.linalg.inv(np.dot(x_matrix, x_matrix.T))
    beta_hat = np.dot(first_part, second_part)
    return beta_hat
```

The result is:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7967 |   0.82   |   0.71    |  0.98  |   0.72   |
|        |   0.34   |   0.85    |  0.22  |          |

#### Least square with L2 regularization

The formula of least square with L2 regularization is:	
$$
S(\beta) = \sum_i residual_i^2 +   \lambda\sum_i\beta_i^2 = (\bold y - \bold X \beta)^T(\bold y - \bold X \beta) + \lambda\beta^T\beta\\
\beta_{LS}=(\bold X^T\bold X)^{-1}\bold X^T\bold y + \lambda\bold I \beta
$$

```python
    y_vec = trainning_set['class']
    x_matrix = trainning_set.drop(['class'], axis = 1)
    # x_matrix = x_matrix.apply(lambda t:((t - np.mean(t)) / np.std(t)))
    x_matrix['one'] = 1
    y_vec = classOneHot(y_vec)
    x_matrix = np.array(x_matrix)
    x_matrix = x_matrix.T
    beta_hat = np.dot(np.dot(y_vec, x_matrix.T),
                      np.linalg.inv(np.dot(x_matrix, x_matrix.T)+
                      lam * np.diag(np.ones(len(trainning_set.T)))))
```

Using the AUC as the criteria of choosing $\lambda$, we get the graph:

![](https://s2.loli.net/2022/03/22/aeKuES1nvJ8DdHM.png)

The results shows that when $\lambda = 2.1$, it gets the best AUC:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7982 |   0.82   |   0.71    |  0.98  |   0.72   |
|        |   0.32   |   0.83    |  0.20  |          |

It says that adding regularization does no help to the model fitting. The possible reasons are: (1) we use the dataset after PCA, so adding L2 regularization does function on the features selection; (2) the data has a small overlap between the two categories, which is difficult to distinguish by linearity and simple nonlinearity.

#### The predictions on the training set: without and with $\lambda = 2.1$

The model performance on training data is:

| $\lambda$ |  AUC   | F1-score | Precision | Recall | Accuracy |
| :-------: | :----: | :------: | :-------: | :----: | :------: |
|     0     | 0.7680 |   0.88   |   0.80    |  0.96  |   0.79   |
|           |        |   0.33   |   0.64    |  0.22  |          |
|    2.1    | 0.7688 |   0.88   |   0.80    |  0.96  |   0.79   |
|           |        |   0.32   |   0.63    |  0.22  |          |

### LS with Gaussian Basis Model

#### How to Choose Appropriate $\mu$ and $\sigma^2$

##### K-means Clustering

​	 This is an appropriate way to determine $\mu$ of the Gaussian Basis functions. It will cluster data points with similar features into the same group, then I can set the values of centroid data in each group as $\mu$ for the Gaussian Basis functions. The data I use can represent the average performance of one group of data. 

<img src="https://s2.loli.net/2022/03/22/ouhePFQEUMqt3YZ.jpg" alt="K-means-clustering-based fiber nonlinearity equalization techniques for  64-QAM coherent optical communication system" style="zoom:50%;" />

I will only use 64 attributes from data to do K-means clustering. Here, I will divide the data into 20 groups, and use the data of the centroid of each group to form the Gaussian function. Here, I will use `KMeans` function from `sklearn.cluster` to do the clustering. The code is the following:

```python
def kClusteringMean(self, n):
    """
    Use k-clustering method to find 20 mean values
    """
    train_adj = self.train_x.copy()
    train_adj["cluster"] = KMeans(n_clusters=n, max_iter=10000, random_state=119020).fit_predict(self.train_x)
    train_adj_mean = train_adj.groupby("cluster").mean()
    self.k_cluster_mean = train_adj_mean
```

##### Random Generate

​	First, I will calculate mean and variance of each attributes, and use these data to form 64 normal distribution functions. After that, I will randomly generate 80 points from each normal distribution functions. Therefore, I get a `80×64` mean value matrix. Here, I will use `numpy.random.normal` function to randomly generate data from the given normal distribution. The code I use is as the following:

```python
def normalRandomMean(self, n, seed):
    mean_var_lst = list(zip(self.train_x.mean(), np.sqrt(self.train_x.var())))
    zeros = np.zeros([n, len(mean_var_lst)])
    for attr in range(len(mean_var_lst)):
        np.random.seed(seed)
        zeros[:,attr] = np.random.normal(mean_var_lst[attr][0], mean_var_lst[attr][1],n)
    zeros = pd.DataFrame(zeros, columns = self.k_cluster_mean.columns)
    self.mean_generate = pd.concat([pd.DataFrame(zeros), self.k_cluster_mean])
```

##### Choose an appropriate $\sigma^2$

​	Here, I use the variance of the origin data as the $\sigma^2$ I use in the Gaussian basis functions. 

#### Using Gaussian Basis Functions to Transform Data

​	From *2.1.1*, I have generated a `100×64` mean value matrix, and a constant variance. I will use the following formula to transform the origin data:
$$
\phi_j(x)=exp\{-\frac{(x-\mu_j)^2}{2\sigma^2}\}
$$
After that, I will get a new training set with the size of `100×949×64`. The code I use is:

```python
def gaussianTransform(self, row):
    gaussian_x_train = np.exp(- (self.train_x - self.mean_generate.iloc[row,])**2 / 2 * self.train_x.var())
    guassian_x_test = np.exp(- (self.test_x - self.mean_generate.iloc[row,])**2 / 2 * self.test_x.var())
    gaussian_x_train["constant"] = 1
    guassian_x_test["constant"] = 1
    self.gaussian_x_train = np.array(gaussian_x_train).T
    self.gaussian_x_test = np.array(guassian_x_test).T
```

#### Calculate Parameters

​	After getting the transformed data from Gaussian basis functions, I can use similar method in Least Square method to calculate the data. After using one-hot method to transform the origin test data $y$(with size `949×1`) into new test data $T$(with size `949×2`). Then I use the following formula to calculate the parameters:
$$
w_{LS}=(X^TX)^{-1}X^TT
$$
The code is the following:

```python
def gaussianTrain(self, row):
    self.gaussianTransform(row)
    T_XT = np.dot(self.train_y, self.gaussian_x_train.T)
    X_XT_inv = np.linalg.pinv(np.dot(self.gaussian_x_train, self.gaussian_x_train.T))
    self.parameters = np.dot(T_XT, X_XT_inv)
    self.gaussian_predict()
```

#### Predict Result

​	From *2.1.3*, I get the parameter of the least square method, then I can use the formula to do the prediction:
$$
\hat y(\bold x)=\bold w^T\bold x+w_0
$$
Now, I will get a `2×151` prediction method, with each column is the predict value of the given test data. For each column $\bold t = [t_1, t_2]^T$, I will compare the predict values under two parameters. I will predict the data as `0` if $t_1>t_2$, and predict it as `+1` otherwise. After that, I will calculate `F1-score`, `Accuracy` and `AUC` of the prediction. The code I use is:

```python
def gaussian_predict(self):
    df = pd.DataFrame(np.dot(self.parameters, self.gaussian_x_test)).T
    df["predict"] = 0
    df.loc[df[1]>df[0], "predict"] = 1
    target_names = ['class 1', 'class 2']
    result = classification_report(np.array(self.test_y), df['predict'], target_names=target_names)
    self.auc_score = roc_auc_score(np.array(self.test_y), df['predict'])
    self.result = result
    self.recordDict["class1 F1"].append(float(self.result.split("\n")[2].split()[4]))
    self.recordDict["class2 F1"].append(float(self.result.split("\n")[3].split()[4]))
    self.recordDict["Accuracy"].append(float(self.result.split("\n")[5].split()[1]))
    self.recordDict["AUC"].append(self.auc_score)
```

#### Result Analysis

​	In total, I tried 100 different Gaussian Basis Functions, with 100 different $\mu$ and 1 constant $\sigma^2$. Among all the 100 models, the average accuracy is **66.45%**, the average AUC (Area Under the Curve) is **63.85%**. The result is as the following:

<img src="https://s2.loli.net/2022/03/22/2tJk49SuAMmo8Uj.png" alt="image-20220319210728865" style="zoom:80%;" />

This average prediction results are relatively lower than other models because:

1. In `k-clustering method`, some groups only contain several points, which makes the mean value can not represent the whole data set. Therefore, the Gaussian transformation will make the prediction worse.
2. In `randomly generating method`, sometimes it will generate illogical data, therefore using this data to do the Gaussian transformation will make the prediction worse.

However, the best prediction model using LS with Gaussian basis function can have **79.23%** accuracy in testing, with **76.37%** AUC, which is pretty good in classification. The result matrix is:

|                  | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 1**      | 0.78      | 0.95   | 0.86     | 100     |
| **Class 2**      | 0.83      | 0.47   | 0.60     | 51      |
| **Accuracy**     |           |        | 0.79     | 151     |
| **Macro Avg**    | 0.80      | 0.71   | 0.73     | 151     |
| **Weighted Avg** | 0.80      | 0.79   | 0.77     | 151     |
| **AUC**          |           |        | 0.76     | 151     |

The ROC and AUC of LS with Gaussian basis for testing data and training data are as the following:

<img src="https://s2.loli.net/2022/03/22/ecqGgp5RJSN1tdl.png" alt="image-20220321224320251" style="zoom: 80%;" />

### Linear Discriminant Analysis (LDA)

#### Overview of LDA

​	Linear Discriminant Analysis (LDA) aims to project data into a dimension that maximizes variance among the different classes and minimizes variance within each of the classes. LDA believes that in this new dimension, better classification can be carried out.

​	We consider the case of a two-class classification problem, we need to find **w** to project data. So, we 
$$
\max _{w} J(w) = \frac{w^T\mathbf{S}_{\mathrm{B}}w}{w^T\mathbf{S}_{\mathrm{W}}w}(Fisher criterion)
$$
subject to
$$
\begin{aligned}
&\mathbf{m}_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N_{k}} \mathbf{x}_{n} \quad \text { center of class } \mathbf{k} \\
&\mathbf{S}_{\mathrm{B}} = (\mathbf{m}_{2}-\mathbf{m}_{1})(\mathbf{m}_{2}-\mathbf{m}_{1})^T
\begin{array}{l}
\text {covariance between class}
\end{array}\\
&\mathbf{S}_{\mathrm{W}}=\sum_{n \in \mathcal{C}_{1}}\left(\mathbf{x}_{n}-\mathbf{m}_{1}\right)\left(\mathbf{x}_{n}-\mathbf{m}_{1}\right)^{\mathrm{T}}+\sum_{n \in \mathcal{C}_{2}}\left(\mathbf{x}_{n}-\mathbf{m}_{2}\right)\left(\mathbf{x}_{n}-\mathbf{m}_{2}\right)^{\mathrm{T}}
\begin{array}{l}
\text {covariance within class}
\end{array}
\end{aligned}
$$
​	This optimization problem can be transformed into a generalized eigenvalue problem.
$$
\mathbf{S}_{\mathrm{B}} \mathbf{w}=J(\mathbf{w}) \mathbf{S}_{\mathbf{w}} \mathbf{w}
$$
So w is the eigenvector corresponding to the maximum eigenvalue of $$\mathbf{S}_{\mathrm{W}}^{-1}\mathbf{S}_{\mathrm{B}}$$

​	The whole process of establishing the LDA of two categories is:

- **Step-1**: Calculate $$\mathbf{m}_{1}$$, $$\mathbf{m}_{2}$$, $$\mathbf{S}_{\mathrm{W}}$$ and $$\mathbf{S}_{\mathrm{B}}$$ by using training data
- **Step-2**: Calculate the eigenvector corresponding to the maximum eigenvalue of $$ \mathbf{S}_{\mathrm{W}}^{-1}\mathbf{S}_{\mathrm{B}} $$ as $$w$$
- **Step-3**: Find threshold $$w_0 = - \frac{w^T\mathbf{m}_{1} + w^T\mathbf{m}_{2}}{2}$$
- **Step-4**: For new data $$x$$, if $$\frac{w_0+w^Tx}{w_0+w^T\mathbf{m}_{2}}>0$$, we put it in class 2. Otherwise, we put it in class 1

See appendix for code.

#### Result Analysis

Now, we use LDA in our pre-processed data. In the training dataset we got the following results:

| Training Data | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Class 1       | 0.87      | 0.69   | 0.77     | 100     |
| Class 2       | 0.39      | 0.67   | 0.50     | 51      |
| Accuracy      |           |        | 0.68     | 151     |
| Macro Avg     | 0.63      | 0.68   | 0.63     | 151     |
| Weighted Avg  | 0.76      | 0.68   | 0.71     | 151     |
| AUC           |           |        | 0.73     | 151     |

The figure below shows the projected data and decision boundaries in the training dataset.

<img src="https://s2.loli.net/2022/03/22/JvhUaWud7qZRsXA.png" alt="LDA train" style="zoom:80%;" />

In the testing dataset we got the following results:

| Testing Data | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Class 1      | 0.82      | 0.80   | 0.81     | 100     |
| Class 2      | 0.63      | 0.67   | 0.56     | 51      |
| Accuracy     |           |        | 0.75     | 151     |
| Macro Avg    | 0.73      | 0.73   | 0.73     | 151     |
| Weighted Avg | 0.76      | 0.75   | 0.76     | 151     |
| AUC          |           |        | 0.76     | 151     |

​	The figure below shows the projected data and decision boundaries in the testing dataset.

<img src="https://s2.loli.net/2022/03/22/Dqhecw7SoYHs3y5.png" alt="LDA test" style="zoom:80%;" />

​	Here is the ROC curve:

<img src="https://s2.loli.net/2022/03/22/7sKN5IGbiuRAQmS.png" style="zoom:80%;" />

​	We were surprised to find that the models on the test dataset performed even better than those on the training dataset. In the projected distribution, we can find that the test dataset has a distribution that is more consistent with our expectations, but in the training dataset, there are a lot of confounding data near the decision boundary, which leads to better performance of our model on the test dataset. This is a coincidence caused by the inconsistent distribution of training and test data sets. 

​	But overall, the results of the test set are satisfactory. It has a good recall rate(67%) for class 2 , and a high AUC(76%) and accuracy rate(75%) overall.

### Logistic classification with/without regularization (L2).

#### Logistic classification without regularization

The binary logistic regression model in the general form:
$$
y = f(\textbf{x};\beta) + \varepsilon
$$
The output y takes on two possible outcomes: "is the case (1)" and "not the case (2)". Then we assume that the output y given a specific x is a Bernoulli random variable with the probability mass function as follows:

|  y   |   Probability mass function    |
| :--: | :----------------------------: |
|  1   |   $P (y = 1|\textbf{x}) = p$   |
|  0   | $P (y = 0|\textbf{x}) = 1 - p$ |

Due to the Bernoulli model assumption, we have:
$$
y = f(\textbf{x};\beta) + \varepsilon \\
	 =E(y|\textbf{x},\beta) + \varepsilon \\
	 =P(y = 1|\textbf{x}) + \varepsilon \\
$$
$P (y = 1|x) = p$ can be regarded as "the probability that the output y takes outcome given the inputs x is equal to p". It is most widely used that:
$$
P(y = 1|x) = e^{\textbf{x}^T\beta}/ (1 + e^{\textbf{x}^T\beta}) = \frac{1}{1 + e^{\textbf{x}^T\beta}}
$$

Then we fit model using MLE Estimation: given the set of 996 data points in $S = { (y_1,x_1) , (y_2,x_2), (y_3,x_3) ...(y_996,x_996) }$, the likelihood function is given:
$$
L(y|X,\beta) = L(y_1,y_2...y_{949}|X, \beta) = \prod_i P(y_i | x_i) = \prod_i p_i^{y_i} (1-p_i)^{1-y_i}\\
I(\beta) = logL = \sum_iln(1-p_i) + \sum_iy_i ln(\frac{p_i} {1-p_i})\\
 = \sum_iln(1-\frac{1}{1 + e^{\textbf{x}^T\beta}}) + \sum_iy_i ln(\frac{1}{1 + e^{\textbf{x}^T\beta}} / (1-\frac{1}{1 + e^{\textbf{x}^T\beta}})
$$
We can clearly see that the partial derivative of $I (\beta)$ does not contain beta, so we choose to use gradient descent method to find the suitable beta which gives the lower loss value:
$$
Loss = \sum_i (\hat{y_i} - y_i)^2
$$

```python
# Loss function
def square_loss(y_pred, target):
  return np.mean(pow((y_pred - target),2))
# Gradient Descent
for i in range(10000):   
    gradient_W = np.dot((y_pred-y_tr).T, X_tr)/(X_tr.shape[0])
    gradient_b = np.mean(y_pred-y_tr)
    W = W - lr * gradient_W
    b = b - lr * gradient_b
    z = np.dot(X_tr, W) + b
    y_pred = sigmoid(z)
    loss.append(square_loss(y_pred, y_tr))
```

The prediction result shows:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7945 |   0.84   |   0.72    |  0.99  |   0.75   |
|        |   0.40   |   0.93    |  0.25  |          |

#### Logistic classification with regularization

The constraint represented by the l2 regularization is a hypersphere in high-dimensional space, and since there are no corners, the probability of the first intersection occurring at a location with sparsity becomes very small, tending to obtain smaller numbers that are all non-zero and therefore smooth.

In addition to preventing overfitting, l2 regularization can also be used as a feature filtering method, so that the weight coefficients of features that are not too important to the model tend to zero, and then we can remove and reselect features according to the specific situation, thus playing a role in improving generalization performance as well as saving memory space and improving operational efficiency.

The log likelihood function becomes:
$$
I(\beta) = logL = \sum(ln(1-p_i)) + \sum y_iln(\frac{p_i}{ 1-p_i}) +\lambda\sum\beta_j\\
 \frac{\partial I(\beta)}{\partial \beta} = (\bold p^T-\bold y^T) \bold X + \lambda\beta
$$
Now, we still need to use gradient descent method to find the best \beta. And in the gradient descent formula, we need to add the L2 norm term :

```python
for i in range(10000):   
        gradient_W = (np.dot((y_pred-y_tr).T, X_tr) + lam * W)/(X_tr.shape[0])
        gradient_b = np.mean(y_pred-y_tr)
        W = W - lr * gradient_W
        b = b - lr* gradient_b
        z = np.dot(X_tr, W) + b
        y_pred = sigmoid(z)
        loss.append(square_loss(y_pred, y_tr))
```

We use for loop to search for the best $\lambda$ from 1 to 200 with step 1 and according to the graph:

<img src="https://s2.loli.net/2022/03/22/ldDtrLUwezuZfjI.png" style="zoom:80%;" />

Then we check the f1 score, precision, recall and accuracy, there exists a trade off between AUC and them:

| $\lambda$ |  AUC   | F1-score | precision | Recall | Accuracy |
| :-------: | :----: | :------: | :-------: | :----: | :------: |
|     1     | 0.796  |   0.83   |   0.72    |  0.99  |   0.74   |
|           |        |   0.38   |   0.92    |  0.24  |          |
|    30     | 0.800  |   0.83   |    0.7    |  1.00  |   0.72   |
|           |        |   0.3    |   1.00    |  0.18  |          |
|    100    | 0.8025 |   0.81   |   0.68    |  1.00  |   0.68   |
|           |        |   0.11   |   1.00    |  0.06  |          |
|    170    | 0.8035 |   0.8    |   0.67    |  1.00  |   0.67   |
|           |        |   0.04   |   1.00    |  0.02  |          |

Obviously, adding L2 regularization does no help toward the Logistic classification: while little increasing in the AUC, there is a huge decrease in F1 score. Let's try small pieces of $\lambda$ from (0, 20) to see the behaviors of AUC:

<img src="https://s2.loli.net/2022/03/22/vRcQItiYo8WXe2x.png" alt="image-20220321213158715" style="zoom:80%;" />

The graph shows that the existence of $\lambda$ leads to random movement of the AUC, meaning that the L2 regularization almost does not work. The possible reasons are: (1) we use the dataset after PCA, so adding L2 regularization does function on the features selection; (2) the data has a small overlap between the two categories, which is difficult to distinguish by linearity and simple nonlinearity. Then we just use the Logistic regression model without regularization. The prediction outcomes in the training set is:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7720 |   0.88   |   0.81    |  0.95  |   0.80   |
|        |   0.38   |   0.64    |  0.27  |          |

### SoftMax Classification Model

#### Overview of SoftMax Model

​	SoftMax model is the generalized form of logistic classification model. It can do the multi-class classification, while the logistic model can only be applied to binary classification problem. In SoftMax model, for the given data $\{x_1,y_1),...,(x_n,y_n)\}$ with $k$ classes, we should first calculate the probability $P(x_i)\in class_k |\theta$ for the given parameter $\theta=[\theta_1,...,\theta_n]$ using the following formula:

![[公式]](https://www.zhihu.com/equation?tex=h_{\theta}\left(x_i\right)%3D\left[\begin{array}{c}{p\left(y_i%3D1+|+x_i+%3B+\theta\right)}+\\+{p\left(y_i%3D2+|+x_i+%3B+\theta\right)}+\\+{\vdots}+\\+{p\left(y_i%3Dk+|+x_i+%3B+\theta\right)}\end{array}\right]%3D\frac{1}{\sum_{j%3D1}^{k}+e^{\theta_{j}^{T}+x_i}}\left[\begin{array}{c}{e^{\theta_{1}^{T}+x_i}}+\\+{e^{\theta_{2}^{T}+x_i}}+\\+{\vdots}+\\+{e^{\theta_{k}^{T}+x_i}}\end{array}\right]\tag{1}+\\)

After that, we want to maximize the likelihood function for the training data, given parameters $\theta$:
$$
maximize\ \prod_{i=1}^n h_\theta(x_i)
$$
I will use the Gradient Descent method to find the parameters.

#### Use Gradient Descent to Find Parameters:

​	I will define the negative logarithm of likelihood (cross-entropy error function) as the loss function. Here, the loss function is:

<img src="https://www.zhihu.com/equation?tex=L(\theta)%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+\sum_{j%3D1}^{k}+1\left\{y_i%3Dj\right\}+\log+\frac{e^{\theta_{j}^{T}+x_i}}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right]\tag{4}+\\" alt="[公式]" style="zoom:80%;" />

By taking gradient for each $\theta_j$, I get the gradient using the following formula[1]:

![[公式]](https://www.zhihu.com/equation?tex=\begin{aligned}++\frac{\partial+L(\theta)}{\partial+\theta_{j}}++%26%3D-\frac{1}{m}+\frac{\partial}{\partial+\theta_{j}}\left[\sum_{i%3D1}^{m}+\sum_{j%3D1}^{k}+1\left\{y_i%3Dj\right\}+\log+\frac{e^{\theta_{j}^{T}+x_i}}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right]+\\++%26%3D-\frac{1}{m}+\frac{\partial}{\partial+\theta_{j}}\left[\sum_{i%3D1}^{m}+\sum_{j%3D1}^{k}+1\left\{y_i%3Dj\right\}\left(\theta_{j}^{T}+x_i-\log+\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}\right)\right]+\\++%26%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+1\left\{y_i%3Dj\right\}\left(x_i-\sum_{j%3D1}^{k}+\frac{e^{\theta_{j}^{T}+x_i}+\cdot+x_i}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right)\right]+\\+%26%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+x_i1\left\{y_i%3Dj\right\}\left(1-\sum_{j%3D1}^{k}+\frac{e^{\theta_{j}^{T}+x_i}}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right)\right]+\\+%26%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+x_i\left(1\left\{y_i%3Dj\right\}-\sum_{j%3D1}^{k}+1\left\{y_i%3Dj\right\}+\frac{e^{\theta_{j}^{T}+x_i}}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right)\right]+\\+%26%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+x_i\left(1\left\{y_i%3Dj\right\}-+\frac{e^{\theta_{j}^{T}+x_i}}{\sum_{l%3D1}^{k}+e^{\theta_{l}^{T}+x_i}}\right)\right]+\\+%26%3D-\frac{1}{m}\left[\sum_{i%3D1}^{m}+x_i\left(1\left\{y_i%3Dj\right\}-p\left(y_i%3Dj+|+x_i+%3B+\theta\right)\right)\right]+\end{aligned}\tag{5}+\\)

I will get the parameters using the following steps:

1. Randomly choose the initial parameters $\theta_0$. 
2. Calculate the loss function $L(\theta)$ mentioned above.
3. Calculate the gradient descent $\Delta \theta$ using the above formula.
4. If $\Delta \theta < \varepsilon$, then stop and return the parameter. Otherwise, continue.
5. Update parameter $\theta$ using $\theta_{n+1} = \theta_n-\alpha \Delta \theta_n$.
6. Update softmax values using the new parameter and the formula (1) mentioned above.
7. Repeat step 2~6 until it reaches the maximum iteration times or the loss function converges.

#### Result Analysis

​	The gradient descent method will converge after 1,000,000 iterations. The loss function is visualized as:

<img src="https://s2.loli.net/2022/03/20/EkvuX6iemAI7Lnl.png" alt="image-20220315175637822" style="zoom:67%;" />

The accuracy of classification is **75.39%** with AUC **68.15%**. The performance of softmax method is pretty good for the classification problem. The result matrix is:

|                  | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 1**      | 0.71      | 0.96   | 0.82     | 100     |
| **Class 2**      | 0.75      | 0.24   | 0.36     | 51      |
| **Accuracy**     |           |        | 0.75     | 151     |
| **Macro Avg**    | 0.74      | 0.68   | 0.69     | 151     |
| **Weighted Avg** | 0.75      | 0.75   | 0.74     | 151     |
| **AUC**          |           |        | 0.75     | 151     |

<img src="C:\Users\Yip\AppData\Roaming\Typora\typora-user-images\image-20220321225812996.png" alt="image-20220321225812996" style="zoom:80%;" />

#### Code of SoftMax method

​	Please refer to the appendix.

### Decision Tree

#### Overview of Decision Tree

Decision Tree is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

<img src="https://s2.loli.net/2022/03/20/bmthuMEwUlF5feG.png" alt="Decision Tree overview" style="zoom:67%;" />

The general process of decision tree is：

- **Step-1**: Begin the tree with the root node, says S, which contains the complete training dataset.
- **Step-2**: Split the node according to its attributes and work out the impurity (entropy) for each subset of every attribute.
- **Step-3**: Compute the information gain for each attribute.
- **Step-4**: Generate the decision tree node, which contains the best attribute with maximum information gain.
- **Step-5**: Repeat step-2 to step-4, until a stage is reached where we cannot further classify the nodes and called the final node as a leaf node.

We use sklearn to implement Decision Tree. See appendix for code.

#### Result Analysis

​	Now, we use Decision Tree in our imputed data since we want to use 100% information. 

The decision tree model has many hyperparameters to choose from, and we use the following three important hyperparameters to optimize the model:

- Split Criterion: Split criterion is the parameter used to determine how impurity is calculated. We have two choices: Entropy and Gini
- Max Depth: Limit the maximum depth of the tree, and all branches exceeding the set depth are cut off to prevent over-fitting of the model.
- Min Samples Leaf: Each child node of a node after sorting must contain at least min samples leaf training samples, otherwise splitting will not occur. This is also a way of limiting overfitting

First, we choose Split Criterion and Max Depth. The following are the changes of AUC, precision rate, recall rate and f1-score with Max Depth between 1 to 20 with different Split Criterion in testing data (threshold = 0.3 for the reason which will be explained later):

<img src="https://s2.loli.net/2022/03/21/6cULXCrjhk8RtiG.png" alt="Decision Tree Split Criterion and Max Depth" style="zoom:80%;" />

It can be seen that using entropy as the splitting criterion and Max depth = 6 has the highest AUC, and precision rate, recall rate and F1-score are close to the highest.

Then we use entropy as the splitting criterion and Max depth = 6, Min Samples Leaf was changed from 1 to 100 to check the AUC, precision rate, recall rate and F1-score:

<img src="https://s2.loli.net/2022/03/21/zfHusBkhwRSLmVx.png" alt="Decision Tree Min Samples Leaf" style="zoom: 67%;" />

It can be seen that using entropy as the splitting criterion, Max depth = 6 and Min sample leaf = 40 has the highest AUC, and precision rate, recall rate and F1-score are close to the highest.

The following is the decision tree model after training with entropy as the splitting criterion, Max depth = 6 and Min sample leaf = 40:

<img src="https://s2.loli.net/2022/03/21/7RvdUJpGwxuaCYQ.png" alt="image-20220321205249036" style="zoom:67%;" />

 In the training dataset we got the following results:

| threshold = 0.5 | Precision | Recall | F1-score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Class 1         | 0.87      | 0.90   | 0.89     | 100     |
| Class 2         | 0.64      | 0.57   | 0.60     | 51      |
| Accuracy        |           |        | 0.83     | 151     |
| Macro Avg       | 0.76      | 0.74   | 0.75     | 151     |
| Weighted Avg    | 0.82      | 0.83   | 0.82     | 151     |
| AUC             |           |        | 0.87     | 151     |

In the testing dataset we got the following results:

| threshold = 0.5 | Precision | Recall | F1-score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Class 1         | 0.80      | 0.94   | 0.87     | 100     |
| Class 2         | 0.82      | 0.55   | 0.66     | 51      |
| Accuracy        |           |        | 0.81     | 151     |
| Macro Avg       | 0.81      | 0.74   | 0.76     | 151     |
| Weighted Avg    | 0.81      | 0.81   | 0.80     | 151     |
| AUC             |           |        | 0.81     | 151     |

Since model have a high AUC, we can change the threshold：

 In the training dataset we got the following results:

| threshold=0.3 | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Class 1       | 0.91      | 0.84   | 0.87     | 100     |
| Class 2       | 0.57      | 0.71   | 0.64     | 51      |
| Accuracy      |           |        | 0.81     | 151     |
| Macro Avg     | 0.74      | 0.78   | 0.75     | 151     |
| Weighted Avg  | 0.83      | 0.81   | 0.82     | 151     |
| AUC           |           |        | 0.87     | 151     |

In the testing dataset we got the following results:

| threshold=0.3 | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| Class 1       | 0.82      | 0.87   | 0.84     | 100     |
| Class 2       | 0.71      | 0.63   | 0.67     | 51      |
| Accuracy      |           |        | 0.79     | 151     |
| Macro Avg     | 0.77      | 0.75   | 0.76     | 151     |
| Weighted Avg  | 0.78      | 0.79   | 0.78     | 151     |
| AUC           |           |        | 0.81     | 151     |

When the threshold is adjusted to 0.3, we can find that although the overall accuracy is somewhat reduced, the recall rate of Class 2 and F1 score are improved.

Here is the ROC curve:

<img src="https://s2.loli.net/2022/03/21/Tr6XQj4vNIC3PoD.png" alt="Decision Tree ROC" style="zoom:80%;" />

### Random Forest Classification Model

#### Overview of Random Forest Model

​	Random forest model is randomly creating several decision trees by choosing certain features and samples. After that, using majority voting method to determine the label of the given input. Compared with decision tree model,  random forest model is more reliable, stable and robust. However, it may take more time in model training and predicting. 

<img src="C:\Users\Yip\AppData\Roaming\Typora\typora-user-images\image-20220319203224716.png" alt="image-20220319203224716" style="zoom: 67%;" />

#### Choose Hyper-parameters

​	Here, I use `GridSearchCV` method from `sklearn` package to choose hyper-parameters *max_features, criterion, max_feature*. I will use the prediction of testing set to estimate the performance of each hyper-parameter. The evaluation criteria I choose is `AUC`. 

##### Determine maximum feature used in each tree

​	`max_feature` parameter determines the number of maximum features used in each decision tree of the model. There are 3 choices: not limit, square root of total number of features and logarithm of the total number of features. From the training process I found that we should not limit the maximum features used in the model.

<img src="https://s2.loli.net/2022/03/21/e68VJmyv3ahSpcr.png" alt="image-20220321183903748" style="zoom:67%;" />

##### Choose Criterion

​	`Criterion` is the evaluation criteria for the classification random forest to decide how to divide each attribute. I will compare the performance of "Gini Index" and "Entropy", when I change the number of the maximum decision trees in the model. 

<img src="https://s2.loli.net/2022/03/21/iOJG3Kkmv8Ht79r.png" style="zoom:67%;" />

From the above figure I found the performance of "Gini index" is much better than the performance of "Entropy". Thus, I will choose "Gini index" as one of the hyperparameter.

##### Choose number of maximum decision trees in the model

​	From above two process, when number of maximum decision trees is 74, the model performs the best.

##### Choose the maximum depth of each decision tree

​	When I set `n_estimator=74, Criterion=Gini, Max_feature=None`, I can view the influence of the maximum depth to the model. 

<img src="https://s2.loli.net/2022/03/21/XiYvSWeZ3mctDP8.png" alt="image-20220321185655895" style="zoom:67%;" />

From the above figure I found that when `max_depth=9`, the model performs the best. 

##### Choose of threshold

​	After observing the result of AUC and weighted average F1-score, I think the default threshold 0.5 is not a good estimator for the model to do the prediction. Therefore, I change the threshold from 0.5 to 0.33. And this significantly increase the final F1-score and recall rate.

##### Choose of Hyper-parameters

From the above figure, I can choose the following hyper-parameter to build the `RandomForest` model:

| Max_estimator | Criterion | Max_feature | Max_Depth | Threshold |
| ------------- | --------- | ----------- | --------- | --------- |
| 74            | Gini      | None        | 9         | 0.33      |

#### Result Evaluation

​	Using the above parameters, the model gets 87.17% AUC score in the testing data with 76.12% average accuracy. This is the best model among all the models we get. The performance on testing data is: 

| Testing Data     | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 1**      | 0.75      | 0.97   | 0.84     | 100     |
| **Class 2**      | 0.86      | 0.35   | 0.50     | 51      |
| **Accuracy**     |           |        | 0.76     | 151     |
| **Macro Avg**    | 0.80      | 0.66   | 0.67     | 151     |
| **Weighted Avg** | 0.78      | 0.76   | 0.73     | 151     |
| **AUC**          |           |        | 0.88     | 151     |

The performance on training data is:

| Training Data    | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Class 1**      | 0.97      | 1.00   | 0.98     | 729     |
| **Class 2**      | 1.00      | 0.88   | 0.94     | 220     |
| **Accuracy**     |           |        | 0.97     | 949     |
| **Macro Avg**    | 0.98      | 0.94   | 0.96     | 949     |
| **Weighted Avg** | 0.97      | 0.97   | 0.96     | 949     |
| **AUC**          |           |        | 1.00     | 949     |

<img src="https://s2.loli.net/2022/03/21/3qGoiDzyF4gSBlE.png" alt="image-20220321213912290" style="zoom:80%;" />

After that, I rank the importance of each attribute in the random forest model. It is as the following:

<img src="https://s2.loli.net/2022/03/21/XI63kpyEvY4QRO5.png" alt="image-20220321195150479" style="zoom:80%;" />

From the above figure, I found that the attribute, which is "Attr24", is of the most important for the model. 

### Neural Network

#### selection of layers and units

Intuitively, we think that the dataset is not that large and we use pca to decrease the feature, therefore, we try layers from 2-4 and units 48, 64 and 80 with active function "tanh" to find better combinations between  layers and units. By comparing the training loss, training accuracy, validation loss, validation accuracy:

![W&B Chart 2022_3_20 19_36_06](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 19_36_06.png)

![W&B Chart 2022_3_20 20_43_16](https://s2.loli.net/2022/03/20/eg3HDZ59zcEWhvf.png)

![W&B Chart 2022_3_20 20_42_04](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 20_42_04.png)

![W&B Chart 2022_3_20 20_42_13](https://s2.loli.net/2022/03/20/O8iYCnZfDIGSskR.png)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
N_HIDDEN = 64
# Add an input shape! (features,)
model.add(Dense(N_HIDDEN, input_shape=(X.shape[1],), activation='tanh')) 
model.add(Dense(N_HIDDEN, activation='tanh'))
model.add(Dense(N_HIDDEN, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.summary() 
```

2 layers with 64 units validation loss decreases fastest and has the best performance in validation accuracy. In the rest two parts, it also outperforms. Followed with it is 3 layers with 64 units and 4 layers with 80 units so we use this three to do further adjusting.

#### Selection of optimizer

Next, we need to choose the best optimizer in finding the best parameters. We find some comments in the web to first learn the difference between optimizers:

AdaGrad penalizes the learning rate too harshly for parameters which are frequently updated and gives more learning rate to sparse parameters, parameters that are not updated as frequently. In several problems often the most critical information is present in the data that is not as frequent but sparse. So if the problem you are working on deals with sparse data such as tf-idf, etc. Adagrad can be useful.

AdaDelta, RMSProp almost works on similar lines with the only difference in Adadelta you don't require an initial learning rate constant to start with.

Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems.

Stochastic gradient descent is very basic and is seldom used now. One problem is with the global learning rate associated with the same. Hence it doesn't work well when the parameters are in different scales since a low learning rate will make the learning slow while a large learning rate might lead to oscillations. Also Stochastic gradient descent generally has a hard time escaping the saddle points. Adagrad, Adadelta, RMSprop, and ADAM generally handle saddle points better. SGD with momentum renders some speed to the optimization and also helps escape local minima better.

Moreover, kerns provides Nadam, adamax and Ftrl with build in functions, we also try them on based on 2 layers and 64 units and get the following results:

![W&B Chart 2022_3_20 20_44_53](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 20_44_53.png)

![W&B Chart 2022_3_20 20_45_04](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 20_45_04.png)

![W&B Chart 2022_3_20 20_45_17](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 20_45_17.png)

![W&B Chart 2022_3_20 20_45_17](C:\Users\Yip\Downloads\W&B Chart 2022_3_20 20_45_17.png)



```python
# compile the model
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
# opt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
# opt = tf.keras.optimizers.Adamax(learning_rate=0.001)
# opt = tf.keras.optimizers.Ftrl(learning_rate=0.001)
model.compile(optimizer = opt, 
              loss='binary_crossentropy',
              metrics=['accuracy'])

```

So we choose to use Adamax and Nadam.

#### Select the callbacks, validation split rate and activation function

As for the callbacks, we choose to use (1) EarlyStopping and (2) ReduceLROnPlateau since this two give the strong power towards the fitting:

(1) EarlyStopping: Stop training when a monitored metric has stopped improving. In our model, it can reduce the training time largely without worse the prediction results.

(2) ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

```python
es_val_loss = EarlyStopping(monitor='val_loss',patience=10)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=0, min_lr=1e-8)
```

As for the validation split rate, in experience, people always set it in a range of (0.1, 0.3). Let's see the training and validation accuracy on [0.1, 0.15, 0.2, 0.25, 0.3]:

![0.1](https://s2.loli.net/2022/03/21/JiFR5TqkIz2wvP6.png)

<img src="https://s2.loli.net/2022/03/21/BRzqALmHjuC438p.png" alt="0.2" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/vEJYfKPzqDlTsXO.png" alt="0.15" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/HBzY1trpJPfTv8e.png" alt="0.25" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/csKZyxuLp2XWBdE.png" style="zoom: 80%;" />

All of them have a good performance in the training set, you can pick whatever you want from set (0.1, 0.3). We use 0.3 here. 

As for the activation function, we try `elu, relu, selu, sigmoid, softsign, tanh`. We set training and validation accuracy as the criteria to choose an appropriate activation function. Here are the results:

<img src="https://s2.loli.net/2022/03/21/QbDP37Zl9YcK4oE.png" alt="elu" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/CDyLl6sVJqPwi5t.png" alt="relu" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/GqV8O1xQKYvEdNe.png" alt="selu" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/FO8Zzr2Ro49CTHp.png" alt="sigmoid" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/FO8Zzr2Ro49CTHp.png" style="zoom:80%;" />

<img src="https://s2.loli.net/2022/03/21/UbcPjvHO2Jt3zRk.png" alt="tanh" style="zoom:80%;" />

According to the graphs, we can see `elu, tanh` perform the best. Therefore, we choose `tanh` function as our activation function.

#### Select the final parameters

After adding the weight of each class, we finally can train this nn model:

```python
counts = np.bincount(y)
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 1.0 / counts[1]
class_weight = {0: weight_for_0, 1: weight_for_1}
history = model.fit(X,
                    y,
                    callbacks=[WandbCallback(), reduce_lr, es_val_acc],
                    epochs=100, # maximum of epoches.
                    batch_size=128,
                    validation_split=0.3,
                    shuffle=True,
                    class_weight = class_weight,
                    verbose=1)
```

| Layers | Units | Optimizer |  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :---: | :-------: | :----: | :------: | :-------: | :----: | :------: |
|   2    |  64   |   Nadam   | 0.7737 |   0.76   |   0.79    |  0.73  |   0.70   |
|        |       |           |        |   0.58   |   0.54    |  0.63  |          |
|   3    |  64   |   Nadam   | 0.7898 |   0.81   |   0.80    |  0.82  |   0.74   |
|        |       |           |        |   0.61   |   0.59    |  0.62  |          |
|   2    |  64   |  Adamax   | 0.7627 |   0.79   |   0.82    |  0.76  |   0.73   |
|        |       |           |        |   0.62   |   0.59    |  0.67  |          |
|   3    |  64   |  Adamax   | 0.7937 |   0.81   |   0.81    |  0.81  |   0.75   |
|        |       |           |        |   0.63   |   0.63    |  0.63  |          |
|   4    |  80   |   Nadam   | 0.7649 |   0.81   |   0.80    |  0.83  |   0.75   |
|        |       |           |        |   0.61   |   0.64    |  0.59  |          |
|   4    |  80   |  Adamax   | 0.7975 |   0.82   |   0.84    |  0.80  |   0.77   |
|        |       |           |        |   0.67   |   0.64    |  0.71  |          |

Finally, we select 4 layers, 80 units, `tanh` activation function, validation rate = 0.3 and Adamax optimizer.

#### Add regularization (set seed 80)

Initial results :

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7982 |   0.83   |   0.83    |  0.83  |   0.77   |
|        |   0.67   |   0.67    |  0.67  |          |

<img src="C:\Users\Yip\Desktop\pic (2)\pic\acc.png" alt="acc" style="zoom:80%;" />

#### L1 and L2 regularization

The layers expose 3 keyword arguments:

- `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
- `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
- `activity_regularizer`: Regularizer to apply a penalty on the layer's output

We firstly add regularization in the first layer:

```python
model.add(Dense(N_HIDDEN, input_shape=(X.shape[1],), activation='tanh',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5)))
```

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7913 |   0.82   |   0.81    |  0.82  |   0.75   |
|        |   0.63   |   0.64    |  0.63  |          |

<img src="C:\Users\Yip\Desktop\pic (2)\pic\acc first layer reg.png" alt="acc first layer reg" style="zoom:80%;" />

Then we add regularization in all layers:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7616 |   0.77   |   0.79    |  0.75  |   0.70   |
|        |   0.58   |   0.55    |  0.61  |          |

<img src="C:\Users\Yip\Desktop\pic (2)\pic\all layer reg.png" alt="all layer reg" style="zoom:80%;" />

After adding the regularization in all layers, it seems that the training accuracy is closer to the validation accuracy, however, the model fitting becomes worse. Maybe the reason is that the use of early stopping leads to extreme regularization. Then we try to delete the early stopping term to see the result.

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7620 |   0.75   |   0.79    |  0.72  |   0.69   |
|        |   0.58   |   0.53    |  0.63  |          |

The results are still bad, which means that probability datasets is so small that it is not suitable by L1 and L2 regularizations. And moreover, we have used the PCA to do the feature selections and in the above training and validation accuracy graph, we also see that there is no overfit problem, therefore, L1 and L2 may not work. Let's try dropout method to do further findings.

#### Dropout

We try to dropout 25% units in each layers to see the results:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7990 |   0.83   |   0.84    |  0.81  |   0.77   |
|        |   0.68   |   0.65    |  0.71  |          |

It seems that there is little improvement in the results. Let's try 20% and 30% dropout rate:

| AUC  | F1-score | Precision | Recall | Accuracy |
| :--: | :------: | :-------: | :----: | :------: |
| 0.8  |   0.83   |   0.80    |  0.82  |   0.77   |
|      |   0.64   |   0.69    |  0.66  |          |

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7998 |   0.82   |   0.83    |  0.81  |   0.77   |
|        |   0.65   |   0.64    |  0.67  |          |

The results show that whether we use dropout method, the accuracy will not increase with a decrease on F1-score. Combined with L1 and L2 regularization, the model may no need further regularizations. And the predicting results in the training set says:

|  AUC   | F1-score | Precision | Recall | Accuracy |
| :----: | :------: | :-------: | :----: | :------: |
| 0.7713 |   0.78   |   0.88    |  0.70  |   0.70   |
|        |   0.52   |   0.41    |  0.69  |          |

## Conclusion

### Comparison of Models

1. First, for linear models: last square, LDA, logistic regression, softmax, the data is not well linearly separable the data has a small overlap between the two categories, which is difficult to distinguish by linearity and simple nonlinearity (regularization). The results of logistic regression and softmax regression are theoretically the same, but the learning rate, number of iterations, and termination conditions are different, resulting in slightly different results. For LDA, the model performs best for the data set with normal distribution, and after PCA, the data basically conforms to the normal distribution, so it stands out among the linear models.
2. Secondly, for the least square & gaussian kernel, the linear model is converted to nonlinear by the transformation of the kernel function, so the f1-score has a significant improvement, but for the AUC, we can see from the ROC picture that it has a high AUC for some thresholds, but there are also thresholds that make it perform badly. In addition, since the mean and variance are randomly generated, it also leads to the instability of the model.
3. Then, for the KNN model, although it is also nonlinear, KNN has a good performance with a large number of samples and a small number of features (the number of samples is much larger than the number of features). For the dataset of this task, the sample size is relatively small and the number of features is too large, so the KNN model does not perform well.
4. For the nn model, it can accurately grasp the nonlinear classification boundary, so the result is significantly better than the linear classifier. However, due to the limited information provided by the data, the real nonlinear boundary is not fully explored, so the effect cannot reach the peak.
5. For the tree model, the amount of data for this task is limited, and it is a problem of exploring the nonlinear boundary, the tree model is a better choice. But for single decision tree, there still appears the problem of too large variance, so the random forest model combines the strength of single tree and makes up for the disadvantage of too large variance, so the random forest model performs the best among the nine models.

### Pros and Cons of each Model

1. KNN

   Pros: can fit nonlinear classification boundary data

   Cons: not very good performance for data with few dimensions

2. Least squares classification.

   Pros: fast, high interpretability

   Cons: It can only be a linear classifier and is easily affected by outliers

3. Least squares classification with Gaussian basis functions.

   Pros: compared to Least squares classification can be generated to do non-linear classification boundary 

   Cons: There is a certain randomness in generating basis functions 

4. LDA

   Pros: if the data is Gaussian distributed, the classification effect is higher than other linear classifiers

   Cons: can only do linear classifier

5. Logistic regression with/without regularization (L2) \ Softmax regression with/without regularization (L2)

   Pros: if the data is not Gaussian distributed, the classification effect is higher than other linear classifiers; and regularization can be added

   Cons: can only do linear classifier

6. Decision tree

   Pros: can generate nonlinear classification boundaries and is not affected by data dimensionality

   Cons: the results have large variance

7. Random forest

   Pros: can generate non-linear classification boundaries, and is not affected by data dimensionality, with small variance

   Cons: long training time 

8. Neural networks

   Pros: can fit all kinds of classification boundaries

   Cons: requires a lot of information from the training set

## Appendix

### References:

[1] Source Data: https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

[2] Prove and application of SoftMax method: https://zhuanlan.zhihu.com/p/98061179
