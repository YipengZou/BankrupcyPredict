# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:01:05 2022

@author: L
"""

import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE ,ADASYN, SVMSMOTE, SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


train = pd.read_csv("..\\bankruptcy data\\adj_imp_train.csv")
train_X = train.iloc[:,:-1]/10
train_y = train.iloc[:,-1]

test = pd.read_csv("..\\bankruptcy data\\adj_imp_test.csv")
test_X = test.iloc[:,:-1]/10
test_y = test.iloc[:,-1]


#%%

smo = BorderlineSMOTE(random_state=0)
# smo = ADASYN(random_state=0)
# smo = SVMSMOTE(random_state=0)
# smo = SMOTE(random_state=0)
# ros = RandomOverSampler(random_state=0)
# ros.fit(train_X, train_y)
# X_resampled, y_resampled = ros.fit_resample(train_X, train_y)

X_resampled, y_resampled = smo.fit_resample(train_X, train_y)

new = X_resampled.join(y_resampled)

new.to_csv("..\\bankruptcy data\\smote_train.csv",index=False)

#%%
pca = PCA(0.97)
pca.fit(train_X)

var = pca.explained_variance_ratio_[0:10] #percentage of variance explainedlabels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
plt.figure(figsize=(15,7))
plt.bar(['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'],var)
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')

newX = pca.transform(train_X)
new_testX = pca.transform(test_X)

newX = pd.DataFrame(newX)
new = newX.join(train_y)

new_testX = pd.DataFrame(new_testX)
new_test = new_testX.join(test_y)



# new.to_csv("..\\bankruptcy data\\pca_train.csv",index=False)
# new_test.to_csv("..\\bankruptcy data\\pca_test.csv",index=False)