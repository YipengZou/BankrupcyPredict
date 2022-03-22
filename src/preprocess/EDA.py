
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% Read data
train = pd.read_csv("..\\bankruptcy data\\train.csv")
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test = pd.read_csv("..\\bankruptcy data\\test.csv")
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

#%% distribution of feature
plt.subplot(121)
plt.pie([train_y[train_y==0].shape[0],train_y[train_y==1].shape[0]],labels=[0,1],autopct='%3.2f%%',textprops={'fontsize':18,'color':'k'})
plt.title("Train Set")
plt.subplot(122)
plt.pie([test_y[test_y==0].shape[0],test_y[test_y==1].shape[0]],labels=[0,1],autopct='%3.2f%%',textprops={'fontsize':18,'color':'k'})
plt.title("Test Set")


#%% heatmap
train_X.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

corr = train_X.corr() # We already examined SalePrice correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.7) | (corr <= -0.7)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)