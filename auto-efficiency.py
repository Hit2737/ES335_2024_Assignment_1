import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"],na_values=['?'])

# Clean the above data by removing redundant columns and rows with junk values
data.dropna(axis="index",inplace=True)
data.drop_duplicates(inplace=True)
cars = data['car name']
data.drop('car name',axis='columns',inplace=True)
y = data['mpg']
X = data.drop('mpg',axis='columns',inplace=False)
columns = list(X.columns)
assert not (X.index!=y.index).sum()
print(X.head())
print(y.head())
idx = np.arange(len(X),dtype=np.int32)
np.random.shuffle(idx)
X,y = X.values.astype(np.float64),y.values.astype(np.float64)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
X = pd.DataFrame(X)
X.columns = columns
y = pd.Series(y)
X = X[:200]
y = y[:200]
test_size = int(len(X)*0.3)
X_train,y_train = X[:-test_size],y[:-test_size]
X_test,y_test = X[-test_size:],y[-test_size:]
print("rmse of naive regression:",((y_test-y_train.mean())**2).mean()**0.5)

# Compare the performance of your model with the decision tree module from scikit learn

# Our model
T = DecisionTree('information_gain')
T.fit(X_train,y_train)
y_pred = T.predict(X_test)
print("rmse of our model:",rmse(y_pred,y_test))

# Sklearn
Tskl = DecisionTreeRegressor()
Tskl.fit(X_train,y_train)
y_pred = Tskl.predict(X_test) # numpy array
y_pred = pd.Series(y_pred,index=y_test.index)
print("rmse of sklearn model:",rmse(y_pred,y_test))
