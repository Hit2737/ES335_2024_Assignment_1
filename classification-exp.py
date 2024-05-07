import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

X = pd.DataFrame(X)
y = pd.Series(y)
total_len = len(X)
train_data_X = X.iloc[:int(0.7*total_len)]
train_data_y = y.iloc[:int(0.7*total_len)]

test_data_X = X.iloc[int(0.7*total_len):]
test_data_y = y.iloc[int(0.7*total_len):]

tree = DecisionTree()
tree.fit(train_data_X,train_data_y)

y_hat = tree.predict(test_data_X)
print("Question 2 : part a \n")
print("Accuracy :- {}".format(accuracy(y_hat,test_data_y)))
print("Precision :- ")
print("            For y=1 :- {}".format(precision(y_hat,test_data_y,1)))
print("            For y=0 :- {}".format(precision(y_hat,test_data_y,0)))
print("Recall :- ")
print("            For y=1 :- {}".format(recall(y_hat,test_data_y,1)))
print("            For y=0 :- {}".format(recall(y_hat,test_data_y,0)))



print("\nQuestion 2 : part b \n")
kf = KFold(5)
opt_max_depth = 0
opt_accuracy = 0
splt_data = list(kf.split(train_data_X))
for max_depth in range(1,9):    # Here we found that 8 is the max depth where the accuracy becomes 100%. And that's why we are considering it as upper bound for max_depth.
    avg_acc = 0
    for train_ind, val_ind in splt_data:
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(train_data_X.iloc[train_ind],train_data_y.iloc[train_ind])
        y_val_hat = tree.predict(train_data_X.iloc[val_ind])
        acc = accuracy(y_val_hat,train_data_y.iloc[val_ind])
        avg_acc += acc/5
    if (np.round(opt_accuracy, decimals=10)<np.round(avg_acc, decimals=10)):
        opt_max_depth = max_depth
        opt_accuracy = avg_acc

print("Optimum Max depth is {}".format(opt_max_depth))
print("The average accuracy over validation data for max_depth {} is:- {}".format(opt_max_depth,opt_accuracy))
tree = DecisionTree(max_depth=opt_max_depth)
tree.fit(train_data_X,train_data_y)
y_hat = tree.predict(test_data_X)
print("After changing max_depth to {}".format(opt_max_depth))
print("Accuracy :- {}".format(accuracy(y_hat,test_data_y)))
print("Precision :- ")
print("            For y=1 :- {}".format(precision(y_hat,test_data_y,1)))
print("            For y=0 :- {}".format(precision(y_hat,test_data_y,0)))
print("Recall :- ")
print("            For y=1 :- {}".format(recall(y_hat,test_data_y,1)))
print("            For y=0 :- {}".format(recall(y_hat,test_data_y,0)))
