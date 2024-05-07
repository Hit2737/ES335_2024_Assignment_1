# assignment_q3_subjective_answers

1.  The decision tree for the automotive efficiency problem is trained in the `auto-efficiency.py` file. The code in the file demonstrates how to train a decision tree model (which is similar to what we did in question 2 part a) and part b) using a dataset of automotive features and their corresponding efficiency values. It shows how to preprocess the data, split it into training and testing sets, train the decision tree model, and evaluate its performance.

2.  We have compared the performance of the model implemented by us with the decision tree module from scikit-learn. By using scikit-learn's decision tree module, we have trained a decision tree model on the same dataset. This allows us to compare the accuracy, precision, recall, and other performance metrics of the two models. The comparison helps us assess the effectiveness and efficiency of our model in relation to the scikit-learn decision tree module. And the results are as follows (for the test data):-

The data has `‘?’` wherever the value has unknown. After converting those to `nan` and dropping some rows, we can use the data.

The output of `auto-efficient.py` is as follows :

```
cylinders  ...  origin
0          8  ...       1
1          8  ...       1
2          8  ...       1
3          8  ...       1
4          8  ...       1

[5 rows x 7 columns]

0    18.0
1    15.0
2    18.0
3    16.0
4    17.0
Name: mpg, dtype: float64

rmse of naive regression: 8.06826225294598
rmse of our model: 4.296095669718226
rmse of sklearn model: 4.382388998404105
```

First, I’, printing `X.head()` and `y.head()` and then the Root-Mean-Squared-Error of

1. The naive regression model : outputs the mean of `y_train`
2. Our Decision Tree model
3. The `sklearn.tree.DecisionTreeRegressor` model from Sci-kit-learn.

As you can see, the RMSE of the prediction from our model and the model implemented in sci-kit-learn is similar.
