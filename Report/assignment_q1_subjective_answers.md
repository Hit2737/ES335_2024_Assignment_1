# assignment_q1_subjective_answers

Our model handles all 4 cases of features and output types. It also handles features of multiple types (discrete or real) in the same data-set.

The different criterion that can be used for selecting the optimum feature are

Entropy (Discrete Outputs / Classification) :
   
   $$
   Ent(x) = -\sum_i p(x_i)\log(p(x_i))
   $$
    
Gini Index (Discrete Outputs / Classification) :
   
   $$
   Gini(x) = 1-\sum_i (p(x_i))^2 \\
   $$
    
Variance (Real Outputs / Regression) :
   
   $$
   Var(x) = E((x-E(x))^2)
   $$
    
   where E() stands for the Expectation, xi is the value of the ith class, and p() is the probability mass function of x.
    

We find the attribute that reduces the criterion the most for every split, and in turn maximises the Gain, given (for a split based on attribute A) as 

$$
Gain(S,A) = L(S) - \sum_{v \in A} \frac{|S_v|}{|S|}L(S_v)
$$

Here ‘L’ is one of the three criterion written above, and ‘S’ is the set containing all possible events for the node that we want to branch further. ‘A’ is any attribute.

Upon running [`usage.py`](http://usage.py) we get this output :

[Output](https://www.notion.so/Output-6ab2c91c316749dbab56f35b4803e765?pvs=21)

Without printing the Tree structure (and instead plotting it via `graphviz`) , we get this output : 

##### Test 1
```
Criteria : information_gain
RMSE:  0.7340963358754425
MAE:  0.4042761530548207
Criteria : gini_index
RMSE:  0.7340963358754425
MAE:  0.4042761530548207
```

##### Test 2
```
Criteria : information_gain
Accuracy:  0.9
Precision:  1.0
Recall:  1.0
Precision:  0.8181818181818182
Recall:  0.9
Precision:  0.8333333333333334
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.3333333333333333
Criteria : gini_index
Accuracy:  0.8333333333333334
Precision:  1.0
Recall:  0.9
Precision:  0.6666666666666666
Recall:  1.0
Precision:  1.0
Recall:  0.6
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.3333333333333333
```

##### Test 3
```
Criteria : information_gain
Accuracy:  0.9666666666666667
Precision:  0.875
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.8888888888888888
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Criteria : gini_index
Accuracy:  0.9666666666666667
Precision:  0.875
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  0.8888888888888888
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
```

##### Test 4
```
Criteria : information_gain
RMSE:  0.0
MAE:  0.0
Criteria : gini_index
RMSE:  0.0
MAE:  0.0
```

