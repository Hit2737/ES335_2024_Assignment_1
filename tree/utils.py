"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
from numpy import log

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_string_dtype(y):
        return False
    if len(y.unique()) < 10:
        return False
    return True

def entropy(Y: pd.Series,epsilon=1e-6) -> float:
    """
    Function to calculate the entropy
    """
    counts = Y.value_counts()
    counts = counts + epsilon #so that no probability is exactly 0, and we don't get errors.
    total = counts.sum()
    prob = counts/total
    plp = prob.apply(lambda x: x*log(x))
    ent = -plp.sum()
    return ent

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    (as described here : https://blog.quantinsti.com/gini-index/)
    """
    counts = Y.value_counts()
    total = counts.sum()
    prob = counts/total
    prob2 = prob*prob
    prob2sum = prob2.sum()
    Gini = 1-prob2sum
    return Gini

def variance(Y: pd.Series) -> float:
    if len(Y)<=1:
        return 0
    v = Y.var()
    assert not pd.isna(v) , f"{Y}\n\n{v}"
    return v

def heur_split(Y: pd.Series, attr: pd.Series,epsilon = 1e-6):
    """
    Returns a good guess for the value used for comparision for a continuous attribute.
    This isn't from some research paper or anythong. I'm just making this up as I go.
    There is the obvious "correct" O(N^2) solution to this. But I don't want to do that.
    Intuition : Take 2 clusters' mean and apply section formula in the ratio of standard deviations to get a good partition
    If someone wants to do this correctly, please go ahead. Also link some research paper (like in the link below)
    https://link.springer.com/content/pdf/10.1023/A:1022638503176.pdf
    (I haven't read it yet)
    """
    groups = attr.groupby(Y, observed=False)
    means = groups.mean()
    stds = groups.std() + epsilon
    sizes = groups.count() + epsilon
    weights = 1/(sizes*stds)
    middle_value = (means*weights).sum()/weights.sum()
    return middle_value

def perfect_split(Y: pd.Series, attr: pd.Series, criterion= None):
    if criterion is None and check_ifreal(Y):
        criterion = variance
    else:
        criterion = entropy
    attr_s = attr.sort_values(inplace=False)
    idx = attr_s.index
    y = Y[idx]
    vbest = attr_s[idx[0]]
    Gbest = 0
    n = len(idx)
    init_crit = criterion(y)
    for i in range(1,len(idx),1):
        v = (attr_s[idx[i-1]]+attr_s[idx[i]])/2
        G = init_crit - (criterion(y[idx[:i]])*(i/n) + criterion(y[idx[i:]])*(1-i/n))
        if G > Gbest:
            vbest = v
            Gbest = G
    return vbest


def information_gain(Y: pd.Series, attr: pd.Series, criterion=None, perfect=True) -> float:
    """
    Function to calculate the information gain
    Y : target values
    attr : values of specific attribute
    gain : the Information gain given by 
    G(S,A) = Ent(S) - sum_{v in A}((S_v/S)*Ent(S_v)) 
    where A is the new attribute 'attr' and S is the grouping done till now.
    """
    assert len(Y) > 0
    if criterion is None:
        if check_ifreal(Y):
            criterion = variance
        else:
            criterion = entropy
    if check_ifreal(attr):
        if perfect:
            middle_value = perfect_split(Y,attr,criterion)
        else:
            middle_value = heur_split(Y,attr)
        attr = (attr > middle_value)
    groups = Y.groupby(attr,observed=False)
    gain = criterion(Y)
    assert not pd.isna(gain)
    for group in groups:
        attr_g,y_g = group
        if len(y_g):
            rent = criterion(y_g)*len(y_g)/len(Y)
            gain = gain-rent
    return gain

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series,perfect=True):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    features = features.values
    if len(features) == 0:
        print("Please provide atleast one feature in opt_split_attribute")
        raise ValueError
    bestFeat = features[0]
    bestGain = information_gain(y,X[bestFeat],criterion,perfect)
    for feature in features:
        Gain = information_gain(y,X[feature],criterion,perfect)
        if Gain > bestGain:
            bestFeat = feature
            bestGain = Gain
    return bestFeat

def split_data(X: pd.DataFrame, y: pd.Series, attribute:str, value=None, perfect = True):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    features: the list of features still left to use
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    attr = X[attribute]
    if check_ifreal(attr):
        if value is None:
            if not perfect:
                value = heur_split(y,attr)
            else:
                value = perfect_split(y,attr)
        attr = (attr > value)
    X.drop(attribute,axis=1)
    X_g = X.groupby(attr, observed=False)
    y_g = y.groupby(attr, observed=False)
    X_g = {a:x for (a,x) in X_g}
    y_g = {a:x for (a,x) in y_g}
    return X_g,y_g

if __name__=="__main__":
    print("Tests\n")
    from sklearn.datasets import load_iris as load
    from matplotlib.pyplot import show,hist,title,figure
    data = load()
    print(data.DESCR)
    X,y = data.data,data.target
    X = pd.DataFrame(X)
    X.columns = ["sepal length","sepal width","petal length","petal width"]
    y = pd.Series(y)
    print("\nX :")
    print(X[:5])
    print("\ny :")
    print(y[:5])

    print("\nTest for check_ifreal :")
    if not check_ifreal(X['sepal length']) or check_ifreal(y):
        print("failed")
    else :
        print("passed")

    print("\nTest for entropy :")
    if (entropy(y[:5]) != 0) or (entropy(y)<=0):
        print("failed")
    else:
        print("passed")

    print("\nTest for gini_index :")
    if gini_index(y[:5]) > 0:
        print("failed")
    else:
        print("passed")
    
    print("\nTest for information_gain :")
    if information_gain(y,X["petal width"],entropy) <= 0:
        print("failed")
    else:
        print("passed")
    print(information_gain(y,X["petal width"],entropy))
    figure()
    for group in X["petal width"].groupby(y, observed=False):
        hist(group[1],10)
    title("petal width")
    
    print("\nTest for opt_split_attribute :")
    opt_attr = opt_split_attribute(X,y,entropy,X.columns)
    if opt_attr not in ['petal width','petal length']:
        print("failed")
    else:
        print("passed")
    print("Optimum attribute :",opt_attr)
    gain = information_gain(y,X[opt_attr],entropy,True)
    print("Gain :",gain)

    print("\nTest for split_data :")
    X_g,y_g = split_data(X,y,opt_attr,perfect=True)
    ent = 0
    for a in y_g:
        ent = ent + entropy(y_g[a])*len(y_g[a])/len(y)
    if abs((entropy(y) - ent) - gain) > 0.1:
        print("failed")
        print(entropy(y) - ent,gain)
    else:
        print("passed")
    for a in X_g:
        print(a,":",len(X_g[a]),len(y_g[a]))
    
    show()
