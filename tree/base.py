"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
from graphviz import Digraph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if __name__=='__main__':
    from utils import *
else:
    from tree.utils import *

np.random.seed(42)
i = 0
class DecisionNode:
    attribute: str
    features_left:list
    X: pd.DataFrame
    y: pd.Series
    children: dict
    id:int
    max_depth:int
    def __init__(self,X,y,criterion,features_left=[],max_depth=0,Gviz=None,id=i,split_value=None):
        assert len(y) > 0
        self.X = X
        self.y = y
        if check_ifreal(y):
            criterion = variance
        self.criterion = criterion
        self.features_left = features_left
        self.Gviz = Gviz
        self.children = {}
        self.id = id
        self.split_value = split_value
        self.max_depth = max_depth
        self.attribute = None
        self.gain = 'No-split-No-gain'
        self.create_ymp()

    def breed(self,perfect=True):
        global i
        if self.max_depth <=0 or len(self.features_left)==0:
            self.create_ymp()
            return
        attribute = opt_split_attribute(
            self.X,
            self.y,
            self.criterion,
            pd.Series(self.features_left),
            perfect
        )
        attr = self.X[attribute]
        gain = information_gain(self.y,attr,self.criterion,perfect)
        assert not pd.isna(gain) , f"y:\n{self.y}\nattr:\n{attr}\ncrit:\n{self.criterion}\nperfect:\n{perfect}\n"
        if gain <= 0:
            self.create_ymp()
            return
        self.gain = gain
        self.attribute = attribute
        fl = self.features_left.copy()
        if check_ifreal(attr):
            if perfect:
                self.split_value = perfect_split(self.y,attr,self.criterion)
            else:
                self.split_value = heur_split(self.y,attr)
        else:
            fl.remove(attribute)
        X_g,y_g = split_data(self.X,self.y,attribute,self.split_value,perfect)
        for a in X_g.keys():
            if a in self.children:
                self.children[a].X = X_g[a]
                self.children[a].y = y_g[a]
                self.children[a].features_left = fl
            elif len(y_g[a]):
                i += 1
                self.children[a] = DecisionNode(
                    X_g[a],
                    y_g[a],
                    self.criterion,
                    fl,
                    self.max_depth-1,
                    self.Gviz,
                    i
                )

    def __repr__(self,indent=0):
        if len(self.children) == 0:
            return '\t'*indent + str(self.ymp)
        decision = "?(" + str(self.attribute)
        if self.split_value is not None:
            decision += " > " + str(np.around(self.split_value,2))
        decision += "):"
        s = ''
        s += "\t"*indent + decision + '\n'
        for a in self.children:
            s += '\t'*(indent+1) + str(a) + ':' + '\n' + self.children[a].__repr__(indent + 2) + "\n"
        return s

    def plot(self):
        label = f"node {self.id}\n"
        if len(self.children) > 0:
            label += "decision : "
            label += str(self.attribute)
            if self.split_value is not None:
                label += f' > {np.round(self.split_value,2)}'
            label += f"\n gain = {self.gain}"
        else:
            label += f'y = {self.ymp}'
        if self.criterion is entropy:
            label += '\nentropy'
        elif self.criterion is gini_index:
            label += '\ngini_index'
        else:
            label += '\nvariance'
        label += f' = {self.criterion(self.y)}'
        self.Gviz.node(str(self.id),label)

    def plot_children(self):
        for a in self.children:
            child = self.children[a]
            child.plot()
            self.Gviz.edge(str(self.id),str(child.id),str(a))

    def propogate(self):
        self.breed()
        self.plot()
        for child in self.children.values():
            child.propogate()
        self.plot_children()
        
    def create_ymp(self):
        if check_ifreal(self.y):
            self.ymp = self.y.mean()
        else:
            self.ymp = self.y.mode().values[0]
        
    def predict(self,X:pd.DataFrame)->pd.Series:
        y_pred = pd.Series(data=[self.ymp]*len(X.index),index=X.index)
        if len(self.children) == 0:
            n = len(X.index)
            return y_pred
        X_g,y_g = split_data(X,y_pred,self.attribute,self.split_value)
        y_pred = pd.concat([self.children[a].predict(X_g[a]) for a in X_g if len(X_g[a])])
        return y_pred


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=5):
        D = {
            "information_gain":entropy,
            "gini_index":gini_index,
        }
        self.criterion = D[criterion]
        self.max_depth = max_depth
        self.g = Digraph()
        self.R = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        global i
        i = 0
        self.g.clear()
        if check_ifreal(y):
            self.criterion = variance
        R = DecisionNode(
            X,
            y,
            self.criterion,
            list(X.columns),
            self.max_depth,
            self.g
        )
        R.propogate()
        self.R = R
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        return self.R.predict(X)

    def plot(self,graph=False) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if graph:
            self.GVizPlot(view=True)
        else:
            print(self.R)

    def GVizPlot(self,file_folder=None,view=False):
        """
        file_folder : 2 tuple of strings file and folder.
        """
        if self.R is None:
            print("Graph empty.")
            return
        if file_folder is not None:
            file,folder = file_folder
            self.g.render(file,folder,format='png',view=view)
        elif view:
            self.g.view(f'graph_{i}')

if __name__=='__main__':
    print("Tests\n")
    from sklearn.datasets import load_iris as load
    data = load()
    X,y = data.data,data.target
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    X = pd.DataFrame(X)
    X.columns = ["sepal length","sepal width","petal length","petal width"]
    y = pd.Series(y)
    test_size = len(X)//3
    X_train,y_train = X[:-test_size],y[:-test_size]
    X_test,y_test =X[-test_size:],y[-test_size:]
    T = DecisionTree('information_gain')
    T.fit(X_train,y_train)
    y_pred = T.predict(X_test)
    correct = 0
    total = 0
    for I in y_pred.index:
        correct = correct + (y_test[I]==y_pred[I])
        total += 1
    print("accuracy:",100*correct/total,'%')
    print(T.R)
    import os
    folder = os.getcwd() #put your folder of choice here (full path)
    file = 'graph'
    T.GVizPlot((file,folder))
