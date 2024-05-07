import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# Function to create fake data (take inspiration from usage.py)
def GenerateData(N=100,M=6,P=5,DiscreteInput=False,DiscreteOutput=False):
    if DiscreteInput:
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
    else:
        X = pd.DataFrame({i : pd.Series(np.random.randn(N)) for i in range(M)})
    if DiscreteOutput:
        y = pd.Series(np.random.randint(P,size=N),dtype='category')
    else:
        y = pd.Series(np.random.randn(N))
    return X,y

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def TimeIt(N,M,P,iterations=num_average_time):
    Mat = []
    for cases in range(4):
        i,j = bool(cases//2),bool(cases%2)
        ttf = [] # time to fit
        ttp = [] # time to predict
        acc = [] # accuracy
        for _ in range(iterations):
            T = DecisionTree('information_gain')
            X,y = GenerateData(N,M,P,i,j)
            t0 = time.time()
            T.fit(X,y)
            ttf.append(time.time() - t0)
            t0 = time.time()
            y_pred = T.predict(X)
            ttp.append(time.time()-t0)
            if j:
                acc.append(accuracy(y_pred,y))
            else:
                acc.append(rmse(y_pred,y))
        row = [
            i,
            j,
            np.mean(ttf),
            np.std(ttf),
            np.mean(ttp),
            np.std(ttp),
            np.mean(acc),
            np.std(acc)
        ]
        Mat.append(row)
    columns = ["Disc. In. ?","Disc. Out. ?","time to fit","std for fit",'time to predict','std for pred.','acc./RMSE','std for acc./RMSE']
    df = pd.DataFrame(Mat)
    df.columns = columns
    return df
    


# Function to plot the results
# ...

def plot_time(P=2,N_low=1,N_high=50):
    parameters = {
        x:{
            "time to fit":[],
            'time to predict':[],
            'acc./RMSE':[],
        } for x in range(4)}
    M = 4
    for logN in range(int(np.log2(N_low)),int(np.log2(N_high))):
        N = 2**logN
        df = TimeIt(N,M,P,10)
        for x in range(4):
            for col in ["time to fit",'time to predict','acc./RMSE']:
                i = df.index[x]
                v = df[col][i]
                parameters[x][col].append(v)
    for col in ["time to fit",'time to predict','acc./RMSE']:
        plt.figure()
        for x in parameters:
            plt.plot(range(int(np.log2(N_low)),int(np.log2(N_high))),parameters[x][col])
        plt.title(col)
        plt.xlabel('log(N)')
        plt.legend(['Re. In. ,Re. Out.','Re. In. ,Di. Out.','Di. In. ,Re. Out.','Di. In. ,Di. Out.'])
    pass

# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
if __name__=='__main__':
    print(TimeIt(10,3,2,10).to_string())
    plot_time()
    plt.show()
