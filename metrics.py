from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    y_hat = y_hat[y.index]
    return (y_hat==y).mean()

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    y_hat = y_hat[y.index]
    mask = (y_hat == cls)
    correct = (y_hat == y)
    masked_correct = correct[mask]
    return masked_correct.mean()


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    y_hat = y_hat[y.index]
    mask = (y == cls)
    correct = (y_hat == y)
    masked_correct = correct[mask]
    return masked_correct.mean()

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    y_hat = y_hat[y.index]
    er = y_hat - y
    ser = er**2
    mse = ser.mean()
    rmse = mse**0.5
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    y_hat = y_hat[y.index]
    er = y_hat - y
    aer = er.abs()
    mae = aer.mean()
    return mae
