from re import S
import numpy as np

#activation functions
def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    c=np.max(x)
    y=np.exp(x-c)
    s=np.sum(y)
    return y/S

#loss function
def sum_squared_error(y,t):
    return 0.5*np.sum(np.square(y-t))

def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))
