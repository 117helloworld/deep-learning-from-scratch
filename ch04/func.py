import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    c=np.max(x)
    exp=np.exp(x-c)
    return exp/np.sum(exp)

def cross_entropy_error(y,t):
    return -np.sum(t*np.log(y))

def square_mean_error(y,t):
    return 0.5*np.sum(np.square(y-t))

def numerical_gradient(f,x):
    grad=np.zeros_like(x)
    h=1e-4
    for i in range(x.size):
        tmp=x[i]
        x[i]=tmp+h
        delta_plus_h=f(x)
        x[i]=tmp-h
        delta_minus_h=f(x)
        x[i]=tmp

        grad[i]=(delta_plus_h-delta_minus_h)/(2*h)
    return grad
