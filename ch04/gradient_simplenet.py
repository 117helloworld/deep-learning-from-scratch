# coding: utf-8
import numpy as np
from gradient_copy import gradient

def softmax(x): #x is vector
    c=np.max(x)
    l=np.exp(x-c)
    return l/np.sum(l)

def cross_entropy_error(y,t):
    delta=1e-4
    return -np.sum(t*np.log(y+delta))  
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)

dW = gradient(f, net.W)

print(dW)
