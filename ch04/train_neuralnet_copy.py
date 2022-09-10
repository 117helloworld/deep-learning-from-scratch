from base64 import b16decode
from tkinter import YView
import numpy as np
from matplotlib import pyplot as plt
from mnist import load_mnist
from func import sigmoid,relu,softmax,cross_entropy_error,square_mean_error


class Net:
    def __init__(self,shape,weight_std): #shape is [input,hidden1,...,output]
        self.params={}
        self.shape=shape
        for i in range(len(shape)):
            self.params['W'+str(i)]=weight_std*np.random.randn(shape[i],shape[i+1])
            self.params['b'+str(i)]=np.zeros(1,shape[i+1])

    def predict(self,x):
        z=x
        for i in range(len(self.shape))
            Wi,bi=self.params['W'+str(i)],self.params['b'+str(i)]
            a=np.dot(z,Wi)+bi
            if i==len(self.shape)-1: #last step
                z=softmax(a)
                y=z
            else:
                z=sigmoid(a)
        return y

    def loss(self,x,t):
        y=self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)

        accu=np.sum(y==t)/float(x.shape[0])
        return accu



train,test=load_mnist(normalize=True,one_hot_label=True)
