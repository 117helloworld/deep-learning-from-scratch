import numpy as np
from func import softmax,sigmoid,relu,cross_entropy_error,square_mean_error,numerical_gradient

class Net:
    def __init__(self,shape,weight_std=0.01): 
        #shape is [input,hidden1,...,output]. input=x, output=t
        self.params={}
        self.shape=shape
        for i in range(len(shape)-1):
            self.params['W'+str(i)]=weight_std*np.random.randn(shape[i],shape[i+1])
            self.params['b'+str(i)]=np.zeros(shape[i+1])

    def predict(self,x):
        if len(x[0])!=self.shape[0]:
            print('shape size (input layer) error')
        z=x
        for i in range(len(self.shape)-1):
            Wi,bi=self.params['W'+str(i)],self.params['b'+str(i)]
            a=np.dot(z,Wi)+bi
            print(a)
            if i==len(self.shape)-2: #last step
                z=softmax(a)
                y=z
            else:
                z=sigmoid(a)
        return y

    def loss(self,x,t):
        if len(t[0])!=self.shape[len(self.shape)-1]:
            print('shape size (output layer) error')   
        y=self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)

        accu=np.sum(y==t)/float(x.shape[0])
        return accu

    def numerical_gradient(self,x,t):
        def loss_W(W):
            return self.loss(x,t)
        grads={}
        for i in range(len(self.shape)-1):
            grads['W'+str(i)]=numerical_gradient(loss_W,self.params['W'+str(i)])
            grads['b'+str(i)]=numerical_gradient(loss_W,self.params['b'+str(i)])
        return grads
    
    def gradient(self,x,t):
        pass
#leave for now