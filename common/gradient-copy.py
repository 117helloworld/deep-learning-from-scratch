import numpy as np
def gradient_1d(f,x): # f is function, x is positional vector(point)
    h=1e-4
    grad=np.zeros(x.shape) #grad is vector
    
    for i in range(x.size):
        x_i=x[i]

        x[i]=x_i+h
        x_pl_h=f(x)
        x[i]=x_i-h
        x_mi_h=f(x)

        grad[i]=(x_pl_h-x_mi_h)/(2*h)

    return grad

def gradient(f,X): # X is multiple positional vector made of X[0],X[1],...
    if X.ndim==1:
        return gradient_1d(f,X)
    else:
        grad=np.zeros(X.shape)

        for i in range(len(X)): # len(X) is number of points
            grad[i]=gradient_1d(f,X[i])
        return grad

def f(X): #X=np.array([[0,1],[5,6]])
    return np.dot(X[0],X[1])
    # return np.sum([np.square(X[0]),np.square(X[1])],axis=0)

# f example
a=np.arange(0,1,0.1)
X=np.array([a,a]).T
print(X)
print(f(X))

print(gradient(f,X))
