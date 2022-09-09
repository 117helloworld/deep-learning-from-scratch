from matplotlib import pyplot as plt
import numpy as np

#no batch
def gradient_no_batch(f,x): #f is function, x is vector
    h=0.00001
    grad=np.zeros(x.size)
    for i in range(x.size):
        tmp=x[i]

        x[i]=float(tmp)+h
        f_plus_h=f(x)

        x[i]=float(tmp)-h
        f_minus_h=f(x)

        x[i]=tmp
        grad[i]=(f_plus_h-f_minus_h)/(2*h)
    return grad

def gradient(f,x): #f is a f[unction, x is a matrix; x=(x0,x1,...)
    grad=[]
    for i in range(len(x)):
        grad.append(gradient_no_batch(f,x[i]))
    return np.asarray(grad)



def f(x):
    return np.sum(np.square(np.square(x)))

x=np.array([[1.00,0,2.00],[3.00,0,5.00]])
print(f(x))
z=gradient(f,x)
print(z)

