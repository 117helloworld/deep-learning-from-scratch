# coding: utf-8
import numpy as np
import pickle
from mnist import load_mnist

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    c=np.max(x)
    exp=np.exp(x-c)
    sum_exp=np.sum(exp)
    return exp/sum_exp

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("C:\\Users\\math2\\OneDrive\\ドキュメント\\VSCode\\Python\\deep_learning_from_scratch\\ch03\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x): # x is matrix
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)
    return y


x, t = get_data() # x is 行列. t is the answer.
network = init_network()
accuracy_cnt = 0
y = predict(network, x)
print(np.shape(y))


batch_size=100
for i in range(0,len(x),batch_size):
    y=predict(network,x[i:i+batch_size])
    p=np.argmax(y,axis=1)
    T=t[i:i+batch_size]
    accu= p==T
    accuracy_cnt+=sum(accu)

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))