import numpy as np
def euclidean_error(input,label):
    return label - input

def crossentropy_error(input,label):
    return label - input
	
def sigm(x):
    return 1/(1+np.exp(-x))
	
def dsigm(x):
    return x*(1-x)
	
def tanh(x):
    return np.tanh(x)
	
def dtanh(x):
    return (1-np.square(x))
	
def relu(x):
    return np.maximum(0,x)
	
def drelu(x):
    return np.array(x>0,dtype=int)
	