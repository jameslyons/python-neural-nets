import numpy as np
from util import relu, drelu

class nn:
    ''' Nhidden is the number of hidden nodes. Nlayers is the number
    of hidden layers i.e. Nlayers = 1 implies a single hidden layer.'''
    def __init__(self,Nin,Nout,Nhidden,Nlayers):
        # initialise the weights and biases, and their velocities
        wstd = 0.2;
        self.weights = [np.random.randn(Nin,Nhidden)*wstd]
        self.wv = [np.zeros((Nin,Nhidden))] # the weight velocity
        self.biases = [np.zeros((1,Nhidden))]
        self.bv = [np.zeros((1,Nhidden))]
        for i in range(1,Nlayers):
            self.weights.append(np.random.randn(Nhidden,Nhidden)*wstd)
            self.wv.append(np.zeros((Nhidden,Nhidden)))
            self.biases.append(np.zeros((1,Nhidden)))
            self.bv.append(np.zeros((1,Nhidden)))
        self.weights.append(np.random.randn(Nhidden,Nout)*wstd)
        self.wv.append(np.zeros((Nhidden,Nout)))
        self.biases.append(np.zeros((1,Nout)))
        self.bv.append(np.zeros((1,Nout)))
        
        self.Nin = Nin
        self.Nout = Nout
        self.Nhidden = Nhidden
        self.Nlayers = Nlayers

    
    ''' do the feedforward prediction of a piece of data'''
# need to add non linearities    
    def predict(self,input):
        activation = [input]
        for i in range(self.Nlayers):
            act = np.dot(activation[i],self.weights[i]) + self.biases[i]
            activation.append(relu(act))
        out = np.dot(activation[-1],self.weights[-1]) + self.biases[-1]
        activation.append(relu(out))
        return activation        
        
    def compute_gradients(self,input,label):
        L,W = np.shape(input)
        activation = self.predict(input)
        delta = (label - activation[-1])*drelu(activation[-1])
        weight_grads = [1.0/L*np.dot(activation[-2].T,delta)]
        bias_grads = [np.mean(delta,1)]
        for i in range(self.Nlayers,0,-1):
            delta = np.dot(delta,self.weights[i].T)*drelu(activation[i])
            weight_grads.append(1.0/L*np.dot(activation[i-1].T,delta))
            bias_grads.append(np.mean(delta,1))
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads
    
    def numerical_gradients(self,input,label,small=0.0001):
        num_grad = []
        for i in range(len(self.weights)):
            H,W = np.shape(self.weights[i])
            num_grad.append(np.zeros((H,W)))
            for j in range(W):
                for k in range(H):
                    self.weights[i][k,j] += small
                    act1 = self.predict(input)
                    err1 = np.mean(np.sum(0.5*np.square(label - act1[-1]),1))
                    self.weights[i][k,j] -= 2*small
                    act2 = self.predict(input)
                    err2 = np.mean(np.sum(0.5*np.square(label - act2[-1]),1))
                    num_grad[i][k,j] = (err1-err2)/(2*small)
                    self.weights[i][k,j] += small
        return num_grad
                    
                    
    def backprop(self,input,label,momentum=0.9,learningRate=0.01):
        weight_grads,bias_grads = compute_gradients(input,label)
        for i in range(len(self.weights)):
            self.wv[i] = momentum*self.wv[i] + LR*weight_grads[i]
            self.bv[i] = momentum*self.bv[i] + LR*bias_grads[i]
            self.weights[i] += self.wv[i]
            self.bias[i] += self.bv[i]
        return 0
    