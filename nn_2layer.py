import numpy as np

def relu(x):
    return np.maximum(0,x)
	
def drelu(x):
    return np.array(x>0,dtype=int)
    
class nn:
    ''' Initialise the weights and biases for a simple 1 hidden layer nn'''
    def __init__(self,numIn,numHidden,numOut):
        # initialise the weights and biases, and their velocities
        wstd = 0.2;
        self.weight1 = np.random.randn(numIn,numHidden)*wstd
        self.w1v = np.zeros((numIn,numHidden)) # the weight velocity to use momentum
        self.bias1 = np.zeros((1,numHidden))
        self.b1v = np.zeros((1,numHidden)) # the bias velocity

        self.weight2 = np.random.randn(numHidden,numOut)*wstd
        self.w2v = np.zeros((numHidden,numOut)) # the weight velocity to use momentum
        self.bias2 = np.zeros((1,numOut))
        self.b2v = np.zeros((1,numOut)) # the bias velocity
        
        self.numIn = numIn
        self.numOut = numOut
        self.numHidden = numHidden
    
    
    ''' do the feedforward prediction of a piece of data'''
    def predict(self,input):
        hidden_act = relu(np.dot(input,self.weight1) + self.bias1)
        out = relu(np.dot(hidden_act,self.weight2) + self.bias2)
        return hidden_act,out        
        
        
    ''' compute the gradients of all the weights and biases using backpropagation equations'''    
    def compute_gradients(self,input,label):
        L,W = np.shape(input)
        hidden_act,out = self.predict(input)
        output_delta = (label - out)*drelu(out)
        weight1_grad = 1.0/L*np.dot(hidden_act.T,output_delta)
        bias1_grad = np.mean(output_delta,1)
        
        hidden_delta = np.dot(output_delta,self.weight2.T)*drelu(hidden_act)
        weight2_grad = 1.0/L*np.dot(input.T,hidden_delta)
        bias2_grad = np.mean(hidden_delta,1)

        return (weight1_grad, weight2_grad), (bias1_grad,bias2_grad)
    

    ''' compute the gradients for all the weights and biases numerically. This is to check
        that the backpropagation equations are correctly implemented. This function only checks
        the weights, the bias gradients are checked in the same way.'''
    def numerical_gradients(self,input,label,small=0.0001):
        wstr = ["weight1","weight2"]
        for i in range(len(wstr)):
            w = getattr(self,wstr[i])
            H,W = np.shape(w)
            num_grad = np.zeros((H,W))
            for j in range(W):
                for k in range(H):
                    w[k,j] += small
                    out1 = self.predict(input)[-1]
                    err1 = np.mean(np.sum(0.5*np.square(label - out1),1))
                    w[k,j] -= 2*small
                    out2 = self.predict(input)[-1]
                    err2 = np.mean(np.sum(0.5*np.square(label - out2),1))
                    num_grad[k,j] = (err1-err2)/(2*small)
                    w[k,j] += small
        return num_grad
                    
                    
    ''' update the weights and biases with learningRate*gradient '''
    def update_weights(self,input,label,learningRate=0.01,momentum=0.9):
        (w1_grad, w2_grad), (b1_grad, b2_grad) = compute_gradients(input,label)
        
        self.w1v = momentum*self.w1v + learningRate*w1_grad
        self.b1v = momentum*self.b1v + learningRate*b1_grad
        self.weight1 += self.w1v
        self.bias1 += self.b1v

        self.w2v = momentum*self.w2v + learningRate*w2_grad
        self.b2v = momentum*self.b2v + learningRate*b2_grad
        self.weight2 += self.w2v
        self.bias2 += self.b2v
        
        return 0
    
