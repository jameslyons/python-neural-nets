import numpy as np
from numpy import shape
from util import sigm, dsigm, tanh, dtanh

class nn:
    ''' Nhidden is the number of hidden nodes. Nlayers is the number
    of hidden layers i.e. Nlayers = 1 implies a single hidden layer.'''
    def __init__(self,Nin,Nhidden,Nout):
        # initialise the weights and biases, and their velocities
        wstd = 0.2;
        self.w1 = np.random.randn(Nin,Nhidden)*wstd
        self.w1v = np.zeros((Nin,Nhidden))
        self.b1 = np.zeros((1,Nhidden))
        self.b1v = np.zeros((1,Nhidden))
        
        self.wz = np.random.randn(2*Nhidden,Nhidden)*wstd
        self.wzv = np.zeros((2*Nhidden,Nhidden)) # the weight velocity
        self.bz = np.zeros((1,Nhidden))
        self.bzv = np.zeros((1,Nhidden)) 
        
        self.wr = np.random.randn(2*Nhidden,Nhidden)*wstd
        self.wrv = np.zeros((2*Nhidden,Nhidden)) # the weight velocity
        self.br = np.zeros((1,Nhidden))
        self.brv = np.zeros((1,Nhidden)) 
        
        self.wh = np.random.randn(2*Nhidden,Nhidden)*wstd
        self.whv = np.zeros((2*Nhidden,Nhidden)) # the weight velocity
        self.bh = np.zeros((1,Nhidden))
        self.bhv = np.zeros((1,Nhidden)) 
        
        self.w2 = np.random.randn(Nhidden,Nout)*wstd
        self.w2v = np.zeros((Nhidden,Nout)) # the weight velocity
        self.b2 = np.zeros((1,Nout))
        self.b2v = np.zeros((1,Nout)) 
        
        self.Nin = Nin
        self.Nout = Nout
        self.Nhidden = Nhidden
    
    ''' do the feedforward prediction of a piece of data'''   
    def predict(self,input):
        L = np.shape(input)[0]
        az = np.zeros((L,self.Nhidden))
        ar = np.zeros((L,self.Nhidden))
        ahhat = np.zeros((L,self.Nhidden))
        ah = np.zeros((L,self.Nhidden))
        
        a1 = tanh(np.dot(input,self.w1) + self.b1)
        x = np.concatenate((np.zeros((self.Nhidden)),a1[1,:]))
        az[1,:] = sigm(np.dot(x,self.wz) + self.bz)
        ar[1,:] = sigm(np.dot(x,self.wr) + self.br)
        ahhat[1,:] = tanh(np.dot(x,self.wh) + self.bh)
        ah[1,:] = az[1,:]*ahhat[1,:]
        
        for i in range(1,L):
            x = np.concatenate((ah[i-1,:],a1[i,:]))
            az[i,:] = sigm(np.dot(x,self.wz) + self.bz)
            ar[i,:] = sigm(np.dot(x,self.wr) + self.br)
            x = np.concatenate((ar[i,:]*ah[i-1,:],a1[i,:]))
            ahhat[i,:] = tanh(np.dot(x,self.wh) + self.bh)
            ah[i,:] = (1-az[i,:])*ah[i-1,:] + az[i,:]*ahhat[i,:]
 
        a2 = tanh(np.dot(ah,self.w2) + self.b2)
        return [a1,az,ar,ahhat,ah,a2]
        
    def compute_gradients(self,input,labels):
        [a1,az,ar,ahhat,ah,a2] = self.predict(input)
        error = (labels - a2)
        
        L = np.shape(input)[0]
        H = self.Nhidden
        dz = np.zeros((L,H))
        dr = np.zeros((L,H))
        dh = np.zeros((L,H))
        d1 = np.zeros((L,H))

        # this is ah from the previous timestep
        ahm1 = np.concatenate((np.zeros((1,H)),ah[:-1,:]))

        d2 = error*dtanh(a2)
        e2 = np.dot(error,self.w2.T)
        dh_next = np.zeros((1,self.Nhidden))
        for i in range(L-1,-1,-1):
            err = e2[i,:] + dh_next
            dz[i,:] = (err*ahhat[i,:] - err*ahm1[i,:])*dsigm(az[i,:])
            dh[i,:] = err*az[i,:]*dtanh(ahhat[i,:])
            dr[i,:] = np.dot(dh[i,:],self.wh[:H,:].T)*ahm1[i,:]*dsigm(ar[i,:])
            dh_next = err*(1-az[i,:]) + np.dot(dh[i,:],self.wh[:H,:].T)*ar[i,:] + np.dot(dz[i,:],self.wz[:H,:].T) + np.dot(dr[i,:],self.wr[:H,:].T)
            d1[i,:] = np.dot(dh[i,:],self.wh[H:,:].T) + np.dot(dz[i,:],self.wz[H:,:].T) + np.dot(dr[i,:],self.wr[H:,:].T)
        d1 = d1*dtanh(a1)
        # all the deltas are computed, now compute the gradients
        gw2 = 1.0/L * np.dot(ah.T,d2)
        gb2 = 1.0/L * np.sum(d2,1)
        x = np.concatenate((ahm1,a1),1)
        gwz = 1.0/L * np.dot(x.T,dz)
        gbz = 1.0/L * np.sum(dz,1)
        gwr = 1.0/L * np.dot(x.T,dr)
        gbr = 1.0/L * np.sum(dr,1)
        x = np.concatenate((ar*ahm1,a1),1)
        gwh = 1.0/L * np.dot(x.T,dh)
        gbh = 1.0/L * np.sum(dh,1)
        gw1 = 1.0/L * np.dot(input.T,d1)
        gb1 = 1.0/L * np.sum(d1,1)
        weight_grads = [gw1,gwr,gwz,gwh,gw2]
        bias_grads = [gb1,gbr,gbz,gbh,gb2]
        
        return weight_grads, bias_grads
        
    def numerical_gradients(self,input,label,small=0.0001):
        weight_grads = []
        bias_grads = []
        wstr = ['w1','wr','wz','wh','w2']
        bstr = ['b1','br','bz','bh','b2']
        
        for i in range(len(wstr)):
            w = self.getattr(wstr[i])
            b = self.getattr(bstr[i])
            H,W = np.shape(w)
            grad = np.zeros((H,W))
            for j in range(W):
                for k in range(H):
                    w[k,j] += small
                    act1 = self.predict(input)
                    err1 = np.mean(np.sum(0.5*np.square(label - act1[-1]),1))
                    w[k,j] -= 2*small
                    act2 = self.predict(input)
                    err2 = np.mean(np.sum(0.5*np.square(label - act2[-1]),1))
                    grad[k,j] = (err1-err2)/(2*small)
                    w[k,j] += small
            weight_grads.append(grad)
        return weight_grads, bias_grads