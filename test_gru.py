import numpy
from gru import nn

data = numpy.array([[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1]])
labels = numpy.array([[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1]])

net = nn(3,5,2)

act = net.predict(data)
print act[-1]
d = net.compute_gradients(data,labels)
print d
#n = net.numerical_gradients(data[:4,:],labels[:4,:])
#print n