import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt


filename1 = "cl-train-1.csv"
filename2 = "cl-test-1.csv"
filename3 = "cl-train-1.csv"
filename4 = "cl-test-2.csv"

def zf(w,x):
	x = np.asarray(x)
	w = np.asarray(w)
	return np.dot(w.T,x)

def sig(z):
	return 1/(1+np.exp(-z))


def derv(x,y,w):	
	sum = np.zeros((len(x),1))

	for i in range(len(y)):
		yi = np.asarray(y[i])
		xi = np.asarray(x[:,i])
		z = zf(w,xi)
		sigma = sig(z)
		aux = np.dot((sigma - yi)[0],xi)		
		aux2 = aux.reshape(3,1)
		sum = np.add(sum,aux2)

	return sum

def logistic_r(x,y,w,max_it, learning_n):

	deriv = derv(x,y,w)
	i = 0;
	
	while((i < max_it)):
		i = i+1;
		deriv = derv(x,y,w)
		w = w - np.dot(learning_n,deriv)
	
	return w

def classifier(h):
	
	count = h.shape[1]	
	c = np.zeros((1,h.shape[1]))
	j = 0
	for i in range(h.shape[1]):
		if(h[0,j] >= 0.5):
			c[0,j] = 1
		else:
			c[0,j] = 0
		j = j+1;
	return np.asarray(c)
		

def main():
	
	train1 = genfromtxt(filename1, delimiter = ",")
	test1  = genfromtxt(filename2, delimiter = ",")
	train2 = genfromtxt(filename3, delimiter = ",")
	test2  = genfromtxt(filename4, delimiter = ",")

	x  = train1[:,0:-1]
	x2 = test1[:,0:-1]
	


	y  = train1[:,-1]
	y2 = test1[:,-1]


	ap1 = np.ones(len(y))
	ap2 = np.ones(len(y2))

	x  = np.column_stack((ap1,x)).T
	x2 = np.column_stack((ap2,x2)).T

	w = np.zeros((1,x.shape[0])).T


	w2 = logistic_r(x,y,w,1000,0.1)

	print "x  shape " + str(x.shape)
	print "x2 shape " + str(x2.shape)
	print "w2 shape " + str(x2.shape)

	h  = zf(w2,x)
	h2 = zf(w2,x2)
	
	
	

	print "y:"
	print y2
	print "c: "
	c = classifier(h2)
	print c

	print np.subtract(y2,c)

	#plt.plot(x[1,:],x[2,:],'o')
	print y

	
	plt.scatter(x[1,:],x[2,:], c = y, s = 15, edgecolor = 'blue')

	plt.show()


main()



























