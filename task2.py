import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt


filename1 = "cl-train-1.csv"
filename2 = "cl-test-1.csv"
filename3 = "cl-train-2.csv"
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
		if(h[0,j] >= 0):
			c[0,j] = 1
		else:
			c[0,j] = 0
		j = j+1;
	return np.asarray(c)

def boundary(w,x):
	teta0 = w[0,0]
	teta1 = w[1,0]
	teta2 = w[-1,0]	
	bias = -teta0/teta2
	slope = -teta1/teta2
	bound_vec = np.array([bias,slope]).reshape(1,2)	
	x_bline = x[0:2,:]
	return np.dot(bound_vec,x_bline)


def main():
	
	train1 = genfromtxt(filename1, delimiter = ",")
	test1  = genfromtxt(filename2, delimiter = ",")
	train2 = genfromtxt(filename3, delimiter = ",")
	test2  = genfromtxt(filename4, delimiter = ",")

	x  = train1[:,0:-1]
	x2 = test1[:,0:-1]
	x3 = train2[:,0:-1]
	x4 = test2[:,0:-1]

	y  = train1[:,-1]
	y2 = test1[:,-1]
	y3 = train2[:,-1]
	y4 = test2[:,-1]

	ap1 = np.ones(len(y))
	ap2 = np.ones(len(y2))
	ap3 = np.ones(len(y3))
	ap4 = np.ones(len(y4))



	x  = np.column_stack((ap1,x)).T
	x2 = np.column_stack((ap2,x2)).T
	x3  = np.column_stack((ap3,x3)).T
	x4  = np.column_stack((ap4,x4)).T

	w  = np.zeros((1,x.shape[0])).T
	w3 = np.zeros((1,x3.shape[0])).T

#	w2 = logistic_r(x,y,w,1000,0.1)
	w4 = logistic_r(x3,y3,w3,1000,0.1)

#	h  = zf(w2,x)
#	h2 = zf(w2,x2)
	h3 = zf(w4,x3)
	h4 = zf(w4,x4)

#	c  = classifier(h2)
	c2 = classifier(h3)

	print y3
	print c2        

	b_line = boundary(w4,x3)

	plt.scatter(x3[1,:],x3[2,:], c = y3, s = 50, edgecolor = 'black')

	plt.plot(x3[1,:].reshape(60,1),b_line[0,:])

	plt.show()

main()



























