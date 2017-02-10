import numpy as np
#import matplotlib.pyplot as plt
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

#	print "sum shape:"
#	print sum.shape

	for i in range(len(y)):
		yi = y[i]
		xi = x[:,i]
		z = zf(w,xi)
#		print "sigma shape:"
		sigma = sig(z)
#		print sigma.shape
#		print "xi shape:"
#		print xi.shape
		aux = np.dot((sigma - yi)[0],xi)
		sum = sum + aux
	return sum

def logistic_r(x,y,w,max_it, learning_n):

	deriv = derv(x,y,w)
	i = 0;
	
	while((i < max_it)):
		i = i+1;
		deriv = derv(x,y,w)
		w = w - np.dot(learning_n,deriv)
	
	return w

def classifier():

def main():
	
	train1 = genfromtxt(filename1, delimiter = ",")
	test1  = genfromtxt(filename2, delimiter = ",")
	train2 = genfromtxt(filename3, delimiter = ",")
	test2  = genfromtxt(filename4, delimiter = ",")

	x = train1[:,0:-1]
	y = train1[:,-1]
	ap1 = np.ones(len(y))
	x = np.column_stack((ap1,x)).T
	w = np.zeros((1,x.shape[0])).T
	
	print x.shape
	xi = x[:,0]
	print xi.shape
	print y.shape
	print w.shape
	print zf(w,x).shape



 	w2 = logistic_r(x,y,w,10,0.5)

	print "End:"
	print y
	print zf(w2,x)

#	print w.shape
#	print w2.shape
#	print x.shape
	#print zf(w2,x)

main()
