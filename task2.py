import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt


filename1 = "cl-train-1.csv"
filename2 = "cl-test-1.csv"
filename3 = "cl-train-2.csv"
filename4 = "cl-test-2.csv"

#Implementation of z = h(x) = wT * x
def zf(w,x):
	x = np.asarray(x)
	w = np.asarray(w)
	return np.dot(w.T,x)

#Implementation of the nonlinear logistic function
def sig(z):
	return 1/(1+np.exp(-z))

#Implementation of the differentiation
def derv(x,y,w):	
	sum = np.zeros((len(x),1))
	for i in range(len(y)):
		yi = np.asarray(y[i])
		xi = np.asarray(x[:,i])
		z = zf(w,xi)
		sigma = sig(z)
		aux = np.dot((sigma - yi)[0],xi)		
		aux2 = aux.reshape(x.shape[0],1)
		sum = np.add(sum,aux2)
	return sum

#Implementation of the logistic regression based in the auxiliar functions above
def logistic_r(x,y,w,max_it, learning_n):
	erro = np.zeros((1,max_it))
	deriv = derv(x,y,w)
	i = 0;
	while((i < max_it)):
		erro[0,i] = cross_entropy(x,y,w)
		i = i+1;
		deriv = derv(x,y,w)
		w = w - np.dot(learning_n,deriv)
	return w, erro

#Classifier function that defines the binary value (classification) for each point... 
#...from the output of h = wT * x but utilizing the new weigh values (After training)
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

#Defines the decision boundary for the linear case
def boundary(w,x):
	teta0 = w[0,0]
	teta1 = w[1,0]
	teta2 = w[-1,0]	
	bias = -teta0/teta2
	slope = -teta1/teta2
	bound_vec = np.array([bias,slope]).reshape(1,2)	
	x_bline = x[0:2,:]
	return np.dot(bound_vec,x_bline)

#Nonlinear transformation of the dataset input which is linearly non-separable
def x_transform(x):
	N = x.shape[1]
	x1_sqr = np.zeros((1,N))
	x2_sqr = np.zeros((1,N))
	for i in range(N):
		x1_sqr[0,i] = x[1,i]**2
		x2_sqr[0,i] = x[2,i]**2

	new_x = np.row_stack((x,x1_sqr,x2_sqr))
	return new_x

#Defines the decision boundary for the dataset input which is linearly non-separable
def boundary_sqr(xx1,xx2,w):
	teta0 = w[0,0]
	teta1 = w[1,0]
	teta2 = w[2,0]	
	teta3 = w[3,0]	
	teta4 = w[4,0]	
	N = xx1.shape[1]
	f = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			x1_i = xx1[i,j]
			x2_i = xx2[i,j]
			h = teta0 + teta1*(x1_i) + teta2*(x2_i) + teta3*(x1_i**2) + teta4*(x2_i**2)
			if (h >= 0):
				f[i,j] = 1
			else:
				f[i,j] = 0
	return f
def cross_entropy(x,y,w):
	N = x.shape[1]
	sum = 0
	for i in range(N):
		yi = np.asarray(y[i])
		xi = np.asarray(x[:,i])
		z = zf(w,xi)
		sigma = sig(z)
		aux = yi*np.log(sigma)+(1-yi)*np.log(1-sigma)
		sum = sum + aux
	return sum

def main():
	# "defining" # of iterations
	iterations_n = 1000

	#Loading data from the training and test sets
	train1 = genfromtxt(filename1, delimiter = ",")
	test1  = genfromtxt(filename2, delimiter = ",")
	train2 = genfromtxt(filename3, delimiter = ",")
	test2  = genfromtxt(filename4, delimiter = ",")

	#Separating the vectors x = (x1,x2) for each dataset (training and test)
	x  = train1[:,0:-1]
	x2 = test1[:,0:-1]
	x3 = train2[:,0:-1]
	x4 = test2[:,0:-1]

	#Separating the vectors y (expected classification) for each dataset (training and test)
	y  = train1[:,-1]
	y2 = test1[:,-1]
	y3 = train2[:,-1]
	y4 = test2[:,-1]

	#Defining the appends to be added to the x vectors
	ap1 = np.ones(len(y))
	ap2 = np.ones(len(y2))
	ap3 = np.ones(len(y3))
	ap4 = np.ones(len(y4))

	#Appending the '1' values to the x vectors
	x  = np.column_stack((ap1,x)).T
	x2 = np.column_stack((ap2,x2)).T
	x3  = np.column_stack((ap3,x3)).T
	x4  = np.column_stack((ap4,x4)).T

	#Setting initial weight vectors (in this case, all 0)
	w  = np.zeros((1,x.shape[0])).T
	w3 = np.zeros((1,x3.shape[0]+2)).T #inserting 2 extra features for nonlinear case

	#Nonlinear transformation for the nonlinear case
	new_x3 = x_transform(x3)
	new_x4 = x_transform(x4)
	
	#New weights for the linear case (max iteratin = 1000, learning-rate = 0.1)
	w2, erro1 = logistic_r(x,y,w,iterations_n,0.1)
	#New weights for the nonlinear case (max iteratin = 1000, learning-rate = 0.1)
	w4, erro2 = logistic_r(new_x3,y3,w3,iterations_n,0.1)


	h  = zf(w2,x)
	h2 = zf(w2,x2)
	h3 = zf(w4,new_x3)
	h4 = zf(w4,new_x4)

	c  = classifier(h2)
	c3 = classifier(h4)
      
#	print boundary_sqr(x3,y3,w4)
#	b_line = boundary(w4,x3)
	x1list = np.linspace(0.0, 1.0, iterations_n)
	x2list = np.linspace(0.0, 1.0, iterations_n)
	xx1, xx2 = np.meshgrid(x1list,x2list)
	print xx1.shape
	print xx2.shape
	F = boundary_sqr(xx1,xx2,w4)
	F = F.reshape(xx1.shape)

	cp = plt.contour(xx1,xx2,F, colors = 'green', linestyles = 'dashed')
#	plt.clabel(cp,inline = True, fontsize = 10)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.scatter(x4[1,:],x4[2,:], c = c3, s = 100, edgecolor = 'black')

	#plt.plot(x3[1,:].reshape(60,1),b_line[0,:])
	aux_x = np.linspace(1,iterations_n, iterations_n)
	plt.figure(2)

	print aux_x.shape
	print erro2.shape
	plt.plot(aux_x, erro2[0,:])

	plt.show()


main()



























