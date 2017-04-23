import sys
import math
import numpy as np
import tensorflow as tf
import helpers 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image
import os



TRAIN_RATE = 0.8
NUMBER_OF_CLASSES = 26
STARTING_LEARNING_RATE = 0.03
NUMBER_OF_EPOCHS = 10
BATCH_SIZE = 4
data_path = './chars74k-lite/'
path_letters_src = {}
path_letters_src[0] = './chars74k-lite/a/a_0.jpg'
data_info = {}
DATA_TOTAL_SIZE = 0

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(STARTING_LEARNING_RATE, global_step, NUMBER_OF_EPOCHS, 0.96, staircase=True)



def create_batchs(x,y, group_size):
    return_list_x = []
    return_list_y = []

    for i in xrange(0, len(x), group_size):
        item_x =  x[i:i+group_size]
        item_y =  y[i:i+group_size]
        return_list_x.append(item_x)
        return_list_y.append(item_y)
    return np.array(return_list_x), np.array(return_list_y)


j = 0
for i in next(os.walk(data_path))[1]:
  new_path = data_path + i
  data_info[j] = (i,len(next(os.walk(new_path))[2]))
  DATA_TOTAL_SIZE = DATA_TOTAL_SIZE +  int(data_info[j][1]*TRAIN_RATE)
  j=j+1

filelist = []

Y_labels = np.zeros((DATA_TOTAL_SIZE, NUMBER_OF_CLASSES))

print Y_labels.shape 
index = 0
for dir_letter in range(NUMBER_OF_CLASSES):
	for image_index in range(int(data_info[dir_letter][1]*TRAIN_RATE)):
		i_path = data_path+data_info[dir_letter][0]+'/'+data_info[dir_letter][0]+'_'+str(image_index)+'.jpg'
		filelist.append(i_path)
		Y_labels[index][dir_letter] = 1.0
#		X[index] =  np.asarray(Image.open(i_path))
		index = index + 1


X_train  = np.array([np.array(Image.open(fname)) for fname in filelist])
X_train  = np.reshape(X_train,(DATA_TOTAL_SIZE, 20, 20, 1))
X_train = X_train/255.0

X = tf.placeholder(tf.float32, [None, 20, 20, 1])

#W1 = tf.Variable(tf.random_normal(shape=[20*20,NUMBER_OF_CLASSES], stddev = 1.0/20*20))
W1 = tf.Variable(tf.truncated_normal([5,5,1,4], stddev = 0.1))
B1 = tf.Variable(tf.zeros([4])) #4 is the number of output channels
stride1 = 1 #output is still 20x20
Ycnv1 = tf.nn.conv2d(X,W1, strides = [1, stride1, stride1, 1], padding = 'SAME')
Y1 = tf.nn.relu(Ycnv1 + B1)

W2 = tf.Variable(tf.truncated_normal([4,4,4,8], stddev = 0.1))
B2 = tf.Variable(tf.zeros([8])) #4 is the number of output channels
stride2 = 1 #output is 10x10
Ycnv2 = tf.nn.conv2d(Y1,W2, strides = [1, stride2, stride2, 1], padding = 'SAME')
Y2 = tf.nn.relu(Ycnv2 + B2)


W3 = tf.Variable(tf.truncated_normal([4,4,8,12], stddev = 0.1))
B3 = tf.Variable(tf.zeros([12])) #4 is the number of output channels
stride3 = 1 #output is 5x5
Ycnv3 = tf.nn.conv2d(Y2,W3, strides = [1, stride3, stride3, 1], padding = 'SAME')
Y3 = tf.nn.relu(Ycnv3 + B3)


W4 = tf.Variable(tf.truncated_normal([20*20*12, 200], stddev = 0.1))
B4 = tf.Variable(tf.zeros([200]))

W5 = tf.Variable(tf.truncated_normal([200 ,NUMBER_OF_CLASSES],stddev = 0.1))
B5 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES])) 



#model
h1 = tf.nn.relu(tf.matmul(tf.reshape(Y3,[-1,20*20*12]), W4) + B4)

Ylogits = tf.matmul(h1, W5) + B5
Y  = tf.nn.softmax(Ylogits)

#placeholder for labels of the classes
Y_hat = tf.placeholder(tf.float32,[None,NUMBER_OF_CLASSES])
#loss function
#cross_entropy = -tf.reduce_sum(Y_hat * tf.log(Y))#tf.log(tf.clip_by_value(Y,1e-10,1.0)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels =  Y_hat)
cross_entropy = tf.reduce_mean(cross_entropy)

#% of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_hat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy) #, global_step = global_step)

#Training session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


x_batch, y_batch = create_batchs(X_train, Y_labels, BATCH_SIZE)

print "x batch"
print x_batch.shape	
NUMBER_OF_BATCHS = x_batch.shape[0]
print NUMBER_OF_BATCHS

ix = np.random.permutation(NUMBER_OF_BATCHS)


np.set_printoptions(precision=3)

for i in range(NUMBER_OF_EPOCHS):
#	load batch of images and labels

	for k in ix:
		train_data = {X : x_batch[k], Y_hat : y_batch[k]}
		sess.run(train_step, feed_dict = train_data)
	
#	j = i%NUMBER_OF_BATCHS
#	index = ix[j] 

#	train_data = {X : x_batch[index], Y_hat : y_batch[index]}

#train
#	sess.run(train_step, feed_dict = train_data)
		prediction = sess.run(Y , feed_dict = train_data)

		if k%100 == 0:
		 	a,c = sess.run([accuracy,cross_entropy], feed_dict = train_data)
		 	print "Accuracy : "
		 	print a
		 	print "Cross entropy:"
		 	print c
		 	print ' --------------- Y label -------------------------'
		 	print y_batch[k]
		 	print ' --------------- Y prediction --------------------'
		 	print prediction











