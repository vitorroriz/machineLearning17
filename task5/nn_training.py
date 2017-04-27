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
STARTING_LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 150
BATCH_SIZE = 64
data_path = './chars74k-lite/'
path_letters_src = {}
path_letters_src[0] = './chars74k-lite/a/a_0.jpg'
data_info = {}
DATA_TOTAL_SIZE = 0
DATA_TOTAL_SIZE_TEST = 0

TRAIN_OR_NOT = 0

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
  DATA_TOTAL_SIZE_TEST = DATA_TOTAL_SIZE_TEST + data_info[j][1]
  j=j+1

DATA_TOTAL_SIZE_TEST = DATA_TOTAL_SIZE_TEST - DATA_TOTAL_SIZE

filelist = []
filelist_test = []

Y_labels = np.zeros((DATA_TOTAL_SIZE, NUMBER_OF_CLASSES))
Y_labels_test = np.zeros((DATA_TOTAL_SIZE_TEST, NUMBER_OF_CLASSES))


print Y_labels.shape 
index = 0
index_test = 0
for dir_letter in range(NUMBER_OF_CLASSES):
	for image_index in range(int(data_info[dir_letter][1]*TRAIN_RATE)):
		i_path = data_path+data_info[dir_letter][0]+'/'+data_info[dir_letter][0]+'_'+str(image_index)+'.jpg'
		filelist.append(i_path)
		Y_labels[index][dir_letter] = 1.0
#		X[index] =  np.asarray(Image.open(i_path))
		index = index + 1


	for image_index in range(int(data_info[dir_letter][1]*TRAIN_RATE),data_info[dir_letter][1]):
		i_path = data_path+data_info[dir_letter][0]+'/'+data_info[dir_letter][0]+'_'+str(image_index)+'.jpg'
		filelist_test.append(i_path)
		Y_labels_test[index_test][dir_letter] = 1.0	
		index_test = index_test + 1


print "DATA:"
print 'Training set size: ' +str(DATA_TOTAL_SIZE)
print 'Test set size:     ' +str(DATA_TOTAL_SIZE_TEST)
print 'Total data size:  ' +str(DATA_TOTAL_SIZE_TEST + DATA_TOTAL_SIZE)

X_train  = np.array([np.array(Image.open(fname)) for fname in filelist])
X_train  = np.reshape(X_train,(DATA_TOTAL_SIZE, 20, 20, 1))
X_train  = X_train/255.0

X_test  = np.array([np.array(Image.open(fname)) for fname in filelist_test])
X_test  = np.reshape(X_test,(DATA_TOTAL_SIZE_TEST, 20, 20, 1))
X_test  = X_test/255.0




#NEURAL NETWORKS START HERE
X = tf.placeholder(tf.float32, [None, 20, 20, 1])
#placeholder for percentage of neurons that are not dropout in each layer during training (should be feed 1.0 for test)
pkeep = tf.placeholder(tf.float32)

CH_L1 = 4
CH_L2 = 8
CH_L3 = 12

PATCH1 = 5
PATCH2 = 4
PATCH3 = 4

#W1 = tf.Variable(tf.random_normal(shape=[20*20,NUMBER_OF_CLASSES], stddev = 1.0/20*20))
W1 = tf.Variable(tf.truncated_normal([PATCH1,PATCH1,1,CH_L1], stddev = 0.1), name = 'W1')
B1 = tf.Variable(tf.zeros([CH_L1]), name = 'B1') #4 is the number of output channels
stride1 = 1 #output is still 20x20
Ycnv1 = tf.nn.conv2d(X,W1, strides = [1, stride1, stride1, 1], padding = 'SAME')
Y1 = tf.nn.relu(Ycnv1 + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

W2 = tf.Variable(tf.truncated_normal([PATCH2,PATCH2,CH_L1,CH_L2], stddev = 0.1), name = 'W2')
B2 = tf.Variable(tf.zeros([CH_L2]), name = 'B2') #4 is the number of output channels
stride2 = 2 #output is 10x10
Ycnv2 = tf.nn.conv2d(Y1d,W2, strides = [1, stride2, stride2, 1], padding = 'SAME')
Y2 = tf.nn.relu(Ycnv2 + B2)
Y2d = tf.nn.dropout(Y2, pkeep)


W3 = tf.Variable(tf.truncated_normal([PATCH3,PATCH3,CH_L2,CH_L3], stddev = 0.1), name = 'W3')
B3 = tf.Variable(tf.zeros([CH_L3]), name = 'B3') #4 is the number of output channels
stride3 = 2 #output is 5x5
Ycnv3 = tf.nn.conv2d(Y2d,W3, strides = [1, stride3, stride3, 1], padding = 'SAME')
Y3 = tf.nn.relu(Ycnv3 + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

W4 = tf.Variable(tf.truncated_normal([5*5*CH_L3, 200], stddev = 0.1), name = 'W4')
B4 = tf.Variable(tf.zeros([200]), name = 'B4')

W5 = tf.Variable(tf.truncated_normal([200 ,NUMBER_OF_CLASSES],stddev = 0.1), name = 'W5')
B5 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]), name = 'B5') 

h1 = tf.nn.relu(tf.matmul(tf.reshape(Y3d,[-1,5*5*CH_L3]), W4) + B4)
h1d = tf.nn.dropout(h1, pkeep)

Ylogits = tf.matmul(h1d, W5) + B5
Y  = tf.nn.softmax(Ylogits)


# Add ops to save all the variables.
saver = tf.train.Saver()



#placeholder for labels of the classes
Y_hat = tf.placeholder(tf.float32,[None,NUMBER_OF_CLASSES])
#loss function
#cross_entropy = -tf.reduce_sum(Y_hat * tf.log(Y))#tf.log(tf.clip_by_value(Y,1e-10,1.0)))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = Ylogits, labels =  Y_hat)
cross_entropy = tf.reduce_mean(cross_entropy)

#% of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_hat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))


optimizer = tf.train.GradientDescentOptimizer(STARTING_LEARNING_RATE)
train_step = optimizer.minimize(cross_entropy)#, global_step = global_step)

#Training session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


print "W1:"
print str(sess.run(W1))


x_batch, y_batch = create_batchs(X_train, Y_labels, BATCH_SIZE)
x_batch_test, y_batch_test = create_batchs(X_test, Y_labels_test, BATCH_SIZE)

print "x batch"
print x_batch.shape	
NUMBER_OF_BATCHS = x_batch.shape[0]
print NUMBER_OF_BATCHS


print "x batch test"
print x_batch_test.shape	
NUMBER_OF_BATCHS_TEST = x_batch_test.shape[0]
print NUMBER_OF_BATCHS_TEST

ix = np.random.permutation(NUMBER_OF_BATCHS)


np.set_printoptions(precision=3)


test_data = {X : X_test, Y_hat : Y_labels_test, pkeep : 1.0}
my_error = {}
my_error_test = {}
my_acc = {}
my_acc_test = {}

for i in range(NUMBER_OF_EPOCHS):
#	load batch of images and labels

	for k in ix:
		train_data = {X : x_batch[k], Y_hat : y_batch[k], pkeep : 0.9}
		sess.run(train_step, feed_dict = train_data)
	
	print '-----------------------'
	print "EPOCH " + str(i)
 	a,c     = sess.run([accuracy,cross_entropy], feed_dict = {X : x_batch[k], Y_hat : y_batch[k], pkeep : 1.0})
 	a_t,c_t = sess.run([accuracy,cross_entropy], feed_dict = test_data)

 	my_error[i] = c
	my_error_test[i] = c_t
	my_acc[i] = a
	my_acc_test[i] = a_t

	print "TRAINING Accuracy : "
	print a
 	print "TRAINING loss:"
 	print c
 	print ''
 	print "TEST Accuracy : "
 	print a_t
 	print "TEST loss"
 	print c_t

#		 	print ' --------------- Y label -------------------------'
#		 	print y_batch[k]
#		 	print ' --------------- Y prediction --------------------'
#		 	print prediction


print "Saving model..."
saver = tf.train.Saver()
saver.save(sess,"model3.ckpt")
print "Model saved!"
print "W1:"
print str(sess.run(W1))

my_error_list = sorted(my_error.items())
my_error_test_list = sorted(my_error_test.items())
e_x, e_y = zip(*my_error_list)
e_x_t, e_y_t = zip(*my_error_test_list)
plt.plot(e_x, e_y, color = 'blue') 
plt.plot(e_x_t, e_y_t, color = 'red')  
l1_1 = mlines.Line2D([], [], color='blue')
l1_2 = mlines.Line2D([], [], color='red')  
plt.legend([l1_1,l1_2],['Training','Test'], loc = 3)
plt.xlabel('Epoch')
plt.ylabel('Error')
title_fig1 = 'learning rate = ' + str(STARTING_LEARNING_RATE)
plt.title(title_fig1)

plt.figure(2)

my_acc_list = sorted(my_acc.items())
my_acc_test_list = sorted(my_acc_test.items())
a_x, a_y = zip(*my_acc_list)
a_x_t, a_y_t = zip(*my_acc_test_list)
plt.plot(a_x, a_y, color = 'blue') 
plt.plot(a_x_t, a_y_t, color = 'red')  
l2_1 = mlines.Line2D([], [], color='blue')
l2_2 = mlines.Line2D([], [], color='red')  
plt.legend([l2_1,l2_2],['Training','Test'], loc = 3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
title_fig2 = 'learning rate = ' + str(STARTING_LEARNING_RATE)
plt.title(title_fig2)




plt.show()








