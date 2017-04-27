import sys
import math
import numpy as np
import tensorflow as tf
import helpers 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image
import os
from random import randint

MIN_PROBABILITY = 1.0

class nn_model(object):


  TRAIN_RATE = 0.8
  NUMBER_OF_CLASSES = 26
  STARTING_LEARNING_RATE = 0.01
  NUMBER_OF_EPOCHS = 5
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

  CH_L1 = 4
  CH_L2 = 8
  CH_L3 = 12

  #NEURAL NETWORKS START HERE
  X = tf.placeholder(tf.float32, [None, 20, 20, 1])
  #placeholder for percentage of neurons that are not dropout in each layer during training (should be feed 1.0 for test)
  pkeep = tf.placeholder(tf.float32)

  #W1 = tf.Variable(tf.random_normal(shape=[20*20,NUMBER_OF_CLASSES], stddev = 1.0/20*20))
  W1 = tf.Variable(tf.truncated_normal([5,5,1,CH_L1], stddev = 0.1), name = 'W1')
  B1 = tf.Variable(tf.zeros([CH_L1]), name = 'B1') #4 is the number of output channels
  stride1 = 1 #output is still 20x20
  Ycnv1 = tf.nn.conv2d(X,W1, strides = [1, stride1, stride1, 1], padding = 'SAME')
  Y1 = tf.nn.relu(Ycnv1 + B1)
  Y1d = tf.nn.dropout(Y1, pkeep)

  W2 = tf.Variable(tf.truncated_normal([4,4,CH_L1,CH_L2], stddev = 0.1), name = 'W2')
  B2 = tf.Variable(tf.zeros([CH_L2]), name = 'B2') #4 is the number of output channels
  stride2 = 2 #output is 10x10
  Ycnv2 = tf.nn.conv2d(Y1d,W2, strides = [1, stride2, stride2, 1], padding = 'SAME')
  Y2 = tf.nn.relu(Ycnv2 + B2)
  Y2d = tf.nn.dropout(Y2, pkeep)


  W3 = tf.Variable(tf.truncated_normal([4,4,CH_L2,CH_L3], stddev = 0.1), name = 'W3')
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
  sess = tf.Session()


  # Restore variables from disk.
  saver.restore(sess, "./model3.ckpt")

  label = np.zeros((1,26))

  def predict_proba(self, x_image):
    X_in = np.array([np.array([np.array(x_image)])])
    X_in = np.reshape(X_in,(1,20,20,1))
    # Y_label = np.zeros((1,26))
    p_vec = np.zeros((1,27))
    prediction = self.sess.run(self.Y, feed_dict = {self.X :X_in, self.Y_hat : self.label, self.pkeep : 1.0})
    #print np.amax(prediction)
    
    if np.amax(prediction,1) < MIN_PROBABILITY:
      p_vec[0][26] = 1.0
      return p_vec
    return np.hstack((prediction,np.zeros((1,1))))


def __main__():
  data_path = './chars74k-lite/'
  label = np.zeros((1,26))
  letter = 'd'
  image_number= str(randint(0,50))
  class_dic = {}
  class_i = 0

  for i in next(os.walk(data_path))[1]: 
    class_dic[i] = class_i
    class_i = class_i + 1
  label[0][class_dic[letter]] = 1.0

  my_nn = nn_model()
  fname = './chars74k-lite/'+letter+'/'+letter+'_'+image_number+'.jpg'
  x_image = Image.open(fname)
  print 'label:'
  print label
  print 'prediction:'
  print my_nn.predict_proba(x_image)

if (__main__):
  pass
  #__main__()

