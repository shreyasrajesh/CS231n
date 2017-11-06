import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)	# This initialises the weights to small positive values so that there is no zero gradient error
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)		#Initialise the bias of the same order as the weight
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides = [1, 1, 1, 1], padding='SAME')	#'SAME' - Tries to pad equally and prioritizes the right side. 

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 2X2 max pooling with 1 feature and 1 channel at a time

W_conv1 = weight_variable([5, 5, 1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])			#Reshaping the image to make it 2D with one channel for black or white
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)	#First hidden layer
h_pool1 = max_pool_2x2(h_conv1)					#Max pooling to halve the size but doubles the number of channels

W_conv2 = weight_variable([5, 5, 32, 64])			#Repeated for the second hidden layer
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])				#Fully connected layer begins	
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])		#the second hidden layer after pooling is made flat
h_fc1= tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)		# Flat second layer is used to compute first fully connected layer

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)			#Dropout is used

W_fc2 = weight_variable([1024, 10])				#Second fully connected layer
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2			#Final output

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))		#Loss using softmax classifier and cross entropy loss function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)		#Loss minimisation using adam optimisation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))		#List of ones and zeros based on match
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))		#Accuracy by converting vector to fraction

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())					#Session variable initialisation
  for i in range(20000):
    batch = mnist.train.next_batch(50)						#Defining one mini-batch
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})				#Training accuracy is computed and printed every 100 time steps
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))












