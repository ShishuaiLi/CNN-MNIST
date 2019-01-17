import tensorflow as tf
import sys
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img=tf.placeholder(tf.float32, [50,784])
image = tf.reshape(img, [50, 28, 28, 1])
flts1=tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
convOut1=tf.nn.conv2d(image, flts1, [1,2,2,1], 'SAME')
convOut1= tf.nn.relu(convOut1)
pool1 = tf.nn.max_pool(convOut1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

#print convOut1.shape    #
#print pool1.shape       # (50, 7, 7, 32)

flts2=tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
convOut2=tf.nn.conv2d(pool1, flts2, [1,2,2,1], 'SAME')
convOut2= tf.nn.relu(convOut2)
pool2 = tf.nn.max_pool(convOut2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

#print convOut2.shape
#print pool2.shape       # (50, 2, 2, 64)

# check online for example
pool2 = tf.reshape(pool2, [50, 256])
#w = tf.Variable(tf.truncated_normal([256,10],stddev=0.1))

ans = tf.placeholder(tf.float32, [50, 10])

# learning rate
lr=math.pow(10,-4)

U = tf.Variable(tf.truncated_normal([256,512], stddev=.1))
bU = tf.Variable(tf.truncated_normal([512], stddev=.1))
U1=tf.Variable(tf.truncated_normal([512,1024], stddev=.1))
bU1=tf.Variable(tf.truncated_normal([1024], stddev=.1))
V = tf.Variable(tf.truncated_normal([1024,10], stddev=.1))
bV = tf.Variable(tf.truncated_normal([10], stddev=.1))

L1Output = tf.matmul(pool2,U)+bU
L1Output=tf.nn.relu(L1Output)

L1Output = tf.matmul(L1Output,U1)+bU1
L1Output=tf.nn.relu(L1Output)
#print L1Output.shape
logits=tf.matmul(L1Output,V)+bV
prbs=tf.nn.softmax(logits)
xEnt=tf.losses.softmax_cross_entropy(ans, logits)
#xEnt = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(prbs), reduction_indices=[1]))
train = tf.train.AdamOptimizer(lr).minimize(xEnt)

numCorrect= tf.equal(tf.argmax(prbs,1), tf.argmax(ans,1))
accuracy = tf.reduce_mean(tf.cast(numCorrect, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#------------------------------------------------------
for i in range(2000):
   imgs, anss = mnist.train.next_batch(50)
   sess.run(train, feed_dict={img: imgs, ans: anss})
   #acc,ignore= sess.run([accuracy,train], feed_dict={img: imgs, ans: anss})

sumAcc=0.0
for i in range(2000):
   imgs, anss= mnist.test.next_batch(50)
   sumAcc=sumAcc+sess.run(accuracy, feed_dict={img: imgs, ans: anss})
	
print "Test Accuracy: %r" % (sumAcc/2000)