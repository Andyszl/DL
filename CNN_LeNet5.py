import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.python.framework import ops 
ops.reset_default_graph()
#导入数据
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/Users/admin/Desktop/database/MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder("float", shape = [None, 28,28,1]) #shape in CNNs is always None x height x width x color channels
y_ = tf.placeholder("float", shape = [None, 10]) #shape is always None x number of classes

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))#shape is filter x filter x input channels x output channels
b_conv1 = tf.Variable(tf.constant(.1, shape = [32])) #shape of the bias just has to match output channels of the filter

#卷基层1
h_conv1 = tf.nn.conv2d(input=x, filter=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
#池化层1
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
#卷基层2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(.1, shape = [64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#池化层2
h_pool2 = max_pool_2x2(h_conv2)

#全链接层1
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 512], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(.1, shape = [512]))
#将池化层2输出结果reshape成一维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout Layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#全链接层2
W_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(.1, shape = [10]))

#Final Layer
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.global_variables_initializer())
testInputs = data.test.images.reshape([data.test.images.shape[0],28,28,1])
test_feed={x:testInputs,y_:data.test.labels,keep_prob:1}
batchSize = 50
for i in range(100):
    batch = data.train.next_batch(batchSize)
    trainingInputs = batch[0].reshape([batchSize,28,28,1])
    trainingLabels = batch[1]
    if i % 10 == 0 or i <= 10:
        trainAccuracy = accuracy.eval(session=sess, feed_dict={x:trainingInputs, y_: trainingLabels, keep_prob: 1.0})
        print ("step %d, training accuracy %g%%"%(i, trainAccuracy*100))
    trainStep.run(session=sess, feed_dict={x: trainingInputs, y_: trainingLabels, keep_prob: 0.5})
print("Testing Accuracyis %g%%"%(accuracy.eval(feed_dict=test_feed)*100))    

'''
step 0, training accuracy 6%
step 1, training accuracy 8%
step 2, training accuracy 14%
step 3, training accuracy 12%
step 4, training accuracy 26%
step 5, training accuracy 16%
step 6, training accuracy 18%
step 7, training accuracy 48%
step 8, training accuracy 24%
step 9, training accuracy 18%
step 10, training accuracy 36%
step 20, training accuracy 62%
step 30, training accuracy 78%
step 40, training accuracy 92%
step 50, training accuracy 88%
step 60, training accuracy 82%
step 70, training accuracy 90%
step 80, training accuracy 98%
step 90, training accuracy 98%
Testing Accuracyis 92.98%
'''
