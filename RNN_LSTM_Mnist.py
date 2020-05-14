import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.python.framework import ops 
ops.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/Users/admin/Desktop/database/MNIST_data/', one_hot=True)

learing_rate = 0.001
batch_size =100
n_steps = 28
n_inputs = 28
n_hidden_units = 128
n_classes = 10
X_holder = tf.placeholder(tf.float32)
Y_holder = tf.placeholder(tf.float32)

reshape_X = tf.reshape(X_holder, [-1, n_steps, n_inputs])
lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units)
outputs, state = tf.nn.dynamic_rnn(lstm_cell, reshape_X, dtype=tf.float32)
cell_list = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
last_cell = cell_list[-1]
Weights = tf.Variable(tf.truncated_normal([n_hidden_units, n_classes]))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))
predict_Y = tf.matmul(last_cell, Weights) + biases
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_Y, labels=Y_holder))
optimizer = tf.train.AdamOptimizer(learing_rate)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

isCorrect = tf.equal(tf.argmax(predict_Y, 1), tf.argmax(Y_holder, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))
for i in range(100):
    X, Y = data.train.next_batch(batch_size)
    session.run(train, feed_dict={X_holder:X, Y_holder:Y})
    step = i + 1
    if step % 10 == 0 or step <= 10:
        test_X, test_Y = data.test.next_batch(10000)
        test_accuracy = session.run(accuracy, feed_dict={X_holder:test_X, Y_holder:test_Y})
        print("step:%d test accuracy:%.4f" %(step, test_accuracy))
        
'''
step:1 test accuracy:0.1036
step:2 test accuracy:0.2059
step:3 test accuracy:0.2393
step:4 test accuracy:0.3058
step:5 test accuracy:0.3327
step:6 test accuracy:0.3480
step:7 test accuracy:0.3354
step:8 test accuracy:0.3181
step:9 test accuracy:0.3170
step:10 test accuracy:0.3659
step:20 test accuracy:0.4500
step:30 test accuracy:0.5713
step:40 test accuracy:0.6123
step:50 test accuracy:0.6566
step:60 test accuracy:0.7156
step:70 test accuracy:0.7337
step:80 test accuracy:0.7826
step:90 test accuracy:0.8117
step:100 test accuracy:0.8110
'''
