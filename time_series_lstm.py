import csv
import numpy as np
import matplotlib.pyplot as plt

def data_process(filename):
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)
        data = [float(row[1]) for row in csvreader if len(row) > 0]
        normalized_data = (data - np.mean(data)) / np.std(data)
        return normalized_data
        
data=data_process('international-airline-passengers.csv')
'''
array([-1.40777884, -1.35759023, -1.24048348, -1.26557778, -1.33249593,
       -1.21538918, -1.10664719, -1.10664719, -1.20702441, -1.34922546,
       -1.47469699, -1.35759023, -1.38268454, -1.29067209, -1.16520057,
       -1.21538918, -1.29903686, -1.09828242, -0.9226223 , -0.9226223 ,
       -1.02299951, -1.23211871, -1.3910493 , -1.17356534, -1.1317415 ,
       -1.08991766, -0.85570415, -0.98117567, -0.90589276, -0.85570415,
       -0.68004402, -0.68004402, -0.80551554, -0.98954044, -1.12337673,
       -0.95608137, -0.91425753, -0.83897462, -0.73023263, -0.83060985,
       -0.81388031, -0.52111343, -0.42073621, -0.320359  , -0.59639634,
       -0.74696217, -0.90589276, -0.72186786, -0.70513833, -0.70513833,
       -0.37054761, -0.37891237, -0.42910098, -0.31199423, -0.1363341 ,
       -0.06941596, -0.36218284, -0.57966681, -0.83897462, -0.66331449,
       -0.63822018, -0.77205647, -0.37891237, -0.44583052, -0.38727714,
       -0.1363341 ,  0.18152708,  0.10624417, -0.17815794, -0.42910098,
       -0.64658495, -0.42910098, -0.320359  , -0.39564191, -0.1112398 ,
       -0.09451026, -0.08614549,  0.29026907,  0.7001427 ,  0.55794164,
        0.26517476, -0.05268642, -0.36218284, -0.01922735,  0.03096126,
       -0.02759212,  0.3069986 ,  0.27353953,  0.31536337,  0.78379038,
        1.11001633,  1.04309819,  0.62485979,  0.21498616, -0.07778072,
        0.21498616,  0.29026907,  0.17316232,  0.63322456,  0.56630641,
        0.62485979,  1.18529925,  1.54498427,  1.56171381,  1.03473342,
        0.55794164,  0.20662139,  0.4659292 ,  0.49938827,  0.31536337,
        0.68341317,  0.56630641,  0.69177793,  1.29404123,  1.76246824,
        1.87957499,  1.03473342,  0.65831886,  0.24844523,  0.47429396,
        0.66668363,  0.5161178 ,  1.05146296,  0.96781528,  1.16856971,
        1.60353765,  2.23926002,  2.33127247,  1.52825474,  1.05982773,
        0.68341317,  1.04309819,  1.14347541,  0.92599144,  1.16020494,
        1.5115252 ,  1.60353765,  2.13051803,  2.85825285,  2.72441656,
        1.9046693 ,  1.5115252 ,  0.91762667,  1.26894693])
'''
#拆分数据，80%训练，20%测试
def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data

#创建训练数据，seq_size窗口，每个数据按照窗口进行切割，y标签取x的t+1的数据
def model_data(seq_size,train_data,test_data):
    train_x, train_y = [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])

    test_x, test_y = [], []
    for i in range(len(test_data) - seq_size - 1):
        test_x.append(np.expand_dims(test_data[i:i+seq_size], axis=1).tolist())
        test_y.append(test_data[i+1:i+seq_size+1])
    return train_x,train_y,test_x,test_y

train_data,test_data=split_data(data, percent_train=0.80)
train_x,train_y,test_x,test_y=model_data(5,train_data,test_data)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import data_loader
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops 
ops.reset_default_graph()

#建立模型进行训练
class SeriesPredictor:

    def __init__(self, input_dim, seq_size, hidden_dim):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim,reuse=tf.AUTO_REUSE)
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_examples = tf.shape(self.x)[0]
        #W_out为100*1维的，outputs为n*100*1维，所以需要进行一个变换，W_out增加一个维度，并用tf.tile复制n（num_examples）次
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            #设定，如果test error连续三次迭代没有下降就终止训练
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            #保存模型
            save_path = self.saver.save(sess, './model/')
            print('Model saved to {}'.format(save_path))

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './model/')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output

#画图展示
def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
        
#开始训练
seq_size = 5
predictor = SeriesPredictor(input_dim=1, seq_size=seq_size, hidden_dim=100)
predictor.train(train_x, train_y, test_x, test_y)
'''
step: 0		train err: 0.7702465653419495		test err: 0.972259521484375
step: 100		train err: 0.042574163526296616		test err: 0.23135648667812347
step: 200		train err: 0.04118794947862625		test err: 0.2661572992801666
step: 300		train err: 0.040119435638189316		test err: 0.2610573470592499
step: 400		train err: 0.038668714463710785		test err: 0.2381153255701065
Model saved to ./model/
'''
with tf.Session() as sess:
    predicted_vals = predictor.test(sess, test_x)[:,0]
    print('predicted_vals', np.shape(predicted_vals))
    plot_results(train_data, predicted_vals, test_data, 'predictions.png')

