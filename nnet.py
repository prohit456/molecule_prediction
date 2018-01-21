import tensorflow as tf
import numpy as np
from tf_utils import *


class nnet:

  def __init__(self, num_features, num_hidden_layers, hl_size_list, hl_keep_prob, learning_rate):
    print 'in init'
    self.num_features = num_features
    self.num_hidden_layers = num_hidden_layers
    self.hl_size_list = hl_size_list
    self.hl_keep_prob = hl_keep_prob;
    self.learning_rate = learning_rate
    self.hl_weights = [];
    self.hl_biases = [];
    self.x = tf.placeholder(tf.float32, shape=[None, num_features], name='inputs');
    self.y = tf.placeholder(tf.float32, shape = [None, 1], name='outputs');
    self.op_wt = tf.Variable(tf.truncated_normal([hl_size_list[-1], 1], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='output_weights')
    self.op_bias = tf.Variable(0.5, name='output_biases')
    self.inputs = [self.x];
    self.init_hidden_layers()
    self.final_output = self.create_nn_map()
    self.loss = self.generate_loss()
    self.trainer = self.create_trainer(learning_rate)

  def init_hidden_layers(self):
    for hl_layer_num in range(self.num_hidden_layers):
      if (hl_layer_num == 0):
        #self.hl_weights.append(tf.nn.dropout(tf.Variable(tf.truncated_normal([self.num_features, self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'), self.hl_keep_prob[0]))
        #self.hl_biases.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_biases'))
        self.hl_weights.append(tf.Variable(tf.truncated_normal([self.num_features, self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'))
        self.hl_biases.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_biases'))
      else:
        #self.hl_weights.append(tf.nn.dropout(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num - 1], self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'), self.hl_keep_prob[hl_layer_num]))
        self.hl_weights.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num - 1], self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'))
        self.hl_biases.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_biases'))

  def create_nn_map(self):
    for hl_layer_num in range(self.num_hidden_layers):
      self.inputs.append(tf.nn.relu(tf.matmul(self.inputs[hl_layer_num], self.hl_weights[hl_layer_num]) + self.hl_biases[hl_layer_num]))
    final_output = tf.matmul(self.inputs[-1], self.op_wt) + self.op_bias;
    return final_output;

  def generate_loss(self):
    loss = tf.reduce_mean(tf.square(self.final_output -self.y))
    return loss;

  def create_trainer(self, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

  def train(self, sess, ip_dict):
    sess.run(self.trainer, feed_dict={self.x:ip_dict['x'], self.y:ip_dict['y']})

  def calc_loss(self, sess, ip_dict):
    return sess.run(self.loss, feed_dict={self.x:ip_dict['x'], self.y:ip_dict['y']})

  def predict(self, sess, ip_dict):
    return sess.run(self.final_output, feed_dict={self.x:ip_dict['x']})
