import tensorflow as tf
import numpy as np
from tf_utils import *
import os


class nnet:

  def __init__(self, num_features, num_hidden_layers, hl_size_list, hl_keep_prob, learning_rate, file_id, nnet_name, load_from_file):
    print 'in init'
    print num_features
    self.num_features = num_features
    self.num_hidden_layers = num_hidden_layers
    self.hl_size_list = hl_size_list
    self.hl_keep_prob = hl_keep_prob;
    self.learning_rate = learning_rate
    self.nnet_name = nnet_name;
    self.hl_weights = [];
    self.hl_biases = [];
    self.np_hl_weights = []
    self.np_hl_biases = []
    self.np_op_wt = []
    self.np_op_bias = []
    self.dbg_classes = []
    self.x = tf.placeholder(tf.float32, shape=[None, num_features], name='inputs');
    self.y = tf.placeholder(tf.float32, shape = [None, 1], name='outputs');
    if (load_from_file == 0):
      self.op_wt = tf.Variable(tf.truncated_normal([hl_size_list[-1], 1], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='output_weights')
      self.op_bias = tf.Variable(0.5, name='output_biases')
    else:
      prefix_str = 'params/' + self.nnet_name+'_';
      fname = prefix_str + 'op_wt_'+str(file_id) + '.npy'
      self.op_wt = tf.Variable(np.load(fname), name='output_weights')
      fname = prefix_str + 'op_bias_'+str(file_id) + '.npy'
      self.op_bias = tf.Variable(np.load(fname), name='output_biases')
    self.inputs = [self.x];
    self.init_hidden_layers(file_id, load_from_file)
    self.final_output = self.create_nn_map()
    self.loss = self.generate_loss()
    self.trainer = self.create_trainer(learning_rate)

  def init_hidden_layers(self, file_id, load_from_file):
    for hl_layer_num in range(self.num_hidden_layers):
      (wt, bias) = self.get_weight_and_bias(hl_layer_num, load_from_file, file_id);
      self.hl_weights.append(tf.nn.dropout(wt, self.hl_keep_prob[hl_layer_num]))
      self.hl_biases.append(bias)
      #self.hl_biases.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_biases'))
      #self.hl_weights.append(tf.Variable(tf.truncated_normal([self.num_features, self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'))
      #self.hl_weights.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num - 1], self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_weights'))
      #self.hl_biases.append(tf.Variable(tf.truncated_normal([self.hl_size_list[hl_layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(hl_layer_num) + '_biases'))

  def get_weight_and_bias(self, layer_num, load_from_file, file_id):
    if (load_from_file == 1):
      (wt, bias) = self.load_from_file(layer_num, file_id)
      return (wt, bias)
    if (layer_num == 0):
      wt = tf.Variable(tf.truncated_normal([self.num_features, self.hl_size_list[0]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(layer_num) + '_weights')
      bias = tf.Variable(tf.truncated_normal([self.hl_size_list[0]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(layer_num) + '_biases')
    else:
      print [self.hl_size_list[layer_num - 1], self.hl_size_list[layer_num]]
      wt = tf.Variable(tf.truncated_normal([self.hl_size_list[layer_num - 1], self.hl_size_list[layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(layer_num) + '_weights')
      bias = tf.Variable(tf.truncated_normal([self.hl_size_list[layer_num]], mean = 0, stddev = 1/np.sqrt(self.num_features)), name='layer' + str(layer_num) + '_biases')
    return (wt, bias)

  def load_from_file(self, layer_num, file_id):
    print 'in load from file'
    hl_file_name = 'params/' + self.nnet_name+'_hl_wts_'+str(layer_num)+'_'+str(file_id)+'.npy';
    wt = tf.Variable(np.load(hl_file_name), name='layer' + str(layer_num) + '_weights')
    hl_file_name = 'params/' + self.nnet_name+ '_hl_biases_'+str(layer_num)+'_'+str(file_id)+'.npy';
    bias = tf.Variable(np.load(hl_file_name), name='layer' + str(layer_num) + '_biases')
    return (wt, bias)

  def create_nn_map(self):
    for hl_layer_num in range(self.num_hidden_layers):
        #self.inputs.append(tf.nn.relu(tf.matmul(self.inputs[hl_layer_num], self.hl_weights[hl_layer_num]) + self.hl_biases[hl_layer_num]))
        self.inputs.append(tf.matmul(self.inputs[hl_layer_num], self.hl_weights[hl_layer_num]) + self.hl_biases[hl_layer_num])
    final_output = tf.matmul(self.inputs[-1], self.op_wt) + self.op_bias;
    return final_output;

  def generate_loss(self):
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.log(self.final_output + 1) -tf.log(self.y + 1))))
    #loss = tf.reduce_mean(tf.square(self.final_output - self.y))
    return loss;

  def create_trainer(self, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
    #return tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

  def train(self, sess, ip_dict):
    sess.run(self.trainer, feed_dict={self.x:ip_dict['x'], self.y:ip_dict['y']})

  def calc_loss(self, sess, ip_dict):
    return sess.run(self.loss, feed_dict={self.x:ip_dict['x'], self.y:ip_dict['y']})

  def predict(self, sess, ip_dict):
    print ip_dict['x'].shape
    return sess.run(self.final_output, feed_dict={self.x:ip_dict['x']})

  def load_params(self, id):
    print 'in load params'

  def log_weights(self, sess):
    for item_no, item in enumerate(self.hl_weights):
      self.dbg_classes[item_no].log_var(item, sess);
    self.dbg_classes[item_no+1].log_var(self.op_wt, sess);
      

  def init_dbg_classes(self):
    for l_no in range(len(self.hl_weights)):
      self.dbg_classes.append(dbg_variable(self.nnet_name+'_hl_'+str(l_no)+'.txt', 1))
    self.dbg_classes.append(dbg_variable(self.nnet_name+'_op_wts.txt', 1))

  def save_params(self, sess):
    print 'in save params'
    prefix_str = 'params/'+self.nnet_name+'_';
    for elem_idx, elem in enumerate(self.hl_weights):
      hl_arr = sess.run(elem)
      if (elem_idx == 0):
       fcount = 0;
       while True:
         fname = prefix_str + 'hl_wts_'+str(elem_idx)
         full_fname = fname+ '_' + str(fcount) + '.npy'
         if not os.path.exists(full_fname):
           np.save(full_fname, hl_arr);
           break;
         else:
           fcount += 1;
      else:
           fname = prefix_str + 'hl_wts_'+str(elem_idx)
           full_fname = fname+ '_' + str(fcount) + '.npy'
           np.save(full_fname, hl_arr);
    for elem_idx, elem in enumerate(self.hl_biases):
           hl_arr = sess.run(elem)
           fname = prefix_str + 'hl_biases_'+str(elem_idx)
           full_fname = fname+ '_' + str(fcount) + '.npy'
           np.save(full_fname, hl_arr);


    hl_arr = sess.run(self.op_wt)
    fname = prefix_str + 'op_wt'
    full_fname = fname+ '_' + str(fcount) + '.npy'
    np.save(full_fname, hl_arr);

    hl_arr = sess.run(self.op_bias)
    fname = prefix_str + 'op_bias'
    full_fname = fname+ '_' + str(fcount) + '.npy'
    np.save(full_fname, hl_arr);
