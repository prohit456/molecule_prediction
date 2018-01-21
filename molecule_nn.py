import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf_utils import *
from nnet import *
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf

class molecule:

  def __init__(self):
    print 'inside init'
    self.df = pd.read_csv('train.csv')
    self.validate_df = self.df.iloc[2080:, 0:]
    self.df = self.df.iloc[0:2080, 0:]
    self.max_dict = {};
    self.min_dict = {};
    self.populate_dicts();
    print self.max_dict
    self.df = self.normalize_df(self.df);
    self.sss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=1)
    (self.df_train_x_mat, self.df_train_y_mat) = self.create_train_matrices(self.df)
    (self.nnet0, self.nnet1) = self.initialize_nnets()
    self.sess = tf.Session();


  def populate_dicts(self):
    for column in self.df.columns:
        if (column == 'id'): 
            continue
        self.max_dict[column] = max(self.df[column])
        self.min_dict[column] = min(self.df[column])
    self.min_dict['predicted_formation_energy_ev_natom'] = self.min_dict['formation_energy_ev_natom']
    self.min_dict['predicted_bandgap_energy_ev'] = self.min_dict['bandgap_energy_ev']
    self.max_dict['predicted_bandgap_energy_ev'] = self.max_dict['bandgap_energy_ev']
    self.max_dict['predicted_formation_energy_ev_natom'] = self.max_dict['formation_energy_ev_natom']

  def normalize_df(self, ip_df):
    for column in ip_df.columns:
        if (column == 'id'): 
            continue
        ip_df[column] = ip_df[column].apply(lambda x : (x - self.min_dict[column])/ (self.max_dict[column] - self.min_dict[column]*1.0))
    return ip_df;

  def create_train_matrices(self, ip_df):
    df_train_x = ip_df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
    df_train_y = pd.DataFrame(ip_df, columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'])
    df_train_x.drop('id', inplace=True, axis=1)
    df_train_y.drop('id', inplace=True, axis=1)
    df_train_x_mat = df_train_x.values
    df_train_y_mat = df_train_y.values
    return (df_train_x_mat, df_train_y_mat)

  def initialize_nnets(self):
    net0_dropout = tf.constant([0.8, 0.8], tf.float32)
    net1_dropout = tf.constant([0.8, 0.8], tf.float32)
    learning_rate = tf.Variable(0.001, tf.float32);
    nnet0 = nnet(11, 2, [64, 32], net0_dropout, learning_rate)
    nnet1 = nnet(11, 2, [64, 32], net1_dropout, learning_rate)
    return (nnet0, nnet1)


  def train(self):
    initial = tf.global_variables_initializer()
    self.sess.run(initial)
    hl0_dbg = dbg_variable('hl0_dbg.txt', 1)
    num_epochs = 300;
    batch_size = 32;
    for train_idx, test_idx in  self.sss.split(self.df_train_x_mat,self.df_train_y_mat):
      x_train, x_test = self.df_train_x_mat[train_idx], self.df_train_x_mat[test_idx]
      y_train, y_test = self.df_train_y_mat[train_idx], self.df_train_y_mat[test_idx]
      num_batches = x_train.shape[0] // batch_size;
    
      for eno in range(num_epochs):
        bcount = 0;
        if (eno < 1250):
          tf.assign(self.nnet0.learning_rate, 0.1)
          tf.assign(self.nnet1.learning_rate, 0.1)
        elif (eno < 2500):
          tf.assign(self.nnet0.learning_rate, 0.1)
          tf.assign(self.nnet1.learning_rate, 0.1)
        else:
          tf.assign(self.nnet0.learning_rate, 0.001)
          tf.assign(self.nnet1.learning_rate, 0.001)
        for bcount in range(num_batches):
          x_batch = x_train[bcount*batch_size:(bcount + 1)*batch_size, 0:];
          y_batch = y_train[bcount*batch_size:(bcount + 1)*batch_size, 0].reshape(batch_size, 1);
          feed_dict={'x':x_batch, 'y':y_batch};
          self.nnet0.train(self.sess, feed_dict)
          y_batch = y_train[bcount*batch_size:(bcount + 1)*batch_size, 1].reshape(batch_size, 1);
          feed_dict={'x':x_batch, 'y':y_batch};
          self.nnet1.train(self.sess, feed_dict)
    
        hl0_dbg.log_var(self.nnet0.hl_weights[0], self.sess)
        feed_dict={'x':x_test, 'y':y_test[0:, 0].reshape(y_test.shape[0], 1)}
        nnet0_loss = self.nnet0.calc_loss(self.sess, feed_dict)
        print 'eno:', eno, 'loss:', nnet0_loss
    
    
        feed_dict={'x':x_test, 'y':y_test[0:, 1].reshape(y_test.shape[0], 1)}
        nnet1_loss = self.nnet1.calc_loss(self.sess, feed_dict)
        print 'eno:', eno, 'loss:', nnet1_loss


  def rescale_data(self, ip_df):
    for column in ip_df.columns:
        if (column == 'id'): 
            continue
        try:
          ip_df[column] = ip_df[column].apply(lambda x : (x *  (self.max_dict[column] - self.min_dict[column]*1.0)) + self.min_dict[column])
        except KeyError:
            print column
            continue
    return ip_df



  def validate(self, op_file):
    validate_df = self.normalize_df(self.validate_df)
    validate_feature_df = validate_df.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'] , axis=1)
    y0 = np.array(self.nnet0.predict(self.sess, ip_dict={'x':validate_feature_df.values})).tolist()
    y1 = np.array(self.nnet1.predict(self.sess, ip_dict={'x':validate_feature_df.values})).tolist()
    fl_y0 = sum(y0, [])
    fl_y1 = sum(y1, [])
    validate_df['predicted_formation_energy_ev_natom'] = fl_y0
    validate_df['predicted_bandgap_energy_ev'] = fl_y1
    validate_df = self.rescale_data(validate_df)
    self.validate_df['predicted_bandgap_energy_ev'] = validate_df['predicted_bandgap_energy_ev'].apply(lambda x: self.zerof(x, 'bandgap_energy_ev'))
    self.validate_df['predicted_formation_energy_ev_natom'] = validate_df['predicted_formation_energy_ev_natom'].apply(lambda x: self.zerof(x,'formation_energy_ev_natom' ))
    self.validate_df['bandgap_energy_error'] = np.square(validate_df['predicted_bandgap_energy_ev'].values - validate_df['bandgap_energy_ev'].values)
    self.validate_df['formation_energy_error'] = np.square(validate_df['predicted_formation_energy_ev_natom'].values - validate_df['formation_energy_ev_natom'].values)
    self.validate_df.to_csv(op_file)



  def predict(self, test_data_file, op_file):
    test_df = pd.read_csv(test_data_file)
    test_df.drop('id', axis=1, inplace=True)
    test_df = self.normalize_df(test_df);
    y0 = np.array(self.nnet0.predict(self.sess, ip_dict={'x':test_df.values})).tolist()
    y1 = np.array(self.nnet1.predict(self.sess, ip_dict={'x':test_df.values})).tolist()
    fl_y0 = sum(y0, [])
    fl_y1 = sum(y1, [])
    test_pred = pd.DataFrame({'formation_energy_ev_natom':fl_y0, 'bandgap_energy_ev':fl_y1})
    test_pred = self.rescale_data(test_pred)
    test_pred['bandgap_energy_ev'] = test_pred['bandgap_energy_ev'].apply(lambda x: self.zerof(x, 'bandgap_energy_ev'))
    test_pred['formation_energy_ev_natom'] = test_pred['formation_energy_ev_natom'].apply(lambda x: self.zerof(x,'formation_energy_ev_natom' ))
    test_pred.index += 1
    test_pred.index.name = 'id'
    test_pred.to_csv(op_file)



  def zerof(self, ip, colname):
      if (ip < 0):
          return self.min_dict[colname];
      else:
          return ip


molecule_inst = molecule();
molecule_inst.train()
molecule_inst.validate('validate.csv')
molecule_inst.predict('test.csv', 'pred_1159.csv')
