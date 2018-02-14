import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf_utils import *
from nnet import *
from sklearn.model_selection import ShuffleSplit
from prop_dicts import *
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

class molecule:

  def __init__(self, nnet0_list, nnet1_list, load_from_file, file_id, sess):
    print 'inside init'
    self.df = pd.read_csv('upd_train.csv')
    self.df = self.enhance_df(self.df)
    self.num_features = 11;
    #print self.df.head()
    self.validate_df = self.df.iloc[2085:, 0:]
    self.df = self.df.iloc[0:2385, 0:]
    #print self.df.head()
    self.nnet0_list = nnet0_list;
    self.nnet1_list = nnet1_list;
    self.max_dict = {};
    self.min_dict = {};
    self.populate_dicts();
    #print self.max_dict
    self.df = self.normalize_df(self.df);
    #self.df = pd.get_dummies(self.df, columns=['spacegroup'])
    #print self.df.columns
    self.sss = ShuffleSplit(n_splits=4, test_size=0.1, random_state=1)
    self.df_train_x_mat = self.create_train_matrix(self.df)
    self.df_train_y_mat = self.create_test_matrix(self.df)
    (self.lr0, self.lr1, self.kr0, self.kr1) = self.fit_for_regression(self.df, self.df_train_x_mat)
    self.df_train_x_mat= self.add_other_predictors(self.df_train_x_mat, self.lr0, self.lr1, self.kr0, self.kr1)
    (self.nnet0, self.nnet1) = self.initialize_nnets(load_from_file, file_id)
    self.sess = sess


  def enhance_df(self, ip_df):
    enhanced_prop_inst = enhance_prop();
    new_props = ['EA', 'EN', 'HOMO', 'IP', 'LUMO', 'MASS', 'rd_max', 'rp_max', 'rs_max'];
    xAl = ip_df['percent_atom_al'].values
    xGa = ip_df['percent_atom_ga'].values
    xIn = ip_df['percent_atom_in'].values
    for prop in new_props:
      ip_df[prop] = enhanced_prop_inst.weighted_sum_prop(prop, xAl, xGa, xIn)
    return ip_df

  def populate_dicts(self):
    for column in self.df.columns:
        if (column == 'id'): 
            continue
        #elif ('spacegroup' in column):
        #  continue
        self.max_dict[column] = max(self.df[column])
        self.min_dict[column] = min(self.df[column])
    self.min_dict['predicted_formation_energy_ev_natom'] = self.min_dict['formation_energy_ev_natom']
    self.min_dict['predicted_bandgap_energy_ev'] = self.min_dict['bandgap_energy_ev']
    self.max_dict['predicted_bandgap_energy_ev'] = self.max_dict['bandgap_energy_ev']
    self.max_dict['predicted_formation_energy_ev_natom'] = self.max_dict['formation_energy_ev_natom']

  def normalize_df(self, ip_df):
    for column in ip_df.columns:
        if (column in ['id', 'bandgap_energy_error', 'formation_energy_error']): 
            continue
        #elif ('spacegroup' in column):
        #  continue
        #elif ('ang' in column):
        #  ip_df[column] = ip_df[column].apply(lambda x : np.cos(x * 2*(np.pi)/ 360))
        #  continue;
        ip_df[column] = ip_df[column].apply(lambda x : (x - self.min_dict[column])/ (self.max_dict[column] - self.min_dict[column]*1.0))
    return ip_df;

  def create_train_matrix(self, ip_df, quadratic_fit=0, add_kr_lr=1):
      try:
        df_train_x = ip_df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
      except ValueError:
        df_train_x = ip_df
        
      df_train_x.drop('id', inplace=True, axis=1)
      if (quadratic_fit == 0):
        df_train_x_mat = df_train_x.values
      else:
        df_train_x_mat = np.concatenate([df_train_x.values, np.square(df_train_x.values)], axis=1)


      self.num_features = df_train_x_mat.shape[1]
      return df_train_x_mat


  def fit_for_regression(self, ip_df, df_train_x_mat, add_kr_lr=1):
      if (add_kr_lr == 1):
        kr0 = KernelRidge(kernel ='polynomial', alpha=1.0)
        lr0 = LinearRegression()
        kr1 = KernelRidge(kernel ='polynomial', alpha=1.0)
        lr1 = LinearRegression()
        lr0.fit(df_train_x_mat, ip_df.bandgap_energy_ev.values)
        lr1.fit(df_train_x_mat, ip_df.formation_energy_ev_natom.values)
        kr0.fit(df_train_x_mat, ip_df.bandgap_energy_ev.values)
        kr1.fit(df_train_x_mat, ip_df.formation_energy_ev_natom.values)
        return (lr0, lr1, kr0, kr1);

  def add_other_predictors(self, df_train_x_mat, lr0, lr1, kr0, kr1):
        num_rows = df_train_x_mat.shape[0]
        df_train_x_mat = np.concatenate([df_train_x_mat, lr0.predict(df_train_x_mat).reshape(num_rows, 1), lr1.predict(df_train_x_mat).reshape(num_rows, 1), kr0.predict(df_train_x_mat).reshape(num_rows, 1), kr1.predict(df_train_x_mat).reshape(num_rows, 1)], axis=1);
        self.num_features = df_train_x_mat.shape[1]
        return df_train_x_mat


  def create_test_matrix(self, ip_df, quadratic_fit=0):
      df_train_y = pd.DataFrame(ip_df, columns=['formation_energy_ev_natom', 'bandgap_energy_ev'])
      df_train_y_mat = df_train_y.values
      return df_train_y_mat


  def initialize_nnets(self, load_from_file, file_id):
    net0_dropout = tf.constant(len(self.nnet0_list) * [1.0], tf.float32)
    net1_dropout = tf.constant(len(self.nnet1_list) * [1.0], tf.float32)
    learning_rate = tf.Variable(0.001, tf.float32);
    nnet0 = nnet(self.num_features, len(self.nnet0_list), self.nnet0_list, net0_dropout, learning_rate, file_id[0], 'nnet0', load_from_file[0])
    nnet1 = nnet(self.num_features, len(self.nnet1_list), self.nnet1_list, net1_dropout, learning_rate, file_id[1], 'nnet1', load_from_file[1])
    return (nnet0, nnet1)


  def train(self, nnet_inst, y_index, arg_learning_rate, num_epochs):
    batch_size = 32
    for train_idx, test_idx in  self.sss.split(self.df_train_x_mat,self.df_train_y_mat):
      x_train, x_test = self.df_train_x_mat[train_idx], self.df_train_x_mat[test_idx]
      y_train, y_test = self.df_train_y_mat[train_idx], self.df_train_y_mat[test_idx]
      num_batches = x_train.shape[0] // batch_size;
    
      for eno in range(num_epochs):
        bcount = 0;
        if (eno < 2000):
          tf.assign(nnet_inst.learning_rate, arg_learning_rate)
        else:
          tf.assign(nnet_inst.learning_rate, 0.001)
        for bcount in range(num_batches):
          x_batch = x_train[bcount*batch_size:(bcount + 1)*batch_size, 0:];
          y_batch = y_train[bcount*batch_size:(bcount + 1)*batch_size, y_index].reshape(batch_size, 1);
          feed_dict={'x':x_batch, 'y':y_batch};
          nnet_inst.train(self.sess, feed_dict)

        bcount += 1;
        x_batch = x_train[bcount*batch_size:, 0:];
        y_batch = y_train[bcount*batch_size:, y_index].reshape(y_train[bcount*batch_size:, y_index].shape[0], 1);
        feed_dict={'x':x_batch, 'y':y_batch};
        nnet_inst.train(self.sess, feed_dict)
        #if (nnet_inst.nnet_name == 'nnet0'):
        #  nnet_inst.log_weights(self.sess)
    
        #hl0_dbg.log_var(nnet_inst.hl_weights[0], self.sess)
        feed_dict={'x':x_test, 'y':y_test[0:, y_index].reshape(y_test.shape[0], 1)}
        nnet0_loss = nnet_inst.calc_loss(self.sess, feed_dict)
        print 'eno:', eno, 'loss:', nnet0_loss
        if (np.isnan(nnet0_loss)):
          break;
 

  def rescale_data(self, ip_df):
    for column in ip_df.columns:
        if (column == 'id'): 
            continue
        #elif ('spacegroup' in column):
        #  continue
        try:
          ip_df[column] = ip_df[column].apply(lambda x : (x *  (self.max_dict[column] - self.min_dict[column]*1.0)) + self.min_dict[column])
        except KeyError:
            print column
            continue
    return ip_df



  def validate(self, op_file, nnet0, nnet1, fidx):
    self.validate_df = self.normalize_df(self.validate_df)
    #self.validate_df = pd.get_dummies(self.validate_df, columns=['spacegroup'])
    validate_x = self.create_train_matrix(self.validate_df)
    validate_x = self.add_other_predictors(validate_x, self.lr0, self.lr1, self.kr0, self.kr1)
    validate_y = self.create_test_matrix(self.validate_df)
    print self.validate_df.columns
    y0 = np.array(nnet0.predict(self.sess, ip_dict={'x':validate_x})).tolist()
    y1 = np.array(nnet1.predict(self.sess, ip_dict={'x':validate_x})).tolist()

    y_fe =validate_y[:, 0] 
    feed_dict={'x':validate_x, 'y':y_fe.reshape(y_fe.shape[0], 1)}
    y0_loss = nnet0.calc_loss(self.sess, feed_dict)
    print 'validation loss formation energy', y0_loss

    y_be = validate_y[:, 1] 
    feed_dict={'x':validate_x, 'y':y_be.reshape(y_be.shape[0], 1)}
    y1_loss = nnet1.calc_loss(self.sess, feed_dict)
    print 'validation loss Band energy', y1_loss
    fidx.write('validation loss formation energy'+str(y0_loss)+'\n');
    fidx.write('validation loss Band energy'+str(y1_loss)+'\n');

    fl_y0 = sum(y0, [])
    fl_y1 = sum(y1, [])
    self.validate_df['predicted_formation_energy_ev_natom'] = fl_y0
    self.validate_df['predicted_bandgap_energy_ev'] = fl_y1
    self.validate_df = self.rescale_data(self.validate_df)
    self.validate_df['predicted_bandgap_energy_ev'] = self.validate_df['predicted_bandgap_energy_ev'].apply(lambda x: self.zerof(x, 'bandgap_energy_ev'))
    self.validate_df['predicted_formation_energy_ev_natom'] = self.validate_df['predicted_formation_energy_ev_natom'].apply(lambda x: self.zerof(x,'formation_energy_ev_natom' ))
    self.validate_df['bandgap_energy_error'] = np.square(self.validate_df['predicted_bandgap_energy_ev'].values - self.validate_df['bandgap_energy_ev'].values)
    self.validate_df['formation_energy_error'] = np.square(self.validate_df['predicted_formation_energy_ev_natom'].values - self.validate_df['formation_energy_ev_natom'].values)
    self.validate_df.to_csv(op_file)
    return (y0_loss, y1_loss)



  def predict(self, test_data_file, op_file, nnet0, nnet1):
    test_df = pd.read_csv(test_data_file)
    test_df = self.enhance_df(test_df);
    test_df = self.normalize_df(test_df);
    #test_df = pd.get_dummies(test_df, columns=['spacegroup'])
    test_values = self.create_train_matrix(test_df)
    test_values = self.add_other_predictors(test_values, self.lr0, self.lr1, self.kr0, self.kr1)
    y0 = np.array(nnet0.predict(self.sess, ip_dict={'x':test_values})).tolist()
    y1 = np.array(nnet1.predict(self.sess, ip_dict={'x':test_values})).tolist()
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



molecule_inst_list = []
molecule0_inst_list = []
molecule1_inst_list = []
molecule2_inst_list = []
molecule3_inst_list = []
molecule_population0 = []
molecule_population1 = []
molecule_population2 = []
molecule_population3 = []
y0_loss_list = []
y1_loss_list = []
sess = tf.Session()
for count in range(4):
  molecule_inst_list.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))
for count in range(4):
  molecule0_inst_list.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))
  molecule1_inst_list.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))

for count in range(10):
  molecule_population0.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))
  molecule_population1.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))
  molecule_population2.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))
  molecule_population3.append(molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess))

final_molecule_inst = molecule([32, 16, 8, 4], [32, 16, 8, 4], [0, 0], [22, 22], sess)

fidx = open('validate_loss.txt', 'w');
initial = tf.global_variables_initializer()
sess.run(initial)
for count, molecule_inst in enumerate(molecule_inst_list):
  molecule_inst.nnet0.init_dbg_classes();
  molecule_inst.train(molecule_inst.nnet1, 1, 0.01, 200)
  molecule_inst.train(molecule_inst.nnet0, 0, 0.01, 200)
  #molecule_inst.nnet0.save_params(molecule_inst.sess)
  #molecule_inst.nnet1.save_params(molecule_inst.sess)
  (y0_loss, y1_loss) = molecule_inst.validate('validate.csv', molecule_inst.nnet0, molecule_inst.nnet1, fidx)
  y0_loss_list.append(y0_loss)
  y1_loss_list.append(y1_loss)
  molecule_inst.predict('test.csv', 'pred_'+str(count)+'.csv', molecule_inst.nnet0, molecule_inst.nnet1)
  molecule_inst.predict('upd_train.csv', 'train_pred_'+str(count)+ '.csv', molecule_inst.nnet0, molecule_inst.nnet1)

print 'init loss', y0_loss_list
print 'init loss', y1_loss_list

import copy
import random

def mutate_nnets(ip_nnet, nnet0, nnet1, sess):
  print 'in mutate_nnets'

  #weights part
  for wt_idx, h_wts in enumerate(nnet0.hl_weights):
     np_h_wts0 = sess.run(h_wts)
     np_h_wts1 = sess.run(nnet1.hl_weights[wt_idx])
     np_h_biases0 = sess.run(nnet0.hl_biases[wt_idx])
     np_h_biases1 = sess.run(nnet1.hl_biases[wt_idx])
     mask = 0*np_h_wts0;
     umask = mask + 1;
     mask_bias = 0*np_h_biases0;
     umask_bias = mask_bias + 1;
     for row in range(mask.shape[0]):
       for col in range(mask.shape[1]):
         if (random.random() > 0.8):
           mask[row, col] = 1;
           umask[row, col] = 0
     tf.assign(ip_nnet.hl_weights[wt_idx], tf.constant(np_h_wts0*mask + np_h_wts1[wt_idx]*umask));
     tf.assign(ip_nnet.hl_biases[wt_idx], tf.constant(np_h_biases0*mask_bias + np_h_biases1[wt_idx]*umask_bias));

  np_h_wts0 = sess.run(nnet0.op_wt)
  np_h_wts1 = sess.run(nnet1.op_wt)
  np_h_biases0 = sess.run(nnet0.op_bias)
  np_h_biases1 = sess.run(nnet1.op_bias)
  mask = 0*np_h_wts0;
  umask = mask + 1;
  mask_bias = 0*np_h_biases0;
  umask_bias = mask_bias + 1;
  for row in range(mask.shape[0]):
    for col in range(mask.shape[1]):
      if (random.random() > 0.8):
        mask[row, col] = 1;
        umask[row, col] = 0
  tf.assign(ip_nnet.op_wt, tf.constant(np_h_wts0*mask + np_h_wts1*umask));
  #tf.assign(ip_nnet.op_bias, tf.constant(np_h_biases0*1+ np_h_biases1*0));
  return ip_nnet
  #for hl_no, hl in enumerate(nnet0.hl_weights):
  #  if (random.random() > 0.5):
      
    
    


# select two
def sel_two(y0_loss_list, y1_loss_list, molecule_inst_list):
  y0_indices = np.argsort(y0_loss_list)
  print y0_indices
  nnet0_list = []
  for idx in range(2):
    nnet0_list.append(molecule_inst_list[y0_indices[idx]].nnet0)
  
  
  y1_indices = np.argsort(y1_loss_list)
  print y1_indices
  nnet1_list = []
  for idx in range(2):
    nnet1_list.append(molecule_inst_list[y1_indices[idx]].nnet1)
  return [nnet0_list, nnet1_list]

print len(molecule_inst_list)
[nnet0_list, nnet1_list] = sel_two(y0_loss_list, y1_loss_list, molecule_inst_list)


for count, molecule_inst in enumerate(molecule_population0):
  print count
  ip_nnet0 = mutate_nnets(molecule_inst.nnet0, nnet0_list[0], nnet0_list[1], sess)
  molecule_inst.train(ip_nnet0, 0, 0.01, 100)
  ip_nnet1 = mutate_nnets(molecule_inst.nnet1, nnet1_list[0], nnet1_list[1], sess)
  molecule_inst.train(ip_nnet1, 1, 0.01, 100)
  (y0_loss, y1_loss) = molecule_inst.validate('validate.csv', ip_nnet0, ip_nnet1, fidx)
  y0_loss_list.append(y0_loss)
  y1_loss_list.append(y1_loss)
  molecule0_inst_list.append(molecule_inst)
print 'second loss', y0_loss_list
print 'second loss', y1_loss_list

[nnet0_list, nnet1_list] = sel_two(y0_loss_list, y1_loss_list, molecule0_inst_list)
y0_loss_list = []
y1_loss_list = []

for count, molecule_inst in enumerate(molecule_population1):
  print count
  ip_nnet0 = mutate_nnets(molecule_inst.nnet0, nnet0_list[0], nnet0_list[1], sess)
  molecule_inst.train(ip_nnet0, 0, 0.01, 200)
  ip_nnet1 = mutate_nnets(molecule_inst.nnet1, nnet1_list[0], nnet1_list[1], sess)
  molecule_inst.train(ip_nnet1, 1, 0.01, 200)
  (y0_loss, y1_loss) = molecule_inst.validate('validate.csv', ip_nnet0, ip_nnet1, fidx)
  y0_loss_list.append(y0_loss)
  y1_loss_list.append(y1_loss)
  molecule1_inst_list.append(molecule_inst)
print y0_loss_list
print y1_loss_list

[nnet0_list, nnet1_list] = sel_two(y0_loss_list, y1_loss_list, molecule1_inst_list)
y0_loss_list = []
y1_loss_list = []



print 'final stretch'
ip_nnet0 = mutate_nnets(final_molecule_inst.nnet0, nnet0_list[0], nnet0_list[0], sess)
final_molecule_inst.train(ip_nnet0, 0, 0.01, 2000)
ip_nnet1 = mutate_nnets(final_molecule_inst.nnet1, nnet1_list[0], nnet1_list[0], sess)
print count
final_molecule_inst.train(ip_nnet1, 1, 0.01, 2000)
(y0_loss, y1_loss) = final_molecule_inst.validate('validate.csv', ip_nnet0, ip_nnet1, fidx)
y0_loss_list.append(y0_loss)
y1_loss_list.append(y1_loss)
print y0_loss_list
print y1_loss_list
final_molecule_inst.predict('test.csv', 'pred_'+str(count)+'.csv', ip_nnet0, ip_nnet1)
