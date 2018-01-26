import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf_utils import *
from nnet import *
from sklearn.model_selection import ShuffleSplit
from prop_dicts import *
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#for kfold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class molecule:

  def __init__(self, nnet0_list, nnet1_list, load_from_file, file_id):
    print 'inside init'
    self.df = pd.read_csv('upd_train.csv')
    self.df = self.enhance_df(self.df)
    self.num_features = 11;
    #print self.df.head()
    self.validate_df = self.df.iloc[2080:, 0:]
    self.df = self.df.iloc[0:2080, 0:]
    print self.df.head()
    self.nnet0_list = nnet0_list;
    self.nnet1_list = nnet1_list;
    self.max_dict = {};
    self.min_dict = {};
    self.populate_dicts();
    print self.max_dict
    self.df = self.normalize_df(self.df);
    #self.df = pd.get_dummies(self.df, columns=['spacegroup'])
    print self.df.columns
    self.sss = ShuffleSplit(n_splits=4, test_size=0.1, random_state=1)
    self.df_train_x_mat = self.create_train_matrix(self.df)
    self.df_train_y_mat = self.create_test_matrix(self.df)
    (self.nnet0, self.nnet1) = self.initialize_nnets(load_from_file, file_id)
    self.sess = tf.Session();


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
        if (column == 'id'): 
            continue
        #elif ('spacegroup' in column):
        #  continue
        #elif ('ang' in column):
        #  ip_df[column] = ip_df[column].apply(lambda x : np.cos(x * 2*(np.pi)/ 360))
        #  continue;
        ip_df[column] = ip_df[column].apply(lambda x : (x - self.min_dict[column])/ (self.max_dict[column] - self.min_dict[column]*1.0))
    return ip_df;

  def create_train_matrix(self, ip_df, quadratic_fit=0):
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

  def create_test_matrix(self, ip_df, quadratic_fit=0):
      df_train_y = pd.DataFrame(ip_df, columns=['formation_energy_ev_natom', 'bandgap_energy_ev'])
      df_train_y_mat = df_train_y.values
      return df_train_y_mat




  def initialize_nnets(self, load_from_file, file_id):
    net0_dropout = tf.constant(len(self.nnet0_list) * [1], tf.float32)
    net1_dropout = tf.constant(len(self.nnet1_list) * [1], tf.float32)
    learning_rate = tf.Variable(0.001, tf.float32);
    nnet0 = nnet(self.num_features, len(self.nnet0_list), self.nnet0_list, net0_dropout, learning_rate, file_id[0], 'nnet0', load_from_file[0])
    nnet1 = nnet(self.num_features, len(self.nnet1_list), self.nnet1_list, net1_dropout, learning_rate, file_id[1], 'nnet1', load_from_file[1])
    return (nnet0, nnet1)


 

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



  def validate(self, op_file):
    self.validate_df = self.normalize_df(self.validate_df)
    #self.validate_df = pd.get_dummies(self.validate_df, columns=['spacegroup'])
    validate_x = self.create_train_matrix(self.validate_df)
    validate_y = self.create_test_matrix(self.validate_df)
    print self.validate_df.columns
    y0 = np.array(self.nnet0.predict(self.sess, ip_dict={'x':validate_x})).tolist()
    y1 = np.array(self.nnet1.predict(self.sess, ip_dict={'x':validate_x})).tolist()

    y_fe =validate_y[:, 0] 
    feed_dict={'x':validate_x, 'y':y_fe.reshape(y_fe.shape[0], 1)}
    y0_loss = self.nnet0.calc_loss(self.sess, feed_dict)
    print 'validation loss formation energy', y0_loss

    y_be = validate_y[:, 1] 
    feed_dict={'x':validate_x, 'y':y_be.reshape(y_be.shape[0], 1)}
    y1_loss = self.nnet1.calc_loss(self.sess, feed_dict)
    print 'validation loss Band energy', y1_loss

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



  def predict(self, test_data_file, op_file):
    test_df = pd.read_csv(test_data_file)
    test_df = self.enhance_df(test_df);
    test_df = self.normalize_df(test_df);
    #test_df = pd.get_dummies(test_df, columns=['spacegroup'])
    test_values = self.create_train_matrix(test_df)
    y0 = np.array(self.nnet0.predict(self.sess, ip_dict={'x':test_values})).tolist()
    y1 = np.array(self.nnet1.predict(self.sess, ip_dict={'x':test_values})).tolist()
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

def get_num_neuron_list(seed):
  nlist = [32, 16, 8, 4]
  olist = []
  for count in range(4):
    olist.append(nlist[seed % 4]);
    seed = seed // 4;
  return olist

#initializing ANN
def build_classifier(input_dim=20, optimizer='adam', dropout_a=0.1, dropout_b=0.1, num_layers=1, seed=0):
    classifier = Sequential();
    olist = get_num_neuron_list(0)
    print "dropout_a:", dropout_a
    print "dropout_b:", dropout_b
    classifier.add(Dense(olist[0],  activation = 'relu', input_dim = 20));
    classifier.add(Dropout(p=dropout_a));
    for lno in range(num_layers):
      classifier.add(Dense(olist[1 + lno], activation = 'relu'));
      classifier.add(Dropout(p=dropout_b));


    classifier.add(Dense(1, activation = 'linear'));
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print classifier.get_layer(index=0).get_config()
    return classifier;

def run_tuning(estimator, param_grid, X, Y):
    #parameters = {'batch_size' : [25, 32],
    #              'nb_epoch' : [100, 500],
    #              'optimizer' : ['adam', 'rmsprop']};
    
    #parameters = {'batch_size' : [25, 32],
    #              'epochs' : [100, 200],
    #              'optimizer' : ['adam'],
    #              'dropout_a' : [0.1, 0.2, 0.3, 0.4],
    #              'dropout_b' : [0.1, 0.2, 0.3, 0.4]};
    
    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = param_grid,
                               scoring = 'neg_mean_squared_log_error',
                               cv = 10);
    grid_search = grid_search.fit(X, Y)
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print best_parameters
    print best_accuracy

def run_kfold(self, classifier):
  
  print self.df_train_x_mat.shape, self.df_train_y_mat.shape
  accuracies = cross_val_score(estimator = classifier, X = self.df_train_x_mat, y = self.df_train_y_mat[:, 0], cv = 10, n_jobs = -1);
  
  print accuracies
  print "mean:", accuracies.mean()
  print "var:", accuracies.std();




molecule_inst = molecule([32, 16], [32, 16, 4], [0, 0], [22, 22]);
#molecule_inst.nnet0.init_dbg_classes();
#molecule_inst.train(molecule_inst.nnet0, 0, 0.1, 600)
##molecule_inst.train(molecule_inst.nnet1, 1, 0.1, 600)
#molecule_inst.nnet0.save_params(molecule_inst.sess)
#molecule_inst.nnet1.save_params(molecule_inst.sess)
#molecule_inst.validate('validate.csv')
#molecule_inst.predict('test.csv', 'pred_2112.csv')
#classifier = build_classifier()
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, nb_epoch = 200, verbose=2, dropout_a=0.1, dropout_b=0.2);
run_tuning(estimator = classifier, param_grid = {'dropout_a' : [0.8, 0.9], 'dropout_b' : [0.8, 0.9], 'num_layers':[0, 1, 2], 'seed':range(64)}, X=molecule_inst.df_train_x_mat, Y=molecule_inst.df_train_y_mat[:,0])
#run_tuning(estimator = classifier, param_grid = {'dropout_a' : [0.8], 'dropout_b' : [0.8]}, X=molecule_inst.df_train_x_mat, Y=molecule_inst.df_train_y_mat[:,0])
#classifier = molecule_inst.build_classifier(molecule_inst)
#classifier.fit(molecule_inst.df_train_x_mat, molecule_inst.df_train_y_mat[:,0])
molecule_inst.run_kfold(classifier);
