from linear_regressor_tf import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf_utils import *
from nnet import *
from sklearn.model_selection import ShuffleSplit
from prop_dicts import *
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

def enhance_df(ip_df):
    enhanced_prop_inst = enhance_prop();
    new_props = ['EA', 'EN', 'HOMO', 'IP', 'LUMO', 'MASS', 'rd_max', 'rp_max', 'rs_max'];
    xAl = ip_df['percent_atom_al'].values
    xGa = ip_df['percent_atom_ga'].values
    xIn = ip_df['percent_atom_in'].values
    for prop in new_props:
      ip_df[prop] = enhanced_prop_inst.weighted_sum_prop(prop, xAl, xGa, xIn)
    return ip_df


def populate_dicts(df):
    min_dict = {}
    max_dict = {}
    for column in df.columns:
        if (column == 'id'): 
            continue
        max_dict[column] = max(df[column])
        min_dict[column] = min(df[column])
    min_dict['predicted_formation_energy_ev_natom'] = min_dict['formation_energy_ev_natom']
    min_dict['predicted_bandgap_energy_ev'] = min_dict['bandgap_energy_ev']
    max_dict['predicted_bandgap_energy_ev'] = max_dict['bandgap_energy_ev']
    max_dict['predicted_formation_energy_ev_natom'] = max_dict['formation_energy_ev_natom']
    return (min_dict, max_dict);

def normalize_df(ip_df, min_dict, max_dict):
    for column in ip_df.columns:
        if (column == 'id'): 
            continue
        ip_df[column] = ip_df[column].apply(lambda x : (x - min_dict[column])/ (max_dict[column] - min_dict[column]*1.0))
    return ip_df;

def create_train_matrix(ip_df, quadratic_fit=0):
      try:
        df_train_x = ip_df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
      except ValueError:
        df_train_x = ip_df
      df_train_x.drop('id', inplace=True, axis=1)
      if (quadratic_fit == 0):
        df_train_x_mat = df_train_x.values
      else:
        df_train_x_mat = np.concatenate([df_train_x.values, np.square(df_train_x.values)], axis=1)
      return df_train_x_mat

def create_test_matrix(ip_df, quadratic_fit=0):
      df_train_y = pd.DataFrame(ip_df, columns=['formation_energy_ev_natom', 'bandgap_energy_ev'])
      df_train_y_mat = df_train_y.values
      return df_train_y_mat


def zerof(min_dict, ip, colname):
      if (ip < 0):
          return min_dict[colname];
      else:
          return ip


def rescale_data(min_dict, max_dict, ip_df):
    for column in ip_df.columns:
        if (column == 'id'): 
            continue
        #elif ('spacegroup' in column):
        #  continue
        try:
          ip_df[column] = ip_df[column].apply(lambda x : (x *  (max_dict[column] - min_dict[column]*1.0)) + min_dict[column])
        except KeyError:
            print column
            continue
    return ip_df


from sklearn.kernel_ridge import KernelRidge
def get_regressors_ensemble(x, y):
  reg_list = []
  lr = LinearRegression();
  lr.fit(x, y)
  reg_list.append(lr);

  svr0 = SVR()
  svr0.fit(x, y)
  reg_list.append(svr0)

  clf = KernelRidge(kernel ='polynomial', alpha=1.0)
  clf.fit(x, y)
  reg_list.append(clf)

  clf = RandomForestRegressor(n_estimators=20, min_samples_split=20, max_features=0.4, min_samples_leaf=10)
  clf.fit(x, y)
  reg_list.append(clf)

  clf = RandomForestRegressor(n_estimators=30, min_samples_split=20, max_features=0.6, min_samples_leaf=10)
  clf.fit(x, y)
  reg_list.append(clf)

  clf = RandomForestRegressor(n_estimators=40, min_samples_split=20, max_features=0.8, min_samples_leaf=10)
  clf.fit(x, y)
  reg_list.append(clf)

  clf = RandomForestRegressor(n_estimators=60, min_samples_split=20, max_features=0.4, min_samples_leaf=10)
  clf.fit(x, y)
  reg_list.append(clf)

  return reg_list


train_df = pd.read_csv('upd_train.csv');
train_df = enhance_df(train_df);
(min_dict, max_dict) = populate_dicts(train_df);
train_df = normalize_df(train_df, min_dict, max_dict)
X = create_train_matrix(train_df)
y = create_test_matrix(train_df)


reg_list = []
reg_list.append(get_regressors_ensemble(X, y[:,0]))
reg_list.append(get_regressors_ensemble(X, y[:,1]))

sess = tf.Session()
num_regressors = len(reg_list[0])
ip = tf.placeholder(tf.float32, shape=[None, num_regressors])
op = []
weights = [];
final_op = []
losses = []
trainers = []
for num_op in range(2):
  op.append(tf.placeholder(tf.float32, shape=[None, 1]));
  weights.append(tf.Variable(tf.truncated_normal([num_regressors, 1], mean=0, stddev=1/np.sqrt(num_regressors))));
  final_op.append(tf.matmul(ip, weights[num_op]))
  losses.append(tf.sqrt(tf.reduce_mean(tf.square(tf.log(final_op[num_op] + 1) -tf.log(op[num_op] + 1)))))
  trainers.append(tf.train.GradientDescentOptimizer(0.01).minimize(losses[num_op]))


sess.run(tf.global_variables_initializer())
for num_op in range(2):
  print 'start'
  for num_epoch in range(300):
    sess.run(trainers[num_op], feed_dict={ip:np.array([item.predict(X) for item in reg_list[num_op]]).T, op[num_op]:y[:, num_op].reshape(y.shape[0], 1)})
    print sess.run(losses[num_op], feed_dict={ip:np.array([item.predict(X) for item in reg_list[num_op]]).T, op[num_op]:y[:, num_op].reshape(y.shape[0], 1)})


test_df = pd.read_csv('test.csv')
test_df = enhance_df(test_df);
test_df = normalize_df(test_df, min_dict, max_dict);
test_values = create_train_matrix(test_df)
test_op = []
fl_y = []
for num_op in range(2):
  test_op.append(sess.run(final_op[num_op],feed_dict={ip: np.array([item.predict(test_values) for item in reg_list[num_op]]).T}))
  fl_y.append([t[0] for t in test_op[num_op]])
test_pred = pd.DataFrame({'formation_energy_ev_natom':fl_y[0], 'bandgap_energy_ev':fl_y[1]})
test_pred = rescale_data(min_dict, max_dict, test_pred)
test_pred['bandgap_energy_ev'] = test_pred['bandgap_energy_ev'].apply(lambda x: zerof(min_dict, x, 'bandgap_energy_ev'))
test_pred['formation_energy_ev_natom'] = test_pred['formation_energy_ev_natom'].apply(lambda x: zerof(min_dict, x,'formation_energy_ev_natom' ))
print test_pred.head()
test_pred.index += 1
test_pred.index.name = 'id'
test_pred.to_csv('ada_op.csv')

train_df['predicted_formation_energy_ev_natom'] = rf0.predict(X);
train_df['predicted_bandgap_energy_ev'] = rf1.predict(X);
train_df = rescale_data(min_dict, max_dict, train_df)
train_df.to_csv('train_pred.csv')


