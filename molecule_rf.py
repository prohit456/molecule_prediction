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






train_df = pd.read_csv('upd_train.csv');
train_df = enhance_df(train_df);
(min_dict, max_dict) = populate_dicts(train_df);
train_df = normalize_df(train_df, min_dict, max_dict)
rf0 = RandomForestRegressor()
rf1 = RandomForestRegressor()


grid_search = GridSearchCV(estimator = rf0, param_grid = {'n_estimators':[5, 10, 15, 20], 'min_samples_split':[2, 20, 80, 200], 'min_samples_leaf':[1, 2, 4, 8, 16, 32], 'max_features':[0.1, 0.2, 0.4, 0.8, 1]}, scoring = 'neg_mean_squared_log_error', cv=5)
X = create_train_matrix(train_df)
y = create_test_matrix(train_df)
grid_search.fit (X, y[:,0])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
rf0_params = best_parameters
print rf0_params
print best_accuracy
rf0 = grid_search.best_estimator_



grid_search = GridSearchCV(estimator = rf1, param_grid = {'n_estimators':[5, 10, 15, 20, 40, 60], 'min_samples_split':[2, 20, 80, 200], 'min_samples_leaf':[1, 2, 4, 8, 16, 32], 'max_features':[0.1, 0.2, 0.4, 0.8, 1]}, scoring = 'neg_mean_squared_log_error', cv=5)
grid_search.fit (X, y[:,1])
rf1_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
rf1 = grid_search.best_estimator_



print rf1_params
print best_accuracy


#rf0 = RandomForestRegressor(n_estimators=rf0_params['n_estimators'], min_samples_split=rf0_params['min_samples_split'], min_samples_leaf=rf0_params['min_samples_leaf'], max_features=rf0_params['max_features'])
#rf1 = RandomForestRegressor(n_estimators=rf1_params['n_estimators'], min_samples_split=rf1_params['min_samples_split'], min_samples_leaf=rf1_params['min_samples_leaf'], max_features=rf1_params['max_features'])
#rf0.fit (X, y[:,0])
#rf1.fit (X, y[:,1])
test_df = pd.read_csv('test.csv')
test_df = enhance_df(test_df);
test_df = normalize_df(test_df, min_dict, max_dict);
test_values = create_train_matrix(test_df)
fl_y0 = rf0.predict(test_values)
fl_y1 = rf1.predict(test_values)
test_pred = pd.DataFrame({'formation_energy_ev_natom':fl_y0, 'bandgap_energy_ev':fl_y1})
test_pred = rescale_data(min_dict, max_dict, test_pred)
test_pred['bandgap_energy_ev'] = test_pred['bandgap_energy_ev'].apply(lambda x: zerof(min_dict, x, 'bandgap_energy_ev'))
test_pred['formation_energy_ev_natom'] = test_pred['formation_energy_ev_natom'].apply(lambda x: zerof(min_dict, x,'formation_energy_ev_natom' ))
test_pred.index += 1
test_pred.index.name = 'id'
test_pred.to_csv('op.csv')

train_df['predicted_formation_energy_ev_natom'] = rf0.predict(X);
train_df['predicted_bandgap_energy_ev'] = rf1.predict(X);
train_df = rescale_data(min_dict, max_dict, train_df)
train_df.to_csv('train_pred.csv')


