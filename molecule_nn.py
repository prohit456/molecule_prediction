import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tf_utils import *
from nnet import *

df = pd.read_csv('train.csv')
validate_df = df.iloc[2080:, 0:]
df = df.iloc[0:2080, 0:]

# In[2]:

print df.head()


# In[3]:

max_dict = {};
min_dict = {};
for column in df.columns:
    if (column == 'id'): 
        continue
    max_dict[column] = max(df[column])
    min_dict[column] = min(df[column])
    df[column] = df[column].apply(lambda x : (x - min_dict[column])/ (max_dict[column] - min_dict[column]*1.0))


# In[4]:

print df.head()


# In[5]:
print df.columns

df_train_x = df.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
df_train_y = pd.DataFrame(df, columns=['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'])
print df_train_y.head()
df_train_x.drop('id', inplace=True, axis=1)
df_train_y.drop('id', inplace=True, axis=1)
print df_train_x.columns
df_train_x_mat = df_train_x.values
df_train_y_mat = df_train_y.values

# In[13]:

from sklearn.model_selection import ShuffleSplit

#for kfold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

from sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=1)



learning_rate = tf.Variable(0.001, tf.float32);

feature_size = 11;
net0_dropout = tf.constant([0.8, 0.8], tf.float32)
net1_dropout = tf.constant([0.8, 0.8], tf.float32)
nnet0 = nnet(11, 2, [64, 32], net0_dropout, learning_rate)
nnet1 = nnet(11, 2, [64, 32], net1_dropout, learning_rate)
writer = tf.summary.FileWriter("debug")
merged_summary = tf.summary.merge_all()
initial = tf.global_variables_initializer()

#creating a session
sess = tf.Session();
sess.run(initial)
hl0_dbg = dbg_variable('hl0_dbg.txt', 1)
num_epochs = 1500;
batch_size = 32;
print sess.run(nnet0.hl_weights)
for train_idx, test_idx in  sss.split(df_train_x_mat,df_train_y_mat):
  x_train, x_test = df_train_x_mat[train_idx], df_train_x_mat[test_idx]
  y_train, y_test = df_train_y_mat[train_idx], df_train_y_mat[test_idx]
  print train_idx
  print x_train,y_train 
  num_batches = x_train.shape[0] // batch_size;
  print num_batches

  for eno in range(num_epochs):
    bcount = 0;
    if (eno < 1250):
      tf.assign(nnet0.learning_rate, 0.1)
      tf.assign(nnet1.learning_rate, 0.1)
    elif (eno < 2500):
      tf.assign(nnet0.learning_rate, 0.1)
      tf.assign(nnet1.learning_rate, 0.1)
    else:
      tf.assign(nnet0.learning_rate, 0.001)
      tf.assign(nnet1.learning_rate, 0.001)
    for bcount in range(num_batches):
      x_batch = x_train[bcount*batch_size:(bcount + 1)*batch_size, 0:];
      y_batch = y_train[bcount*batch_size:(bcount + 1)*batch_size, 0].reshape(batch_size, 1);
      #[summary, dat] = sess.run([merged_summary, nnet0.trainer], feed_dict={nnet0.x:x_batch, nnet0.y:y_batch});
      feed_dict={'x':x_batch, 'y':y_batch};
      nnet0.train(sess, feed_dict)
      y_batch = y_train[bcount*batch_size:(bcount + 1)*batch_size, 1].reshape(batch_size, 1);
      feed_dict={'x':x_batch, 'y':y_batch};
      nnet1.train(sess, feed_dict)


    hl0_dbg.log_var(nnet0.hl_weights[0], sess)


    feed_dict={'x':x_test, 'y':y_test[0:, 0].reshape(y_test.shape[0], 1)}
    nnet0_loss = nnet0.calc_loss(sess, feed_dict)
    print 'eno:', eno, 'loss:', nnet0_loss


    feed_dict={'x':x_test, 'y':y_test[0:, 1].reshape(y_test.shape[0], 1)}
    nnet1_loss = nnet1.calc_loss(sess, feed_dict)
    print 'eno:', eno, 'loss:', nnet1_loss
    #writer.add_summary(summary, eno)


test_df = pd.read_csv('test.csv')
#print test_df.head()

for column in df.columns:
    if (column == 'id'): 
        continue
    try:
      test_df[column] = test_df[column].apply(lambda x : (x - min_dict[column])/ (max_dict[column] - min_dict[column]*1.0))
    except KeyError:
        continue

test_df.drop('id', axis=1, inplace=True)
y0 = np.array(nnet0.predict(sess, ip_dict={'x':test_df.values})).tolist()
y1 = np.array(nnet1.predict(sess, ip_dict={'x':test_df.values})).tolist()
fl_y0 = sum(y0, [])
fl_y1 = sum(y1, [])
test_pred = pd.DataFrame({'formation_energy_ev_natom':fl_y0, 'bandgap_energy_ev':fl_y1})

print test_pred.head()

for column in df.columns:
    if (column == 'id'): 
        continue
    try:
      test_pred[column] = test_pred[column].apply(lambda x : (x *  (max_dict[column] - min_dict[column]*1.0)) + min_dict[column])
    except KeyError:
        continue

print test_pred.head()

def zerof(ip, colname):
    if (ip < 0):
        return min_dict[colname];
    else:
        return ip
test_pred['bandgap_energy_ev'] = test_pred['bandgap_energy_ev'].apply(lambda x: zerof(x, 'bandgap_energy_ev'))
test_pred['formation_energy_ev_natom'] = test_pred['formation_energy_ev_natom'].apply(lambda x: zerof(x,'formation_energy_ev_natom' ))
test_pred.index += 1
test_pred.index.name = 'id'
test_pred.to_csv('test_pred.csv')


# In[ ]:

