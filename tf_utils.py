import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


class dbg_variable:

  def __init__(self, log_path, wr_flag):
    print 'inside init'
    self.log_path = log_path
    self.sess = tf.Session()
    if (wr_flag == 1):
      self.log_fidx = open(log_path, 'w')
    else:
      self.log_fidx = open(log_path, 'r')

  def log_var(self, ip_tensor, sess):
      flattened_arr = sess.run(tf.reshape(ip_tensor, [-1])).tolist()
      for val in flattened_arr[:-1]:
        self.log_fidx.write(str(val) + ',')
      self.log_fidx.write(str(flattened_arr[-1]) + '\n')

  def log_scalar(self, scal_val):
      self.log_fidx.write(str(scal_val) + '\n')



  def close_file(self):
    self.log_fidx.close();

  def plot_dim(self, ip_dim, save_fig=0, save_str=''):
    df = pd.read_csv(self.log_path);
    arr = df.iloc[0:, ip_dim].values
    plt.plot(range(len(arr)), arr)
    if (save_fig == 0):
      plt.show()
    else:
      plt.savefig(save_str)

  def plot_scalar(self, save_fig=0, save_str=''):
    f = open(self.log_path, 'r');
    arr = map(lambda x: float(x), f.readlines())
    plt.scatter(range(len(arr)), arr)
    if (save_fig == 0):
      plt.show()
    else:
      plt.savefig(save_str)




#hl0_dbg = dbg_variable('lin.txt', 0)
#hl0_dbg.plot_dim(8)
