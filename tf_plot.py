from tf_utils import *

dbg_inst = dbg_variable('nnet0_hl_1.txt', 0);
dbg_inst1 = dbg_variable('nnet0_op_wts.txt', 0);
for i in range(200):
 dbg_inst.plot_dim(i)
for i in range(15):
 dbg_inst1.plot_dim(i)
