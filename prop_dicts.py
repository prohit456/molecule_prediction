import numpy as np
class enhance_prop:

  def __init__(self):
    self.prop_dict = self.get_dicts()

  def get_dicts(self):
    prop_dict = {}
    prop_dict['EA'] = {'In':-0.3125, 'Ga':-0.1081, 'Al':-0.2563, 'O' : -0.22563};
    prop_dict['EN'] = {'In':1.78, 'Ga':1.81, 'Al':1.61, 'O' : 3.44};
    prop_dict['HOMO'] = {'In':-2.784, 'Ga':-2.732, 'Al':-2.697, 'O' : -2.74};
    prop_dict['IP'] = {'In':-5.5374, 'Ga':-5.8182, 'Al':-5.78, 'O' : -5.712};
    prop_dict['LUMO'] = {'In':0.695, 'Ga':0.13, 'Al':0.368, 'O' : 0.398};
    prop_dict['MASS'] = {'In':114.818, 'Ga':69.723, 'Al':26.9815, 'O' : 15.9994};
    prop_dict['rd_max'] = {'In':1.94, 'Ga':2.16, 'Al':3.11, 'O' : 2.403};
    prop_dict['rp_max'] = {'In':1.39, 'Ga':1.33, 'Al':1.5, 'O' : 1.407};
    prop_dict['rs_max'] = {'In':1.09, 'Ga':0.99, 'Al':1.13, 'O' : 1.07};
    return prop_dict


  def weighted_sum_prop(self, prop, xAl, xGa, xIn):
    avg_prop = xAl*self.prop_dict[prop]['Al'] + xGa*self.prop_dict[prop]['Ga'] + xIn*self.prop_dict[prop]['In'];
    return avg_prop

      


