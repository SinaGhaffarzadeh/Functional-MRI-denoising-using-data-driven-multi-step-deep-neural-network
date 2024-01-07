
import h5py
import numpy as np


def readMatVars(filepath,varname):
    """
    This function used to read .mat file from directory
    varname: a tuple of variables to load
    return:
        a list of ndarray
    """
    A = h5py.File(filepath,'r')
    var = list()
    for i in range(len(varname)):
        temp = A[varname[i]][:] #[()]
        var.append(temp)
    return var


def signaltonoise(a, axis=0, ddof=0):
    '''
    Signal to Noise Ratio
    a: a signal that we want to check its signal to noise ratio
    '''
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate
