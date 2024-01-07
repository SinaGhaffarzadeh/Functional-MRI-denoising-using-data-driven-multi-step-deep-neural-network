
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

def PreProcessing(fMRIdata,fMRIdata_GM, fMRIdata_nonGM, X, sc23, Cor, tre_GM):
  """
  PreProcessing function use for select and extract Gray matter & non-Gray 
  matter voxels using probebility mask of structural MRI data and its segments.   
  --------------------------------------------------------------------------
  Iputs of this function:
    fMRIdata : Row fMRI
    fMRIdata_GM : Extracted Gray matter segment from fMRI data
    fMRIdata_nonGM: Extracted non-Gray matter (White and CSF) from fRMI data
    X : 
    sc23:
    Cor:
    tre_GM: 
  ---------------------------------------------------------------------------
  Output
    fMRIdata_GM_train, fMRIdata_nonGM_train
  """
  
  # Creating a mask to apply on Gray-matter and select voxels to process
  Cor = np.reshape(Cor,(Cor.size,))
  mask_q = np.ones((fMRIdata.shape[0],))
  perct = np.percentile(Cor,90.0)
  mask_q[np.logical_and(Cor<perct,Cor>0)==True] = 2
  massk = mask_q[tre_GM.reshape(tre_GM.shape[1])==1]

  # Design matrix and its inverse
  X = X[:,15:]
  X = zscore(X.T)
  pinvX = np.linalg.pinv(X)
  pinvX = pinvX.T

  # Normalize preprocessed data (Gray-matter and non-Gray matter) using Zscore
  fMRIdata_GM = zscore(fMRIdata_GM,axis=-1)
  fMRIdata_GM = np.reshape(fMRIdata_GM,fMRIdata_GM.shape+(1,))
  fMRIdata_nonGM = zscore(fMRIdata_nonGM,axis=-1)
  n_q = np.sum(massk==1)

  # Apply mask on Gray-matter data to select voxels
  fMRIdata_GM_train = fMRIdata_GM[massk==1,:,:]

  # Selecting non-Gray matter voxels from its mask according to number of selected voxels
  sc23_new = np.reshape(sc23,(sc23.size,))
  sc23_new_nor = (sc23_new - np.min(sc23_new))/(np.max(sc23_new)-np.min(sc23_new))
  sc23_new_nor_arg = np.argsort(sc23_new_nor)
  sc23_new_nor_arg = sc23_new_nor_arg[::-1]

  fMRIdata_nonGM_train = fMRIdata_nonGM[sc23_new_nor_arg[:fMRIdata_GM_train.shape[0]],:]
  fMRIdata_nonGM_train = np.reshape(fMRIdata_nonGM_train,fMRIdata_nonGM_train.shape+(1,))

  return fMRIdata_GM_train, fMRIdata_nonGM_train