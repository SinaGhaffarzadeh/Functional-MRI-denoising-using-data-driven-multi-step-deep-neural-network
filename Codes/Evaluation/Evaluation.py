import sys
import h5py
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from os import listdir
from os.path import isfile, join

# %cd "/content/drive/MyDrive/"

# from Evaluation import Evaluation 
# from Model import ModelAndObjectiveFunction
# from Preprocessing import funcs

datadir = "/content/drive/MyDrive/project/HCP/data/simo/Gaussian"
%cd "/content/drive/MyDrive/project/HCP/data/simo/Gaussian"


subjlist = listdir(datadir)

# Information of Raw data
mask,X = readMatVars(
  subjlist[7],varname=("mask","X_tgtlure"))

# Information about processed data
fMRIdata,fMRIdata_GM,fMRIdata_nonGM,Cor,tre_nonGM,tre_GM,Cor_nonGM,Cor_GM,Cor_tgtlure,tre_nonGM_sc23,sc23_new,tre_GM_sc1,sc1_new = readMatVars(subjlist[8],varname=("fMRIdata","fMRIdata_q","fMRIdata_dwt_q",
                                                                                                                   "Cor",'tre_WM','tre_GM','Cor_WM','Cor_GM',
                                                                                                                   "Cor_tgtlure1",'tre_WM_sc23','sc23_new',
                                                                                                                   'tre_GM_sc1','sc1_new'))
# Information about working memory regions
insula,anterior_cingulate_cortex,frontal_inferior_gyrus,middel_temporal_gyrus,middle_frontal,precentral = readMatVars(
    subjlist[8],varname=("insula", "anterior_cingulate_cortex","frontal_inferior_gyrus","middel_temporal_gyrus",
                          "middle_frontal","precentral"))

# Preprocessing part
fMRIdata_GM_train, fMRIdata_nonGM_train, X, pinvX, n_q, sc23_new_nor_arg, massk = PreProcessing(fMRIdata,fMRIdata_GM, fMRIdata_nonGM, X, sc23_new, Cor, tre_GM)

plt.figure(figsize=(10,10))
for i in range(6):
  plt.plot(region_Corr[i,:],label=f' Region{i}')
  plt.title('variation of corrolaton on 6 Region')
  plt.ylabel('correlation')
  plt.legend()
plt.grid()
plt.show

# Prediction
MSDNN = model.predict([fMRIdata_GM_train,fMRIdata_nonGM_train],batch_size=1000)

# Raw GM and non-GM and their copies
fMRIdata_nonGM = fMRIdata_nonGM.reshape(fMRIdata_nonGM.shape[0],390)
fMRIdata_GM = fMRIdata_GM.reshape(fMRIdata_GM.shape[0],390)
fMRIdata_nonGM_Copy = fMRIdata_nonGM.copy()
fMRIdata_GM_Copy = fMRIdata_GM.copy()

# Apply De-noised signal to raw data
fMRIdata_nonGM_Copy[sc23_new_nor_arg[:fMRIdata_GM_train.shape[0]],:] = MSDNN[:,:,1]
fMRIdata_GM_Copy[massk==1,:] = MSDNN[:,:,0]

# Extract Gray matter segment of working memory region
insula = insula[tre_GM==1]
anterior_cingulate_cortex = anterior_cingulate_cortex[tre_GM==1]
frontal_inferior_gyrus = frontal_inferior_gyrus[tre_GM==1]
middel_temporal_gyrus = middel_temporal_gyrus[tre_GM==1]
middle_frontal = middle_frontal[tre_GM==1]
precentral = precentral[tre_GM==1]

insula = insula[massk==1]
anterior_cingulate_cortex = anterior_cingulate_cortex[massk==1]
frontal_inferior_gyrus = frontal_inferior_gyrus[massk==1]
middel_temporal_gyrus = middel_temporal_gyrus[massk==1]
middle_frontal = middle_frontal[massk==1]
precentral = precentral[massk==1]

GM_Mask = fMRIdata_GM_Copy[massk==1,:]

insula_DNN = GM_Mask[insula == 1,:]
anterior_cingulate_cortex_DNN = GM_Mask[anterior_cingulate_cortex == 1,:]
frontal_inferior_gyrus_DNN = GM_Mask[frontal_inferior_gyrus == 1,:]
middel_temporal_gyrus_DNN = GM_Mask[middel_temporal_gyrus == 1,:]
middle_frontal_DNN = GM_Mask[middle_frontal == 1,:]
precentral_DNN = GM_Mask[precentral == 1,:]

Denoised_segments = [insula_DNN,anterior_cingulate_cortex_DNN,frontal_inferior_gyrus_DNN,middel_temporal_gyrus_DNN,middle_frontal_DNN,precentral_DNN]

for i in range(len(Denoised_segments)):

  Segment = Denoised_segments[i] - np.mean(Denoised_segments[i],axis = -1,keepdims=True)

  plt.figure(figsize=(15,10))
  plt.imshow(Segment, cmap='gray')
  plt.clim(-0.2,0.2)
  plt.axis('off')
  plt.show()

li = []
for i in range(len(Denoised_segments)):
  li.append(corr(Denoised_segments[i]))
  print(f'Denoised Region correlation {i}: ',corr(Denoised_segments[i]),'\n')
  