import sys
import h5py
import os
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model

from Evaluation import Evaluation_func
from Model import ModelAndObjectiveFunction
from Preprocessing import funcs

datadir = "...\\Codes\\Data"
weightsdir= "...\\Codes\\Weights"
os.chdir(datadir)

subjlist = listdir(datadir)

# Information about processed data
fMRIdata,fMRIdata_GM,fMRIdata_nonGM,Cor,tre_nonGM,tre_GM,Cor_nonGM,Cor_GM,Cor_tgtlure,tre_nonGM_sc23,sc23_new,tre_GM_sc1,sc1_new,mask,X = funcs.readMatVars(subjlist[1],varname=('fMRIdata',"fMRIdata_GM","fMRIdata_nonGM",
                                                                                                                                        'Activity_Mask_1D_array','threshold_of_nonGM_1D','threshold_of_GM_1D',
                                                                                                                                        'NonGM_in_Active_area','GM_in_Active_area','Activity_Mask','threshold_of_nonGM_3D',
                                                                                                                                        'src23_thresholded','threshold_of_GM_3D','rc1_thresholded','mask','X_tgtlure'))


# Information about working memory regions
insula,anterior_cingulate_cortex,frontal_inferior_gyrus,middel_temporal_gyrus,middle_frontal,precentral = funcs.readMatVars(
    subjlist[2],varname=("insula", "anterior_cingulate_cortex","frontal_inferior_gyrus","middel_temporal_gyrus",
                          "middle_frontal","precentral"))

# Preprocessing part
fMRIdata_GM_train, fMRIdata_nonGM_train, X, pinvX, n_q, sc23_new_nor_arg, massk = funcs.PreProcessing(fMRIdata,fMRIdata_GM, fMRIdata_nonGM, X, sc23_new, Cor, tre_GM)


# Prediction
os.chdir(weightsdir)
model = load_model('MSDNN.h5',custom_objects={ 'inner_Ccoefficient_loss': ModelAndObjectiveFunction.Ccoefficient_loss(X,pinvX)})
MSDNN = model.predict([fMRIdata_GM_train,fMRIdata_nonGM_train],batch_size=1000)

# Raw GM and non-GM and their copies
fMRIdata_nonGM = fMRIdata_nonGM.reshape(fMRIdata_nonGM.shape[0],390)
fMRIdata_GM = fMRIdata_GM.reshape(fMRIdata_GM.shape[0],390)
fMRIdata_nonGM_Copy = fMRIdata_nonGM.copy()
fMRIdata_GM_Copy = fMRIdata_GM.copy()

# Apply De-noised signal to raw data
fMRIdata_nonGM_Copy[sc23_new_nor_arg[:fMRIdata_GM_train.shape[0]],:] = MSDNN[:,:,1]
fMRIdata_GM_Copy[massk==1,:] = MSDNN[:,:,0]
tre_GM = tre_GM.reshape(tre_GM.shape[1],)

# Extract Gray matter segment of working memory region
insula = insula[mask>0][tre_GM==1]
anterior_cingulate_cortex = anterior_cingulate_cortex[mask>0][tre_GM==1]
frontal_inferior_gyrus = frontal_inferior_gyrus[mask>0][tre_GM==1]
middel_temporal_gyrus = middel_temporal_gyrus[mask>0][tre_GM==1]
middle_frontal = middle_frontal[mask>0][tre_GM==1]
precentral = precentral[mask>0][tre_GM==1]

insula = insula[massk==1]
anterior_cingulate_cortex = anterior_cingulate_cortex[massk==1]
frontal_inferior_gyrus = frontal_inferior_gyrus[massk==1]
middel_temporal_gyrus = middel_temporal_gyrus[massk==1]
middle_frontal = middle_frontal[massk==1]
precentral = precentral[massk==1]

GM_Mask = fMRIdata_GM_Copy[massk==1,:]
GM_Mask_init = fMRIdata_GM[massk==1,:]

insula_DN = GM_Mask[insula == 1,:]
anterior_cingulate_cortex_DN = GM_Mask[anterior_cingulate_cortex == 1,:]
frontal_inferior_gyrus_DN = GM_Mask[frontal_inferior_gyrus == 1,:]
middel_temporal_gyrus_DN = GM_Mask[middel_temporal_gyrus == 1,:]
middle_frontal_DN = GM_Mask[middle_frontal == 1,:]
precentral_DN = GM_Mask[precentral == 1,:]

insula_init = GM_Mask_init[insula == 1,:]
anterior_cingulate_cortex_init = GM_Mask_init[anterior_cingulate_cortex == 1,:]
frontal_inferior_gyrus_init = GM_Mask_init[frontal_inferior_gyrus == 1,:]
middel_temporal_gyrus_init = GM_Mask_init[middel_temporal_gyrus == 1,:]
middle_frontal_init = GM_Mask_init[middle_frontal == 1,:]
precentral_init = GM_Mask_init[precentral == 1,:]

Denoised_segments = [insula_DN,anterior_cingulate_cortex_DN,frontal_inferior_gyrus_DN,middel_temporal_gyrus_DN,middle_frontal_DN,precentral_DN]
init_segments = [insula_init,anterior_cingulate_cortex_init,frontal_inferior_gyrus_init,middel_temporal_gyrus_init,middle_frontal_init,precentral_init]

print('Correlation in each region (from primary correlation to final correlation','\n',23*'***','\n')
for i in range(len(Denoised_segments)):
    print(f'Region correlation {i}: ',
        Evaluation_func.corr(init_segments[i],X,pinvX),'------>',Evaluation_func.corr(Denoised_segments[i],X,pinvX),'\n')

print(23*'***','\n',22*'***','\n')

denoised_data = np.copy(fMRIdata)
GM = fMRIdata_GM.copy()
GM[massk==1,:] = MSDNN[:,:,0]
denoised_data[tre_GM==1,:] = GM

Map = Evaluation_func.Activity_map(zscore(denoised_data.T),X,mask)
Map=Map.reshape(91,109,91);
Map_Nor = (Map - np.min(Map))/(np.max(Map)-np.min(Map))
plt.figure(figsize=(15,10))
plt.imshow(Map_Nor[45,:,:].T)
plt.title("After de-noising")
plt.clim(0.2,0.6)
plt.colorbar()
plt.show()

Map = Evaluation_func.Activity_map(zscore(fMRIdata.T),X,mask)
Map=Map.reshape(91,109,91);
Map_Nor = (Map - np.min(Map))/(np.max(Map)-np.min(Map))
plt.figure(figsize=(15,10))
plt.imshow(Map_Nor[45,:,:].T)
plt.title("Before de-noising")
plt.clim(0.2,0.6)
plt.colorbar()
plt.show()

