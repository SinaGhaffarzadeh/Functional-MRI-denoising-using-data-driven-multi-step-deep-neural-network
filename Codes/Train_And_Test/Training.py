

import sys
import h5py
import numpy as np
import scipy.io as si
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore
from scipy.io import savemat
from os import listdir
from os.path import isfile, join
from tensorflow.keras.utils import plot_model


datadir = "/content/drive/MyDrive/project/HCP/data/simo/Gaussian"
%cd "/content/drive/MyDrive/project/HCP/data/simo/Gaussian"

loss = []
error_log_GM_training = []
error_log_GM_validation = []
error_log_nonGM_training = []
error_log_nonGM_validation = []
error_log_nonGM_validation_loss = []
error_log_GM_validation_loss = []

mean_SNR_GM = []
mean_SNR_nonGM = []
mean_SNR_Box1 = []

intermediate_output_loss= []


subjlist = listdir(datadir)

# Information of Raw data
mask,X = readMatVars(
  subjlist[7],varname=("mask","X_tgtlure"))

# Information about processed data
fMRIdata,fMRIdata_GM,fMRIdata_nonGM,Cor,tre_nonGM,tre_GM,Cor_nonGM,Cor_GM,Cor_tgtlure,tre_nonGM_sc23,sc23_new,tre_GM_sc1,sc1_new = readMatVars(subjlist[8],varname=("fMRIdata","fMRIdata_q","fMRIdata_dwt_q",
                                                                                                                   "Cor",'tre_WM','tre_GM','Cor_WM','Cor_GM',
                                                                                                                   "Cor_tgtlure1",'tre_WM_sc23','sc23_new',
                                                                                                                   'tre_GM_sc1','sc1_new'))
# Preprocessing part
fMRIdata_GM_train, fMRIdata_nonGM_train, X, pinvX, n_q = PreProcessing(fMRIdata,fMRIdata_GM, fMRIdata_nonGM, X, sc23_new, Cor, tre_GM)

# Calculating primary SNR in Gray-matter and non-Gray matter
SNR_GM = signaltonoise(fMRIdata_GM_train[:,:,0].reshape(fMRIdata_GM_train[:,:,0].shape[0],fMRIdata_GM_train[:,:,0].shape[1]).T)
SNR_nonGM = signaltonoise(fMRIdata_nonGM_train[:,:,0].reshape(fMRIdata_nonGM_train[:,:,0].shape[0],fMRIdata_nonGM_train[:,:,0].shape[1]).T)
mean_SNR_GM.append(np.mean(SNR_GM))
mean_SNR_nonGM.append(np.mean(SNR_nonGM))

# Load MSDNN model
model = MSDNN_model(np.reshape(fMRIdata_GM,fMRIdata_GM.shape+(1,)).shape[1:]) # input shape for this case is (390,1)

# Plot and Show model
model.summary()

# Compiling process
opt = Adam(learning_rate=0.0006)
model.compile(optimizer=opt,loss=Ccoefficient_loss(X,pinvX))
randval = np.random.rand(n_q)
split = 0.90
epochs = 20
region_Corr = np.zeros((6,epochs))

# Training part
for e in range(epochs):
    print("epoch: ",e)

    history = model.fit([fMRIdata_GM_train[randval<=split,:,:],fMRIdata_nonGM_train[randval<=split,:,:]],
                        y=np.ones((np.sum(randval<=split),390,2)),batch_size = 1000,epochs = 1)
    loss.append(history)

    # testing dataset
    Prediction = model.predict([fMRIdata_GM_train,fMRIdata_nonGM_train],batch_size=1000)
    corr_Y,corr_Y_dwt, Yest, Yest_dwt,beta,beta_dwt = Ccoefficient(Prediction,X,pinvX)
    error_log_GM_training.append(np.mean(corr_Y[randval<=split]))
    error_log_nonGM_training.append(np.mean(corr_Y_dwt[randval<=split]))
    error_log_GM_validation.append(np.mean(corr_Y[randval>split]))
    error_log_nonGM_validation.append(np.mean(corr_Y_dwt[randval>split]))

    error_log_GM_validation_loss.append(np.sum(corr_Y[randval>split]))
    error_log_nonGM_validation_loss.append(np.sum(corr_Y_dwt[randval>split]))


    SNR_GM = signaltonoise(Prediction[:,:,0].reshape(Prediction[:,:,0].shape[0],Prediction[:,:,0].shape[1]).T)
    SNR_nonGM = signaltonoise(Prediction[:,:,1].reshape(Prediction[:,:,1].shape[0],Prediction[:,:,1].shape[1]).T)

    mean_SNR_GM.append(np.mean(SNR_GM))
    mean_SNR_nonGM.append(np.mean(SNR_nonGM))

