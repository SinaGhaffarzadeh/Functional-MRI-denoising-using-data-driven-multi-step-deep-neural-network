

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import*
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
import tensorflow.keras.backend as T
from tensorflow.keras.losses import mse
from tensorflow.keras.constraints import non_neg



# lstmCCA_model

def MSDNN_model(shape_fmri):
    '''
    MSDNN function
    In this function we able to defined each path of filters (Gray-matter and non-gray matter)
    Each path in this study is composed by two separate DNN for increase BOLD signal in gray-matter
    and decrease affects of BOLD signal in non-gray matter part.
    '''

    # Definition of required layers

    input_GM = Input(shape = shape_fmri)
    input_nonGM = Input(shape = shape_fmri)

    # Definition of BOLD signal reducing layers in non-gray matter
    shared_conv1D_nonGM = Conv1D(2,15,padding='same')
    shared_lstm_nonGM = LSTM(2,return_sequences = True)
    shared_tdense_nonGM = TimeDistributed(Dense(1))

    # Definition of BOLD signal enhancing layers in non-gray matter
    shared_conv1D = Conv1D(4,85,padding='same')
    shared_lstm = LSTM(4,return_sequences = True,)
    shared_tdense = TimeDistributed(Dense(1))
    shared_conv1D2 = Conv1D(8,15,padding='same')
    shared_conv1D3 = Conv1D(4,5,padding='same')
    shared_lstm2 = LSTM(4,return_sequences = True)
    shared_tdense2 = TimeDistributed(Dense(1))

    # Bold signal strength booster network structure
    conv1D_GM = shared_conv1D(input_GM)
    lstm_GM = shared_lstm(conv1D_GM)
    tdense_GM = shared_tdense(lstm_GM)

    conv1D_GM2 = shared_conv1D2(tdense_GM)
    conv1D_GM3 = shared_conv1D3(conv1D_GM2)

    lstm_GM2 = shared_lstm2(conv1D_GM3)
    tdense_GM2 = shared_tdense2(lstm_GM2)

    # Bold signal strength reducing network structure in non-Gray matter
    conv1D_nonGM = shared_conv1D_nonGM(input_nonGM)
    lstm_nonGM = shared_lstm_nonGM(conv1D_nonGM)
    tdense_nonGM = shared_tdense_nonGM(lstm_nonGM)

    merged_data = concatenate([tdense_GM2,tdense_nonGM],axis = -1)

    model = Model(inputs = [input_GM,input_nonGM],
                  outputs = merged_data)

    return model


# Definition of Objective function
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = T.mean(x,axis=-1,keepdims=True)
    my = T.mean(y,axis=-1,keepdims=True)
    xm, ym = x-mx, y-my
    r_num = T.sum(xm*ym,axis=-1)
    r_den = T.sqrt(T.sum(T.square(xm),axis=-1)* T.sum(T.square(ym),axis = -1))
    r = r_num / r_den
    r = T.sum(r)
    return r

def Ccoefficient_loss(X,pinvX):

    '''
    Ccoefficient_loss function
    In this function we able to defined our objective using correlation between GLM output and each network output
    Y & Y_nonGM = output of networks
    Yest & Yest_nonGM = output of GLM that computed by design matrix
    corr_Y & corr_Y_nonGM = Correlation coefficients between networks output and GLM output
    Output = diffrence of GM and non-GM
    '''

    def inner_Ccoefficient_loss(y_true,y_pred):
        Y = y_pred[:,:,0] # GM
        Y_nonGM = y_pred[:,:,1] # non-GM
        Y = Y - T.mean(Y,axis = -1,keepdims=True)
        Y_nonGM = Y_nonGM - T.mean(Y_nonGM,axis = -1,keepdims=True)
        beta = T.dot(Y,T.constant(pinvX))
        beta_nonGM = T.dot(Y_nonGM,T.constant(pinvX))
        Yest = T.dot(beta,T.constant(X.T))
        Yest_nonGM = T.dot(beta_nonGM,T.constant(X.T))
        corr_Y = correlation_coefficient_loss(Y,Yest)
        corr_Y_nonGM = correlation_coefficient_loss(Y_nonGM,Yest_nonGM)
        return corr_Y_nonGM - corr_Y

    return inner_Ccoefficient_loss
