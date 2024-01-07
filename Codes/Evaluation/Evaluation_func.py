
def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x,axis=-1,keepdims=True)
    my = np.mean(y,axis=-1,keepdims=True)
    xm, ym = x-mx, y-my
    r_num = np.sum(xm*ym,axis=-1)
    r_den = np.sqrt(np.sum(np.square(xm),axis=-1)* np.sum(np.square(ym),axis = -1))
    r = r_num / r_den
    return r

# lstm_corr
def Ccoefficient(y_pred,X,pinvX):
    Y = y_pred[:,:,0] # GM
    Y_nonGM = y_pred[:,:,1] # non-GM
    Y = Y - np.mean(Y,axis = -1,keepdims=True)
    Y_nonGM = Y_nonGM - np.mean(Y_nonGM,axis = -1,keepdims=True)
    beta = np.dot(Y,pinvX)
    beta_nonGM = np.dot(Y_nonGM,pinvX)
    Yest = np.dot(beta,X.T)
    Yest_nonGM = np.dot(beta_nonGM,X.T)
    corr_Y = correlation_coefficient(Y,Yest)
    corr_Y_nonGM = correlation_coefficient(Y_nonGM,Yest_nonGM)
    return corr_Y,corr_Y_nonGM, Yest, Yest_nonGM,beta,beta_nonGM

def correlation(act, simu):
  x = act
  y = simu
  mx = np.mean(x,axis=-1,keepdims=True)
  my = np.mean(y,axis=-1,keepdims=True)
  xm, ym = x-mx, y-my
  r_num = np.sum(xm*ym,axis=-1)
  r_den = np.sqrt(np.sum(np.square(xm),axis=-1)* np.sum(np.square(ym),axis = -1))
  r = r_num / r_den
  r = np.mean(r)
  return r

def corr(pre):
  Y = pre # fMRI
  Y = Y - np.mean(Y,axis = -1,keepdims=True)
  beta = np.dot(Y,pinvX)
  Yest = np.dot(beta,X.T)
  return correlation(Y,Yest)