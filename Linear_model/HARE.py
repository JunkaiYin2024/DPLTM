import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

def get_Rt(train_Rt, train_obs, time, q):
    quantile = np.percentile(time, q)
    Rt_sum = 0
    for i in range(train_Rt.shape[0]):
        index = train_obs[i].searchsorted(quantile, side = "right")        
        Rt_sum += train_Rt[i, index]
    Rt = Rt_sum / train_Rt.shape[0]
    return Rt

def prob_est(r, Rt, risk):
    if r == 0:
        P_hat = 1 - np.exp(- np.exp(-risk) * Rt)
    elif r > 0:
        P_hat = 1 - np.exp(- np.log(1 + r * Rt * np.exp(-risk)) / r) 
    return P_hat

def HARE(quantile, event, time, P_hat):
    P_hat = np.minimum(P_hat, 1 - 1e-5)
    P_hat = np.maximum(P_hat, 1e-5)
    hazard = np.log(-np.log(1 - P_hat))

    revent = ro.FloatVector(event)
    rtime = ro.FloatVector(time)
    rhazard = ro.r.matrix(hazard, nrow = hazard.shape[0], ncol = 1)
    
    pol = importr("polspline")
    calibrate = pol.hare(data = rtime, delta = revent, cov = rhazard)
    rP_cal = pol.phare(quantile, rhazard, calibrate)

    P_cal = np.array(rP_cal)
    ICI = np.mean(np.abs(P_hat - P_cal))
    return ICI