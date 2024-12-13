import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

def prob_est(r, risk, H):
    pred = np.exp(risk + H)
    if r == 0:
        P_hat = 1 - np.exp(- pred)
    elif r > 0:
        P_hat = 1 - np.exp(- np.log(1 + r * pred) / r)
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
