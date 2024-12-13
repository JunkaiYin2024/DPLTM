import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

def calculate_cum_hazard(risk, event, time, quantile):
    ascending_sort_time = np.argsort(time)
    descending_sort_time = np.flip(ascending_sort_time)
    risk = risk[descending_sort_time]
    hazard = 1 / (np.cumsum(np.exp(risk)))
    hazard = np.flip(hazard)
    
    time = time[ascending_sort_time]
    event = event[ascending_sort_time]
    NA_est = np.cumsum(event * hazard)
    index = time.searchsorted(quantile)
    cum_hazard = NA_est[index]
    return cum_hazard

def prob_est(risk, cum_hazard):
    P_hat = 1 - np.exp(- cum_hazard * np.exp(risk))
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
