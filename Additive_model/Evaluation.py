import numpy as np
from HARE import *

def WISE_func(r, train_Rt, train_obs):
    num_sim = train_obs.shape[0]
    train_Rt = np.concatenate((1e-3 * np.ones((num_sim, 1)), train_Rt), axis = 1)
    WISE = np.zeros(num_sim)
    for i in range(num_sim):
        tmax = np.max(train_obs[i, :])
        T = np.arange(0.001, tmax, 0.001) - 0.0005
        if r == 0:
            H0 = np.log(T)
        elif r == 0.5:
            H0 = np.log(2 * np.exp(0.5 * T) - 2)
        elif r == 1:
            H0 = np.log(np.exp(T) - 1)
        index = train_obs[i].searchsorted(T, side = "right")
        H = np.log(train_Rt[i, index])
        SE = (H - H0) ** 2
        WISE[i] = 0.001 * np.sum(SE) / tmax
    return WISE

def RE_func(spline_coef, splines, g0):
    g_hat = np.dot(spline_coef, splines)
    RE = np.sqrt(np.sum((g_hat - np.mean(g_hat) - g0) ** 2) / np.sum(g0 ** 2))
    return RE

def ICI_func(r, q, risk, train_Rt, train_obs, event, time): 
    Rt = get_Rt(train_Rt, train_obs, time, q)
    P_hat = prob_est(r, Rt, risk)
    ICI = HARE(np.percentile(time, q), event, time, P_hat) 
    return ICI

def c_index_func(risk, time, event):
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += event[j] * (risk[j] <= risk[k]).astype(np.float64) * (time[j] <= time[k]).astype(np.float64)
            denominator += event[j] * (time[j] <= time[k]).astype(np.float64)
    c_index = numerator / denominator
    return c_index