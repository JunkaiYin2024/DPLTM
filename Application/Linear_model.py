import numpy as np
from Functions_and_Models import *
from HARE import *

def Linear_Estimation(r, train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta):
    p = train_z.shape[1] + train_x.shape[1]
    p0 = train_z.shape[1]
    train_v = np.concatenate([train_z, train_x], axis = 1).T
    n = train_z.shape[0]
    dx = 1e-3
    iter_max = 1000
    beta_est = np.zeros(train_z.shape[1])
    beta_sd = np.zeros(train_z.shape[1])
    num_ticks = 80
    ICI = np.zeros(num_ticks)

    beta_ini = np.zeros(p)
    Rt_ini = np.array([1/j for j in range(n, 0, -1)]).cumsum()
                
    beta, train_Rt, sigma, unconverged = solve_beta(beta_ini, Rt_ini, train_obs, train_delta, train_v, r, dx, iter_max)
    beta_est = beta[: p0]
    beta_sd = np.sqrt(np.diag(sigma)[: p0])

    train_obs.sort()

    risk = np.dot(beta[: p0], test_z.T) + np.dot(beta[p0: ], test_x.T) - np.mean(np.dot(beta[p0: ], test_x.T))
    time = test_obs
    event = test_delta
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += event[j] * (risk[j] <= risk[k]).astype(np.float64) * (time[j] <= time[k]).astype(np.float64)
            denominator += event[j] * (time[j] <= time[k]).astype(np.float64)
    c_index = numerator / denominator                     

    for k in range(num_ticks):
        time_tick = (k + 1) / 12
        test_Rt = get_Rt(train_Rt, train_obs, time_tick)
        P_hat = prob_est(r, test_Rt, risk)
        ICI[k] = HARE(time_tick, event, time, P_hat)
    
    return beta_est, beta_sd, c_index, ICI