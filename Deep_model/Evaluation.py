import torch
import math
import numpy as np
from HARE import *

def B_spline(i, k, T, t):
    if k == 0:
        if t < T[i] or t > T[i + 1]:
            return 0
        else:
            return 1
    else:
        coef_1 = 0 if (T[i + k] == T[i]) else (t - T[i]) / (T[i + k] - T[i])
        coef_2 = 0 if (T[i + k + 1] == T[i + 1]) else (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1])
        result = coef_1 * B_spline(i, k - 1, T, t) + coef_2 * B_spline(i + 1, k - 1, T, t)
        return result
    
def WISE_func(r, theta_est, g_mean, train_obs):
    num_splines = theta_est.shape[1]
    num_sim = train_obs.shape[0]
    WISE = torch.zeros(num_sim)
    for i in range(num_sim):
        tmax = torch.max(train_obs[i, :])
        T = torch.zeros(num_splines + 4)
        T[num_splines + 1: ] = torch.ones(3) * tmax
        T[3: num_splines + 1] = torch.linspace(0, tmax, num_splines - 2)
        t = torch.arange(0.001, tmax, 0.001) - 0.0005
        if r == 0:
            H0 = torch.log(t)
        elif r == 0.5:
            H0 = torch.log(2 * torch.exp(0.5 * t) - 2)
        elif r == 1:
            H0 = torch.log(torch.exp(t) - 1)
        H = torch.zeros_like(H0)
        theta = theta_est[i]
        for j in range(H.shape[0]):
            for k in range(num_splines):
                H[j] += theta[k] * B_spline(k, 3, T, t[j])
        H += g_mean[i]
        SE = (H - H0) ** 2
        WISE[i] = 0.001 * torch.sum(SE) / tmax
    return WISE
    
def RE_func(g_hat, x, case):
    g_hat.detach()
    x.detach()
    if case == 'Linear':
        g0 = 0.25 * (x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + 4 * x[:, 3] + 5 * x[:, 4] - 15)
    elif case == 'Additive':
        g0 = 1.85 * (torch.sin(2 * torch.Tensor([math.pi]) * x[:, 0]) + torch.cos(torch.Tensor([math.pi]) * x[:, 1] / 2) / 2 + torch.log(x[:, 2] ** 2 + 1) / 3 + (x[:, 3] - x[:, 3] ** 3) / 4 + (torch.exp(x[:, 4]) - 1) / 5 - 0.428)
    elif case == 'Deep':
        g0 = 2.2 * (torch.sin(2 * torch.Tensor([math.pi]) * x[:, 0] * x[:, 1]) + torch.cos(torch.Tensor([math.pi]) * x[:, 1] * x[:, 2]/ 2) / 2 + torch.log(x[:, 2] * x[:, 3] + 1) / 3 + (x[:, 3] - x[:, 2] * x[:, 3] * x[:, 4]) / 4 + (torch.exp(x[:, 4]) - 1) / 5 - 0.778)
    RE = torch.sqrt(torch.sum((g_hat - torch.mean(g_hat) - g0) ** 2) / torch.sum(g0 ** 2))
    return RE

def ICI_func(r, q, risk, theta, event, time): 
    risk = risk.detach().numpy()
    theta = theta.detach().numpy()  
    event = event.detach().numpy()
    time = time.detach().numpy()
    quantile = np.percentile(time, q)

    num_splines = theta.shape[0]
    H = 0
    tmax = np.max(time)
    T = np.zeros(num_splines + 4)
    T[num_splines + 1: ] = np.ones(3) * tmax
    T[3: num_splines + 1] = np.linspace(0, tmax, num_splines - 2)
    for j in range(num_splines):
        H += theta[j] * B_spline(j, 3, T, quantile)
    P_hat = prob_est(r, risk, H)
    ICI = HARE(quantile, event, time, P_hat)
    return ICI

def c_index_func(risk, time, event):
    risk = risk.detach()
    time = time.detach()
    event = event.detach()
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += event[j] * (risk[j] >= risk[k]).to(torch.float32) * (time[j] <= time[k]).to(torch.float32)
            denominator += event[j] * (time[j] <= time[k]).to(torch.float32)
    c_index = numerator / denominator
    return c_index