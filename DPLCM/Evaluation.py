import torch
import numpy as np
import math
from HARE import *
   
def WISE_func(r, train_obs, train_delta, train_risk):
    num_sim = train_obs.shape[0]
    WISE = np.zeros(num_sim)
    descending_sort_time = torch.argsort(train_obs, dim = 1, descending = True)
    risk = torch.gather(train_risk, 1, descending_sort_time)
    hazard = 1 / (torch.cumsum(torch.exp(risk), 1))
    hazard = torch.flip(hazard, [1])
    ascending_sort_time = torch.argsort(train_obs, dim = 1, descending = False)
    time = torch.gather(train_obs, 1, ascending_sort_time)
    train_delta = torch.gather(train_delta, 1, ascending_sort_time)
    train_delta[:, 0] = 1  
    NA_est = torch.cumsum(train_delta * hazard, 1)
    for i in range(num_sim):
        tmax = time[i, -1]
        t = torch.arange(0.001, tmax, 0.001)
        if r == 0:
            H0 = torch.log(t)
        elif r == 0.5:
            H0 = torch.log(2 * torch.exp(0.5 * t) - 2)
        elif r == 1:
            H0 = torch.log(torch.exp(t) - 1)
        index = torch.searchsorted(time[i], t)
        H = torch.log(NA_est[i, index])
        SE = (H - H0) ** 2
        WISE[i] = 0.001 * torch.sum(SE) / tmax
    return WISE

def RE_func(g_hat, x):
    g_hat.detach()
    x.detach()
    g0 = 2.2 * (torch.sin(2 * torch.Tensor([math.pi]) * x[:, 0] * x[:, 1]) + torch.cos(torch.Tensor([math.pi]) * x[:, 1] * x[:, 2]/ 2) / 2 + torch.log(x[:, 2] * x[:, 3] + 1) / 3 + (x[:, 3] - x[:, 2] * x[:, 3] * x[:, 4]) / 4 + (torch.exp(x[:, 4]) - 1) / 5 - 0.778)
    RE = torch.sqrt(torch.sum((g_hat - torch.mean(g_hat) - g0) ** 2) / torch.sum(g0 ** 2))
    return RE

def ICI_func(q, risk, event, time): 
    risk = risk.detach().numpy()
    event = event.detach().numpy()
    time = time.detach().numpy()
    quantile = np.percentile(time, q)
    cum_hazard = calculate_cum_hazard(risk, event, time, quantile)
    P_hat = prob_est(risk, cum_hazard)
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