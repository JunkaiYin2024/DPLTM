from scipy.stats import norm
import math
import torch

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
    
def spline_derivative(i, k, T, t):
    if k == 0:
        return 0 
    else:
        coef_1 = 0 if (T[i + k] == T[i]) else k / (T[i + k] - T[i])
        coef_2 = 0 if (T[i + k + 1] == T[i + 1]) else k / (T[i + k + 1] - T[i + 1])        
        result =  coef_1 *  B_spline(i, k - 1, T, t) - coef_2 * B_spline(i + 1, k - 1, T, t)
        return result

def DataGeneration(n, r, c, sim, num_splines, set, case):
    beta_0 = torch.Tensor([1, -1])
    z = torch.zeros((n, 2))
    splines = torch.zeros((n, num_splines))
    derivatives = torch.zeros((n, num_splines))
    if torch.cuda.is_available():
        if set == 'train':
            torch.cuda.manual_seed(873 * (sim + 1))
        elif set == 'test':
            torch.cuda.manual_seed(3407 * (sim + 1))
    else:        
        if set == 'train':
            torch.manual_seed(873 * (sim + 1))
        elif set == 'test':
            torch.manual_seed(3407 * (sim + 1))
    
    ctime = c * torch.rand(n)
    z[:, 0] = torch.normal(0.5 * torch.ones(n), 0.5 * torch.ones(n))
    random = 0.5 * torch.ones(n)
    z[:, 1] = torch.bernoulli(random)

    corr = torch.Tensor([[1, 0.5, 0.5, 0.5, 0.5], [0.5, 1, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 1, 0.5], [0.5, 0.5, 0.5, 0.5, 1]])
    A = torch.linalg.cholesky(corr)
    x = torch.matmul(torch.normal(0, 1, (n, 5)), A.T)
    x = torch.Tensor(norm.cdf(x) * 2)
    if case == 'Linear':
        g = 0.25 * (x[:, 0] + 2 * x[:, 1] + 3 * x[:, 2] + 4 * x[:, 3] + 5 * x[:, 4] - 15)
    elif case == 'Additive':
        g = 1.85 * (torch.sin(2 * torch.Tensor([math.pi]) * x[:, 0]) + torch.cos(torch.Tensor([math.pi]) * x[:, 1] / 2) / 2 + torch.log(x[:, 2] ** 2 + 1) / 3 + (x[:, 3] - x[:, 3] ** 3) / 4 + (torch.exp(x[:, 4]) - 1) / 5 - 0.428)
    elif case == 'Deep':
        g = 2.2 * (torch.sin(2 * torch.Tensor([math.pi]) * x[:, 0] * x[:, 1]) + torch.cos(torch.Tensor([math.pi]) * x[:, 1] * x[:, 2]/ 2) / 2 + torch.log(x[:, 2] * x[:, 3] + 1) / 3 + (x[:, 3] - x[:, 2] * x[:, 3] * x[:, 4]) / 4 + (torch.exp(x[:, 4]) - 1) / 5 - 0.778)

    temp = torch.rand(n)
    if r == 0: 
        dtime = -torch.log(1 - temp) * torch.exp(- torch.matmul(z, beta_0) - g)
    elif r == 0.5:
        dtime = 2 * torch.log(torch.exp(- torch.matmul(z, beta_0) - g) * (torch.sqrt(1 / (1 - temp)) - 1) + 1)
    elif r == 1:
        dtime = torch.log(temp * torch.exp(- torch.matmul(z, beta_0) - g) / (1 - temp) + 1)
        
    obs_t = torch.minimum(ctime, dtime)
    delta = (dtime <= ctime).to(torch.float32)

    T = torch.zeros(num_splines + 4)
    T[num_splines + 1: ] = torch.ones(3) * torch.max(obs_t)
    T[3: num_splines + 1] = torch.linspace(0, torch.max(obs_t), num_splines - 2)
    for m in range(n):
        for k in range(num_splines):
            splines[m, k] = B_spline(k, 3, T, obs_t[m])
            derivatives[m, k] = spline_derivative(k, 3, T, obs_t[m])

    return z, x, splines, derivatives, obs_t, delta