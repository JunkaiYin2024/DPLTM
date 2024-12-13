from scipy.stats import norm
import torch
import math

def DataGeneration(n, r, c, sim, set):
    beta_0 = torch.Tensor([1, -1])
    z = torch.zeros((n, 2))
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

    return z, x, obs_t, delta