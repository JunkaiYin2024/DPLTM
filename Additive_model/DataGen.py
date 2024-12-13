import numpy as np
from scipy.stats import norm

def spline(i, k, T, t):
    if 0 <= i <= k - 1:
        return np.power(t, i + 1)
    else:
        if 0 <= t <= T[i - k + 1]:
            return 0
        else:
            return np.power(t - T[i - k + 1], k)

def DataGeneration(n, r, c, sim, num_splines, set, case):
    beta_0 = np.array([1, -1])
    z = np.zeros((2, n))
    splines = np.zeros((5 * num_splines, n))
    T = np.around(np.linspace(0, 2, num_splines + 1), 2)
    if set == 'train':
        rng = np.random.RandomState(873 * (sim + 1))
    elif set == 'test':
        rng = np.random.RandomState(3407 * (sim + 1))
    ctime = c * rng.rand(n)
    z[0, :] = rng.normal(0.5, 0.5, n)
    z[1, :] = rng.binomial(1, 0.5, n)

    corr = np.array([[1, 0.5, 0.5, 0.5, 0.5], [0.5, 1, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 1, 0.5], [0.5, 0.5, 0.5, 0.5, 1]])
    A = np.linalg.cholesky(corr)
    x = np.dot(A, rng.normal(0, 1, n * 5).reshape((n, 5)).T)
    x = norm.cdf(x) * 2
    if case == 'Linear':
        g = 0.25 * (x[0, :] + 2 * x[1, :] + 3 * x[2, :] + 4 * x[3, :] + 5 * x[4, :] - 15)
    elif case == 'Additive':
        g = 1.85 * (np.sin(2 * np.pi * x[0, :]) + np.cos(np.pi * x[1, :] / 2) / 2 + np.log(x[2, :] ** 2 + 1) / 3 + (x[3, :] - x[3, :] ** 3) / 4 + (np.exp(x[4, :]) - 1) / 5 - 0.428)
    elif case == 'Deep':
        g = 2.2 * (np.sin(2 * np.pi * x[0, :] * x[1, :]) + np.cos(np.pi * x[1, :] * x[2, :]/ 2) / 2 + np.log(x[2, :] * x[3, :] + 1) / 3 + (x[3, :] - x[2, :] * x[3, :] * x[4, :]) / 4 + (np.exp(x[4, :]) - 1) / 5 - 0.778)
    
    for j in range(5):
        for k in range(num_splines):
            for l in range(n):
                splines[j * num_splines + k, l] = spline(k, 3, T, x[j, l])

    temp = rng.rand(n)
    if r == 0: 
        dtime = -np.log(1 - temp) * np.exp(np.dot(beta_0, z) + g)
    elif r == 0.5:
        dtime = 2 * np.log(np.exp(np.dot(beta_0, z) + g) * (np.sqrt(1 / (1 - temp)) - 1) + 1)
    elif r == 1:
        dtime = np.log(temp * np.exp(np.dot(beta_0, z) + g) / (1 - temp) + 1)
        
    obs_t = np.minimum(ctime, dtime)
    delta = (dtime <= ctime).astype(np.float64)

    return z, g, splines, obs_t, delta