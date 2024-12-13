import numpy as np

def S0_fun(ord_bz, Rt, r, n):
    Rt0 = np.zeros(n)
    Rt0[1: n] = Rt[0: n - 1]
    s0 = np.zeros(n)
    if r == 0:
        a = np.exp(-ord_bz)
        s0 = np.sum(a) - np.cumsum(a) + a
    elif r > 0:
        for i in range(n):
            a = np.exp(-ord_bz)/(1 + r * Rt0[i] * np.exp(-ord_bz))
            s0[i] = np.sum(a[i: n])
    return s0   

def Rt_fun(ord_delta, s0):
    Rt = np.cumsum(ord_delta / s0)
    return Rt

def Ubeta_fun(ord_delta, ord_z, ord_bz, Rt, r):
    if r == 0:
        a = ord_z * (ord_delta - Rt * np.exp(-ord_bz))
    if r > 0:
        a = ord_z * (ord_delta - np.log(1 + r * Rt * np.exp(-ord_bz)) / r)
    b = np.sum(a, axis = 1)
    return b

def Hbeta_fun(ord_z, ord_bz, Rt, r, p):
    common = Rt * np.exp(-ord_bz) / (1 + r * Rt * np.exp(-ord_bz))
    a = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            a[i, j] = -np.sum((ord_z[i, :]) * (ord_z[j, :]) * common)
    return a

def solve_beta(beta_ini, Rt_ini, obs_t, delta, z, r, dx, iter_max):
    p = beta_ini.shape[0]
    n = Rt_ini.shape[0]
    zc = z - np.mean(z, axis = 1).reshape((p, 1))
    index = np.argsort(obs_t)
    ord_delta = delta[index]
    ord_z = zc[:, index]
    diff = 1
    iter = 0
    while(diff > dx and iter <= iter_max):
        ord_bz = np.dot(beta_ini, ord_z)
        s0 = S0_fun(ord_bz, Rt_ini, r, n)
        Rt = Rt_fun(ord_delta, s0)
        Ubeta = Ubeta_fun(ord_delta, ord_z, ord_bz, Rt, r)
        Hbeta = Hbeta_fun(ord_z, ord_bz, Rt, r, p)
        beta = beta_ini + np.dot(Ubeta, np.linalg.inv(Hbeta))
        diff = np.max(np.abs(beta - beta_ini))
        iter += 1
        beta_ini = beta
        Rt_ini = Rt
    unconverged = (diff > dx).astype(np.float64)

    Rt = np.maximum(Rt, 1e-3)
    Rt0 = np.zeros(n)
    Rt0[0] = 1e-3
    Rt0[1: n] = Rt[0: n - 1]
    dRt = Rt - Rt0
    ord_bz = np.dot(beta, ord_z)
    a = Rt * np.exp(-ord_bz) / (1 + r * Rt * np.exp(-ord_bz))
    B2 = np.zeros(n)
    B1 = np.zeros(n)
    for i in range(n):
        B2[i] = np.sum(Rt[i] * np.exp(-ord_bz[i: n]) / (1 + r * Rt[i] * np.exp(-ord_bz[i: n])))
        B1[i] = np.sum(np.exp(-ord_bz[i: n]) / (1 + r * Rt[i] * np.exp(-ord_bz[i: n])) ** 2)
    bt = np.exp(np.cumsum(dRt * B1 / B2))
    B_zt = np.zeros((p, n))
    Z_bart = np.zeros((p, n))
    A = np.zeros((p, p))
    V = np.zeros((p, p))
    for i in range(n - 1):
        B = np.ones(n - i) * bt[i] / bt[i: n]
        B_zt[:, i] = np.sum(ord_z[:, i: n] * a[i: n] * B, axis = 1)
        Z_bart[:, i] = B_zt[:, i] / B2[i]
        c1 = np.exp(-ord_bz) * dRt[i] / (1 + r * Rt[i] * np.exp(-ord_bz)) ** 2
        A = A + np.dot((ord_z[:, i: n] * c1[i: n]), (ord_z[:, i: n] - Z_bart[:, i].reshape((-1, 1))).T)
        c2 = np.exp(-ord_bz) * dRt[i] / (1 + r * Rt[i] * np.exp(-ord_bz)) 
        V = V + np.dot(((ord_z[:, i: n] - Z_bart[:, i].reshape((-1, 1))) * c2[i: n]), (ord_z[:, i: n] - Z_bart[:, i].reshape((-1, 1))).T)
    B_zt[:, n - 1] = ord_z[:, n - 1] * a[n - 1]
    Z_bart[:, n - 1] = B_zt[:, n - 1] / B2[n - 1]
    d = ord_z[:, n - 1] - Z_bart[:, n - 1]
    A = A + np.dot(d.reshape((p, 1)), ord_z[:, n - 1].reshape((1, p)) * np.exp(-ord_bz[n - 1]) * dRt[n - 1] / (1 + r * Rt[n - 1] * np.exp(-ord_bz[n - 1])) ** 2)
    V = V + np.dot(d.reshape((p, 1)), d.reshape((1, p)) * np.exp(-ord_bz[n - 1]) * dRt[n - 1] / (1 + r * Rt[n - 1] * np.exp(-ord_bz[n - 1])))
    sigma = (np.linalg.inv(A)).T @ V @ np.linalg.inv(A)

    return beta, Rt, sigma, unconverged