import numpy as np
from DataGen import *
from Evaluation import *
from Algorithm import *

if __name__ == "__main__":
    n_set = [1000, 2000]
    r_set = [0, 0.5, 1]
    c_rate = [0.4, 0.6]
    c_set = [2.95, 0.85, 2.75, 0.9, 2.55, 1]
    for i1 in range(len(r_set)): 
        for i2 in range(len(c_rate)):
            for i3 in range(len(n_set)):
                n = n_set[i3]
                p = 7
                beta_0 = np.array([1, -1])
                c = c_set[2 * i1 + i2]
                num_sim = 200
                r = r_set[i1]
                dx = 1e-3
                iter_max = 1000
                
                beta_est = np.zeros((2, num_sim))
                g_est = np.zeros((p - 2, num_sim))
                beta_sd = np.zeros((2, num_sim))
                train_obs = np.zeros((num_sim, n))
                train_Rt = np.zeros((num_sim, n))

                for i in range(num_sim):
                    z, x, obs_t, delta = DataGeneration(n, r, c, i, 'train', 'Linear')
                    v = np.concatenate([z, x], axis = 0)
                    beta_ini = np.zeros(p)
                    Rt_ini = np.array([1/j for j in range(n, 0, -1)]).cumsum()                
                    beta, Rt, sigma, unconverged = solve_beta(beta_ini, Rt_ini, obs_t, delta, v, r, dx, iter_max)
                    beta_est[:, i] = beta[0: 2]
                    g_est[:, i] = beta[2: p]
                    beta_sd[:, i]= np.sqrt(np.diag(sigma)[0: 2])
                    obs_t.sort()
                    train_obs[i, :] = obs_t
                    train_Rt[i, :] = Rt

                est_mean = beta_est.mean(axis = 1)
                g_coef = g_est.mean(axis = 1)
                est_sd = beta_est.std(axis = 1)
                sd_mean = beta_sd.mean(axis = 1)

                b1 = beta_est + 1.96 * beta_sd
                b2 = beta_est - 1.96 * beta_sd

                cov_p1 = (((b1[0, :] >= 1) * (b2[0, :] <= 1)).sum()) / num_sim
                cov_p2 = (((b1[1, :] >= -1) * (b2[1, :] <= -1)).sum()) / num_sim

                print("n: {}, r: {}, censoring_rate: {}, c: {}".format(n, r, c_rate[i2], c))
                print("bias_mean: beta_1: {:.4f}, beta_2:{:.4f}\nest_sd: beta_1:{:.4f}, beta_2:{:.4f}\nsd_mean: beta_1:{:.4f}, beta_2:{:.4f}\ncov_p: beta_1:{}, beta_2:{}" \
                    .format(est_mean[0] - 1, est_mean[1] + 1, est_sd[0], est_sd[1], sd_mean[0], sd_mean[1], cov_p1, cov_p2))

                WISE = WISE_func(r, train_Rt, train_obs)
                print("WISE_mean: {:.4f}, WISE_std: {:.4f}".format(WISE.mean(), WISE.std()))

                n_test = n // 5
                sim_test = 200
                RE = np.zeros(sim_test)
                ICI_25 = np.zeros(sim_test)
                ICI_50 = np.zeros(sim_test)
                ICI_75 = np.zeros(sim_test)
                c_index = np.zeros(sim_test)

                for i in range(sim_test):
                    z, x, obs_t, delta = DataGeneration(n_test, r, c, i, 'test', 'Linear')
                    risk = np.dot(est_mean, z) + np.dot(g_coef, x) - np.mean(np.dot(g_coef, x))
                    ICI_25[i] = ICI_func(r, 25, risk, train_Rt, train_obs, delta, obs_t)
                    ICI_50[i] = ICI_func(r, 50, risk, train_Rt, train_obs, delta, obs_t)
                    ICI_75[i] = ICI_func(r, 75, risk, train_Rt, train_obs, delta, obs_t)
                    RE[i] = RE_func(g_coef, x, 'Linear')
                    c_index[i] = c_index_func(risk, obs_t, delta)

                print("RE_mean: {:.4f}, RE_std: {:.4f}".format(RE.mean(), RE.std()))
                print("q: 25, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_25[~np.isnan(ICI_25)].mean(), ICI_25[~np.isnan(ICI_25)].std()))
                print("q: 50, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_50[~np.isnan(ICI_50)].mean().item(), ICI_50[~np.isnan(ICI_50)].std()))
                print("q: 75, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_75[~np.isnan(ICI_75)].mean().item(), ICI_75[~np.isnan(ICI_75)].std()))
                print("c_index_mean: {:.4f}, c_index_std: {:.4f}\n" .format(c_index.mean(), c_index.std()))