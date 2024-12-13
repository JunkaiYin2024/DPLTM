import matplotlib.pyplot as plt
from scipy.stats import norm
from Preprocess import *
from Application.Linear_model import *
from Application.Additive_model import *
from Application.Deep_model import *
from Application.DPLCM import *

if __name__ == "__main__":
    r_set = [0, 0.5, 1]
    num_ticks = 80
    Likelihood_Deep = np.zeros(3)
    train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta = Dataset()
    z_dim = train_z.shape[1]
    beta_est = np.zeros((3, z_dim))
    beta_sd = np.zeros((3, z_dim))
    c_index_set = np.zeros(3)
    ICI_set = np.zeros((3, num_ticks))
    ICI_all = np.zeros((4, num_ticks))

    for i in range(len(r_set)):
        r = r_set[i]
        beta, sd, Likelihood, c_index, ICI = Deep_Estimation(r, train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta) 
        beta_est[i] = beta
        beta_sd[i] = sd
        Likelihood_Deep[i] = Likelihood
        c_index_set[i] = c_index
        ICI_set[i] = ICI
    
    print('r=0, Likelihood: {:.2f}\nr=0.5, Likelihood: {:.2f}\nr=1, Likelihood: {:.2f}'.format(Likelihood_Deep[0], Likelihood_Deep[1], Likelihood_Deep[2]))
    index = np.argmax(Likelihood_Deep)
    print('Best r: {}\n'.format(r_set[index]))

    z_value = beta_est / beta_sd
    p_value = 2 * (1 - norm.cdf(np.abs(z_value)))
    print('beta:\n', beta_est, '\nstandard error:\n', beta_sd, '\np_value\n', p_value, '\n')

    c_index_Deep = c_index_set[index]
    ICI_all[2] = ICI_set[index]
    beta_Linear, sd_Linear, c_index_Linear, ICI_Linear = Linear_Estimation(r_set[index], train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta)
    ICI_all[0] = ICI_Linear
    beta_Additive, sd_Additive, c_index_Additive, ICI_Additive = Additive_Estimation(r_set[index], train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta)
    ICI_all[1] = ICI_Additive
    Likelihood, c_index_DPLCM, ICI_DPLCM = DPLCM_Estimation(train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta)
    ICI_all[3] = ICI_DPLCM

    print('c_index:\nLinear model: {:.4f}\nAdditive model: {:.4f}\nDeep model: {:.4f}\nDPLCM: {:.4f}'
            .format(c_index_Linear, c_index_Additive, c_index_Deep, c_index_DPLCM))

    fig, ax1 = plt.subplots()
    t = np.arange(1, num_ticks + 1)
    ax1.plot(t, ICI_all[0], color = 'orange', label='LTM')
    ax1.plot(t, ICI_all[1], color = 'red', label='PLATM')
    ax1.plot(t, ICI_all[2], color = 'blue', label='DPLTM')
    ax1.grid()
    ax1.set_xlabel('Follow-up Time (Month)')
    ax1.set_ylabel('ICI(t)')
    fig.legend(loc = 'upper right')
    plt.savefig('./ICI_comparison1.pdf', dpi=300, bbox_inches='tight')

    fig, ax2 = plt.subplots()
    t = np.arange(1, num_ticks + 1)
    ax2.plot(t, ICI_all[2], color = 'blue', label='DPLTM')
    ax2.plot(t, ICI_all[3], color = 'red', label='DPLCM')
    ax2.grid()
    ax2.set_xlabel('Follow-up Time (Month)')
    ax2.set_ylabel('ICI(t)')
    fig.legend(loc = 'upper right')
    plt.savefig('./ICI_comparison2.pdf', dpi=300, bbox_inches='tight')