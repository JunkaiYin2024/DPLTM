import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from Preprocess import *
from Functions_and_Models import *
from HARE import *
from Variance import *
from DPLTM_selection import *

def Deep_Estimation(r, train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta):
    z_dim = train_z.shape[1]
    x_dim = train_x.shape[1]
    n = train_z.shape[0]
    n_test = test_z.shape[0]

    train_z = torch.from_numpy(train_z).float()
    test_z = torch.from_numpy(test_z).float()
    train_x = torch.from_numpy(train_x).float()
    test_x = torch.from_numpy(test_x).float()
    train_obs = torch.from_numpy(train_obs).float()
    test_obs = torch.from_numpy(test_obs).float()
    train_delta = torch.from_numpy(train_delta).float()
    test_delta = torch.from_numpy(test_delta).float()

    best_hyperparams = DPLTM_selection(train_z, train_x, train_obs, train_delta, r)
    num_hidden = best_hyperparams['num_hidden']
    num_neurons = best_hyperparams['num_neurons']
    num_epochs = best_hyperparams['num_epochs']
    learning_rate = best_hyperparams['learning_rate']
    dropout_rate = best_hyperparams['dropout_rate']
    num_splines = best_hyperparams['num_splines']

    T = torch.zeros(num_splines + 4)
    max_time = torch.maximum(torch.max(train_obs), torch.max(test_obs))
    T[num_splines + 1: ] = torch.ones(3) * max_time
    T[3: num_splines + 1] = torch.linspace(0, max_time, num_splines - 2)
    train_splines = torch.zeros((n, num_splines))
    test_splines = torch.zeros((n_test, num_splines))
    train_derivatives = torch.zeros((n, num_splines))
    for i in range(n):
        for j in range(num_splines):
            train_splines[i, j] = B_spline(j, 3, T, train_obs[i])
            train_derivatives[i, j] = spline_derivative(j, 3, T, train_obs[i])

    for i in range(n_test):
        for j in range(num_splines):
            test_splines[i, j] = B_spline(j, 3, T, test_obs[i])

    beta_est = np.zeros(z_dim)
    beta_sd = np.zeros(z_dim)
    num_ticks = 80
    ICI = np.zeros(num_ticks)

    weight_decay = 1e-3
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_z = train_z.to(device)
    train_x = train_x.to(device)
    train_splines = train_splines.to(device)
    train_derivatives = train_derivatives.to(device)
    train_obs = train_obs.to(device)
    train_delta = train_delta.to(device)
    
    final_train_z = train_z[0: n // 5 * 4]
    val_z = train_z[n // 5 * 4: n]
    final_train_x = train_x[0: n // 5 * 4]
    val_x = train_x[n // 5 * 4: n]
    final_train_splines = train_splines[0: n // 5 * 4]
    val_splines = train_splines[n // 5 * 4: n]
    final_train_derivatives = train_derivatives[0: n // 5 * 4]
    val_derivatives = train_derivatives[n // 5 * 4: n]
    final_train_obs = train_obs[0: n // 5 * 4].view(-1, 1)
    val_obs = train_obs[n // 5 * 4: n]
    final_train_delta = train_delta[0: n // 5 * 4]
    val_delta = train_delta[n // 5 * 4: n]

    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    else:
        torch.manual_seed(3407)
    model = DPLTM(z_dim = z_dim, x_dim = x_dim, num_splines = num_splines, num_hidden = num_hidden, num_neurons = num_neurons, dropout_rate = dropout_rate)
    model = model.to(device)
    lossfun = LogLikelihood(r)
    lossfun = lossfun.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    train_data = TensorDataset(final_train_z, final_train_x, final_train_splines, final_train_derivatives, final_train_delta)
    val_data = TensorDataset(val_z, val_x, val_splines, val_derivatives, val_delta)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    early_stopping_flag = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_all = 0
        for z, x, splines, derivatives, event in train_loader:
            beta, theta, g, risk, pred = model(z, x, splines)
            loss = lossfun(pred, event, theta, derivatives)                            
            loss.backward()
            train_loss_all += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        train_loss[epoch] = train_loss_all

        model.eval()
        val_loss_all = 0
        for z, x, splines, derivatives, event in val_loader:
            beta, theta, g, risk, pred = model(z, x, splines)
            loss = lossfun(pred, event, theta, derivatives)
            val_loss_all += loss.item()

        val_loss[epoch] = val_loss_all

        if epoch == 0:
            early_stopping_flag = 0
        else:
            if val_loss[epoch] <= val_loss[epoch - 1]:
                early_stopping_flag = 0
            else:
                early_stopping_flag += 1
                if early_stopping_flag > patience:
                    break

    beta, theta, g, risk, pred = model(final_train_z, final_train_x, final_train_splines)
    theta = theta.detach()
    pred = pred.detach()

    beta_est = beta.cpu().detach().numpy()
    sd = variance(final_train_z, final_train_x, pred, final_train_obs, final_train_delta, r, device)
    beta_sd = sd.cpu().detach().numpy()

    beta, theta, g, risk, pred = model(val_z, val_x, val_splines)
    Likelihood = - lossfun(pred, val_delta, theta, val_derivatives).item()

    model = model.to('cpu')
    beta, theta, g, risk, pred = model(test_z, test_x, test_splines)
    theta = theta.detach().numpy()

    risk = risk.detach().numpy()
    time = test_obs.detach().numpy()
    event = test_delta.detach().numpy()
    numerator = 0
    denominator = 0
    for j in range(time.shape[0]):
        for k in range(time.shape[0]):
            numerator += event[j] * (risk[j] >= risk[k]).astype(np.float64) * (time[j] <= time[k]).astype(np.float64)
            denominator += event[j] * (time[j] <= time[k]).astype(np.float64)
    c_index = numerator / denominator                     

    for k in range(num_ticks):
        time_tick = (k + 1) / 12
        H = 0
        for j in range(num_splines):
            H += theta[j] * B_spline(j, 3, T, time_tick).numpy()
        pred = np.exp(risk + H)
        if r == 0:
            P_hat = 1 - np.exp(- pred)
        elif r > 0:
            P_hat = 1 - np.exp(- np.log(1 + r * pred) / r)
        ICI[k] = HARE(time_tick, test_delta, test_obs, P_hat)
    
    return beta_est, beta_sd, Likelihood, c_index, ICI