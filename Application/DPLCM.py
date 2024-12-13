import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from Functions_and_Models import *
from HARE import *
from DPLCM_selection import *

def DPLCM_Estimation(train_z, test_z, train_x, test_x, train_obs, test_obs, train_delta, test_delta):
    z_dim = train_z.shape[1]
    x_dim = train_x.shape[1]
    n = train_z.shape[0]
    
    train_z = torch.from_numpy(train_z).float()
    test_z = torch.from_numpy(test_z).float()
    train_x = torch.from_numpy(train_x).float()
    test_x = torch.from_numpy(test_x).float()
    train_obs = torch.from_numpy(train_obs).float()
    test_obs = torch.from_numpy(test_obs).float()
    train_delta = torch.from_numpy(train_delta).float()
    test_delta = torch.from_numpy(test_delta).float()

    best_hyperparams = DPLCM_selection(train_z, train_x, train_obs, train_delta)
    num_hidden = best_hyperparams['num_hidden']
    num_neurons = best_hyperparams['num_neurons']
    num_epochs = best_hyperparams['num_epochs']
    learning_rate = best_hyperparams['learning_rate']
    dropout_rate = best_hyperparams['dropout_rate']

    num_ticks = 80
    ICI = np.zeros(num_ticks)

    weight_decay = 1e-3
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
            torch.cuda.manual_seed(3407)
    else:
        torch.manual_seed(3407)

    model = DPLCM(z_dim = z_dim, x_dim = x_dim, num_hidden = num_hidden, num_neurons = num_neurons, dropout_rate = dropout_rate)
    model = model.to(device)
    lossfun = PartialLikelihood()
    lossfun = lossfun.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    train_z = train_z.to(device)
    train_x = train_x.to(device)
    train_obs = train_obs.to(device)
    train_delta = train_delta.to(device)

    final_train_z = train_z[0: n // 5 * 4]
    val_z = train_z[n // 5 * 4: n]
    final_train_x = train_x[0: n // 5 * 4]
    val_x = train_x[n // 5 * 4: n]
    final_train_obs = train_obs[0: n // 5 * 4]
    val_obs = train_obs[n // 5 * 4: n]
    final_train_delta = train_delta[0: n // 5 * 4]
    val_delta = train_delta[n // 5 * 4: n]

    train_data = TensorDataset(final_train_z, final_train_x, final_train_obs, final_train_delta)
    val_data = TensorDataset(val_z, val_x, val_obs, val_delta)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    early_stopping_flag = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_all = 0
        for z, x, time, event in train_loader:
            beta, g, risk = model(z, x)
            loss = lossfun(risk, time, event)                            
            loss.backward()
            train_loss_all += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        train_loss[epoch] = train_loss_all

        model.eval()
        val_loss_all = 0
        for z, x, time, event in val_loader:
            beta, g, risk = model(z, x)
            loss = lossfun(risk, time, event)
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

    beta, g, risk = model(val_z, val_x)
    Likelihood = - lossfun(risk, val_obs, val_delta).item()
    
    model = model.to('cpu')
    beta, g, risk = model(test_z, test_x)

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

        ascending_sort_time = np.argsort(time)
        descending_sort_time = np.flip(ascending_sort_time)
        risk = risk[descending_sort_time]
        hazard = 1 / (np.cumsum(np.exp(risk)))
        hazard = np.flip(hazard)
        time = time[ascending_sort_time]
        event = event[ascending_sort_time]
        NA_est = np.cumsum(event * hazard)
        index = time.searchsorted(time_tick, side = 'right')
        cum_hazard = NA_est[index - 1]

        P_hat = 1 - np.exp(- cum_hazard * np.exp(risk))
        ICI[k] = HARE(time_tick, event, time, P_hat)

    return Likelihood, c_index, ICI