import numpy as np
import torch
import math
from itertools import product
from torch.utils.data import TensorDataset, DataLoader
from Functions_and_Models import *

def DPLTM_selection(z, x, obs, delta, r):
    num_hidden_set = [1, 2, 3, 5, 7]
    num_neurons_set = [5, 10, 15, 20, 50]
    num_epochs_set = [100, 200, 500]
    learning_rate_set = [1e-3, 2e-3, 5e-3, 1e-2]
    dropout_rate_set = [0, 0.1, 0.2, 0.3]
    num_splines_set = [i for i in range(math.floor(math.pow(n, 1/3 + 1e-7)), 2 * math.floor(math.pow(n, 1/3 + 1e-7)) + 1)]
    iter = product(num_hidden_set, num_neurons_set, num_epochs_set, learning_rate_set, dropout_rate_set, num_splines_set)
   
    batch_size = 64
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_decay = 1e-3
    max_likelihood = -1e7
    best_hyperparams = {}

    n = z.shape[0]
    z = z.to(device)
    x = x.to(device)
    obs = obs.to(device)
    delta = delta.to(device)
    
    for hyperparams in iter:
        num_hidden = hyperparams[0]
        num_neurons = hyperparams[1]
        num_epochs = hyperparams[2]
        learning_rate = hyperparams[3]
        dropout_rate = hyperparams[4]
        num_splines = hyperparams[5]

        T = torch.zeros(num_splines + 4)
        max_time = torch.max(obs)
        T[num_splines + 1: ] = torch.ones(3) * max_time
        T[3: num_splines + 1] = torch.linspace(0, max_time, num_splines - 2)
        splines = torch.zeros((n, num_splines))
        derivatives = torch.zeros((n, num_splines))
        for i in range(n):
            for j in range(num_splines):
                splines[i, j] = B_spline(j, 3, T, obs[i])
                derivatives[i, j] = spline_derivative(j, 3, T, obs[i])

        train_z = z[0: n // 5 * 4]
        val_z = z[n // 5 * 4: n]
        train_x = x[0: n // 5 * 4]
        val_x = x[n // 5 * 4: n]
        train_splines = splines[0: n // 5 * 4]
        val_splines = splines[n // 5 * 4: n]
        train_derivatives = derivatives[0: n // 5 * 4]
        val_derivatives = derivatives[n // 5 * 4: n]
        train_delta = delta[0: n // 5 * 4]
        val_delta = delta[n // 5 * 4: n]

        train_data = TensorDataset(train_z, train_x, train_splines, train_derivatives, train_delta)
        val_data = TensorDataset(val_z, val_x, val_splines, val_derivatives, val_delta)
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(3407)
        else:
            torch.manual_seed(3407)

        model = DPLTM(z_dim = train_z.shape[1], x_dim = train_x.shape[1], num_splines = num_splines, num_hidden = num_hidden, num_neurons = num_neurons, dropout_rate = dropout_rate)
        model = model.to(device)
        lossfun = LogLikelihood(r)
        lossfun = lossfun.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)                        

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

        beta, theta, g, risk, pred = model(val_z, val_x, val_splines)
        Likelihood = - lossfun(pred, val_delta, theta, val_derivatives).item()
        print(Likelihood)
        if Likelihood > max_likelihood:
            max_likelihood = Likelihood
            best_hyperparams = {'num_hidden': num_hidden, 'num_neurons': num_neurons, 'num_epochs': num_epochs, 'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'num_splines': num_splines}
    return best_hyperparams