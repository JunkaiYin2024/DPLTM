import numpy as np
import torch
from DataGen import *
from Network import *
from itertools import product
from torch.utils.data import TensorDataset, DataLoader

def model_selection(n, r, c):
    num_hidden_set = [1, 2, 3, 5, 7]
    num_neurons_set = [5, 10, 15, 20, 50]
    num_epochs_set = [100, 200, 500]
    learning_rate_set = [1e-3, 2e-3, 5e-3, 1e-2]
    dropout_rate_set = [0, 0.1, 0.2, 0.3]
    iter = product(num_hidden_set, num_neurons_set, num_epochs_set, learning_rate_set, dropout_rate_set)
   
    num_sim = 10
    batch_size = n // 100
    patience = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_decay = 1e-3
    max_likelihood = -1e7
    best_hyperparams = {}
    
    for hyperparams in iter:
        num_hidden = hyperparams[0]
        num_neurons = hyperparams[1]
        num_epochs = hyperparams[2]
        learning_rate = hyperparams[3]
        dropout_rate = hyperparams[4]
        Likelihood = 0

        for i in range(num_sim):
            train_z, train_x, train_time, train_event = DataGeneration(n, r, c, i, 'train')
            train_z = train_z.to(device)
            train_x = train_x.to(device)
            train_time = train_time.to(device)
            train_event = train_event.to(device)
            train_data = TensorDataset(train_z[0: n // 5 * 4], train_x[0: n // 5 * 4], train_time[0: n // 5 * 4], train_event[0: n // 5 * 4])
            val_data = TensorDataset(train_z[n // 5 * 4: n], train_x[n // 5 * 4: n], train_time[n // 5 * 4: n], train_event[n // 5 * 4: n])
            train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
            val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(3407 * (i + 1))
            else:
                torch.manual_seed(3407 * (i + 1))

            model = DNN(z_dim = train_z.shape[1], x_dim = train_x.shape[1], num_hidden = num_hidden, num_neurons = num_neurons, dropout_rate = dropout_rate)
            model = model.to(device)
            lossfun = PartialLikelihood()
            lossfun = lossfun.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)                        

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

            beta, g, risk = model(train_z[n // 5 * 4: n], train_x[n // 5 * 4: n])
            Likelihood -= lossfun(risk, train_time[n // 5 * 4: n], train_event[n // 5 * 4: n]).item()

        Likelihood /= num_sim
        print(Likelihood)
        if Likelihood > max_likelihood:
            max_likelihood = Likelihood
            best_hyperparams = {'num_hidden': num_hidden, 'num_neurons': num_neurons, 'num_epochs': num_epochs, 'learning_rate': learning_rate, 'dropout_rate': dropout_rate}
    return best_hyperparams

