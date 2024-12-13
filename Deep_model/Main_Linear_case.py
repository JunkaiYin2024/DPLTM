import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from DataGen import *
from Network import *
from Evaluation import *
from Variance import *
from Model_selection import *
    
if __name__ == "__main__":
    n_set = [1000, 2000]
    r_set = [0, 0.5, 1]
    c_rate = [0.4, 0.6]
    c_set = [2.95, 0.85, 2.75, 0.9, 2.55, 1]
    num_sim = 200
    for i1 in range(len(r_set)): 
        for i2 in range(len(c_rate)):
            for i3 in range(len(n_set)):
                c = c_set[2 * i1 + i2]
                r = r_set[i1]
                n = n_set[i3]

                best_hyperparams = model_selection(n, r, c, 'Linear')
                num_hidden = best_hyperparams['num_hidden']
                num_neurons = best_hyperparams['num_neurons']
                num_epochs = best_hyperparams['num_epochs']
                learning_rate = best_hyperparams['learning_rate']
                dropout_rate = best_hyperparams['dropout_rate']
                num_splines = best_hyperparams['num_splines']

                beta_est = torch.zeros((num_sim, 2))                
                beta_sd = torch.zeros((num_sim, 2))
                theta_est = torch.zeros((num_sim, num_splines))
                train_obs = torch.zeros((num_sim, n // 5 * 4))
                Likelihood = torch.zeros(num_sim)
                g_mean = torch.zeros(num_sim)

                weight_decay = 1e-3
                batch_size = n // 100
                patience = 7
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               
                for i in range(num_sim):
                    train_z, train_x, train_splines, train_derivatives, train_time, train_event = DataGeneration(n, r, c, i, num_splines, 'train', 'Linear')
                    train_obs[i] = train_time[0: n // 5 * 4]
                    train_z = train_z.to(device)
                    train_x = train_x.to(device)
                    train_splines = train_splines.to(device)
                    train_derivatives = train_derivatives.to(device)
                    train_time = train_time.to(device)
                    train_event = train_event.to(device)
                    train_data = TensorDataset(train_z[0: n // 5 * 4], train_x[0: n // 5 * 4], train_splines[0: n // 5 * 4], train_derivatives[0: n // 5 * 4], train_event[0: n // 5 * 4])
                    val_data = TensorDataset(train_z[n // 5 * 4: n], train_x[n // 5 * 4: n], train_splines[n // 5 * 4: n], train_derivatives[n // 5 * 4: n], train_event[n // 5 * 4: n])
                    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
                    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)

                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(3407 * (i + 1))
                    else:
                        torch.manual_seed(3407 * (i + 1))
                    model = DNN(z_dim = train_z.shape[1], x_dim = train_x.shape[1], num_splines = num_splines, num_hidden = num_hidden, num_neurons = num_neurons, dropout_rate = dropout_rate)
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
                                
                    train_z = train_z[0: n // 5 * 4]
                    train_x = train_x[0: n // 5 * 4]
                    train_splines = train_splines[0: n // 5 * 4]
                    train_derivatives = train_derivatives[0: n // 5 * 4]
                    train_time = train_time[0: n // 5 * 4].view(-1, 1)
                    train_event = train_event[0: n // 5 * 4]
                    beta, theta, g, risk, pred = model(train_z, train_x, train_splines)
                    theta = theta.detach()
                    pred = pred.detach()

                    beta_est[i] = beta.cpu().detach()
                    beta_sd[i] = variance(train_z, train_x, pred, train_time, train_event, r, device)
                    theta_est[i] = theta.cpu().detach()
                    g_mean[i] = g.mean()

                est_mean = beta_est.mean(0)
                est_sd = beta_est.std(0)
                sd_mean = beta_sd.mean(0)
                theta_hat = theta_est.mean(0)

                b1 = beta_est + 1.96 * beta_sd
                b2 = beta_est - 1.96 * beta_sd

                cov_p1 = (((b1[:, 0] >= 1) * (b2[:, 0] <= 1)).sum().item()) / num_sim
                cov_p2 = (((b1[:, 1] >= -1) * (b2[:, 1] <= -1)).sum().item()) / num_sim

                print("n: {}, r: {}, censoring_rate: {}, c: {}\nBest hyperparameters: {}".format(n, r, c_rate[i2], c, best_hyperparams))
                print("bias_mean: beta_1: {:.4f}, beta_2:{:.4f}\nest_sd: beta_1:{:.4f}, beta_2:{:.4f}\nsd_mean: beta_1:{:.4f}, beta_2:{:.4f}\ncov_p: beta_1:{}, beta_2:{}" \
                    .format(est_mean[0].item() - 1, est_mean[1].item() + 1, est_sd[0].item(), est_sd[1].item(), sd_mean[0].item(), sd_mean[1].item(), cov_p1, cov_p2))
                
                WISE = WISE_func(r, theta_est, g_mean, train_obs)
                print("WISE_mean: {:.4f}, WISE_std: {:.4f}".format(WISE.mean().item(), WISE.std().item()))

                model = model.to('cpu')
                n_test = n // 5
                sim_test = 200
                RE = torch.zeros(sim_test)
                ICI_25 = torch.zeros(sim_test)
                ICI_50 = torch.zeros(sim_test)
                ICI_75 = torch.zeros(sim_test)
                c_index = torch.zeros(sim_test)

                for i in range(sim_test):
                    test_z, test_x, test_splines, test_derivatives, test_time, test_event = DataGeneration(n_test, r, c, i, num_splines, 'test', 'Linear')
                    beta, theta, g_hat, risk, pred = model(test_z, test_x, test_splines)
                    RE[i] = RE_func(g_hat, test_x, 'Linear')
                    ICI_25[i] = ICI_func(r, 25, risk, theta_hat, test_event, test_time)
                    ICI_50[i] = ICI_func(r, 50, risk, theta_hat, test_event, test_time)
                    ICI_75[i] = ICI_func(r, 75, risk, theta_hat, test_event, test_time)
                    c_index[i] = c_index_func(risk, test_time, test_event)

                print("RE_mean: {:.4f}, RE_std: {:.4f}".format(RE.mean().item(), RE.std().item()))
                print("q: 25, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_25[~torch.isnan(ICI_25)].mean().item(), ICI_25[~torch.isnan(ICI_25)].std().item()))
                print("q: 50, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_50[~torch.isnan(ICI_50)].mean().item(), ICI_50[~torch.isnan(ICI_50)].std().item()))
                print("q: 75, ICI_mean: {:.4f}, ICI_std: {:.4f}" .format(ICI_75[~torch.isnan(ICI_75)].mean().item(), ICI_75[~torch.isnan(ICI_75)].std().item()))
                print("c_index_mean: {:.4f}, c_index_std: {:.4f}\n" .format(c_index.mean().item(), c_index.std().item()))