import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class DNN_var(nn.Module):
    def __init__(self, x_dim, num_hidden, num_neurons, dropout_rate):
        super(DNN_var, self).__init__()
        layers1 = []
        layers1.append(nn.Linear(1, num_neurons))
        layers1.append(nn.ReLU())
        layers1.append(nn.Dropout(dropout_rate))
        for i in range(num_hidden - 1):
            layers1.append(nn.Linear(num_neurons, num_neurons))
            layers1.append(nn.ReLU())
            layers1.append(nn.Dropout(dropout_rate))
        layers1.append(nn.Linear(num_neurons, 1))
        self.model1 = nn.Sequential(*layers1)

        layers2 = []
        layers2.append(nn.Linear(x_dim, num_neurons))
        layers2.append(nn.ReLU())
        layers2.append(nn.Dropout(dropout_rate))
        for i in range(num_hidden - 1):
            layers2.append(nn.Linear(num_neurons, num_neurons))
            layers2.append(nn.ReLU())
            layers2.append(nn.Dropout(dropout_rate))
        layers2.append(nn.Linear(num_neurons, 1))
        self.model2 = nn.Sequential(*layers2)

    def forward(self, x, t):
        h_proj = torch.squeeze(self.model1(t), dim = 1)
        g_proj = torch.squeeze(self.model2(x), dim = 1)
        return h_proj, g_proj
    
class varloss(nn.Module):
    def __init__(self):
        super(varloss, self).__init__() 
    
    def forward(self, event, z, h_proj, g_proj):
        information = z - h_proj - g_proj
        varloss = event * information ** 2      
        return varloss.sum() / event.shape[0]
    
def variance(z, x, time, event, device):  
    num_covariates = z.shape[1]
    x_dim = x.shape[1]
    learning_rate = 1e-3
    weight_decay = 1e-3
    batch_size = 64
    epochs = 200
    sd = torch.zeros(num_covariates)

    for i in range(num_covariates):
        z0 = z[:, i]
        data = TensorDataset(z0, x, time, event)
        loader = DataLoader(data, batch_size = batch_size, shuffle = True)
        model = DNN_var(x_dim = x_dim, num_hidden = 2, num_neurons = 10, dropout_rate = 0)
        model.to(device)
        lossfun = varloss()
        lossfun.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        Loss = np.zeros(epochs)

        for epoch in range(epochs):
            model.train()
            num, train_loss_all = 0, 0
            for z_temp, x_temp, time_temp, event_temp in loader:
                h_proj, g_proj = model(x_temp, time_temp)
                loss = lossfun(event_temp, z_temp, h_proj, g_proj)                            
                loss.backward()
                train_loss_all += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                num += 1
            Loss[epoch] = train_loss_all / num

        h_proj, g_proj = model(x, time)
        loss = lossfun(event, z0, h_proj, g_proj)
        loss = loss.detach()
        sd[i] = 1 / torch.sqrt(loss * event.shape[0])
    return sd