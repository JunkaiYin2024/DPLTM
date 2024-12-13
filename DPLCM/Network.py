import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, z_dim, x_dim, num_hidden, num_neurons, dropout_rate):
        super(DNN, self).__init__()
        self.beta = nn.Parameter(torch.zeros(z_dim))
        layers = []
        layers.append(nn.Linear(x_dim, num_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for i in range(num_hidden - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_neurons, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, z, x):
        g = torch.squeeze(self.model(x), dim = 1)
        risk = torch.matmul(z, self.beta) + g
        return self.beta, g, risk

class PartialLikelihood(nn.Module):
    def __init__(self):
        super(PartialLikelihood, self).__init__()
    
    def forward(self, pred, time, event):
        sort_time = torch.argsort(time, 0, descending = True)
        event = torch.gather(event, 0,  sort_time)
        risk = torch.gather(pred, 0, sort_time)
        exp_risk = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(exp_risk, 0))
        censored_likelihood = (risk - log_risk) * event
        return - censored_likelihood.sum()

