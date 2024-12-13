import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, z_dim, x_dim, num_splines, num_hidden, num_neurons, dropout_rate):
        super(DNN, self).__init__()
        self.beta = nn.Parameter(torch.zeros(z_dim))
        self.gamma = nn.Parameter(-torch.ones(num_splines))
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
    def forward(self, z, x, splines):
        g = torch.squeeze(self.model(x), dim = 1)
        theta = torch.zeros_like(self.gamma)
        theta[0] = self.gamma[0]
        theta[1: ] = torch.exp(self.gamma[1: ])
        theta = torch.cumsum(theta, dim = 0)
        risk = torch.matmul(z, self.beta) + g
        pred = torch.exp(risk + torch.matmul(splines, theta))
        return self.beta, theta, g, risk, pred

class LogLikelihood(nn.Module):
    def __init__(self, r):
        super(LogLikelihood, self).__init__()
        self.r = r 
    
    def forward(self, pred, event, theta, derivatives):
        derivative = torch.matmul(derivatives, theta)
        if self.r == 0:
            hazard = derivative * pred
            cumhazard = pred
        elif self.r > 0:
            hazard = derivative * pred / (1 + self.r * pred)
            cumhazard = torch.log(1 + self.r * pred) / self.r
        Likelihood = event * torch.log(hazard) - cumhazard       
        return - Likelihood.sum()