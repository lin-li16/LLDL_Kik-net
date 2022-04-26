import torch
import torchvision
import torch.nn as nn
import numpy as np


class PhyCNN(nn.Module):
    def __init__(self):
        super(PhyCNN, self).__init__()
        n = 3000
        dt = 0.02
        phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
        temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
        temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
        phi2 = temp1 + temp2
        phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
        Phi_t = 1 / dt * np.concatenate([np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
        Phi_t = torch.tensor(Phi_t)
        if torch.cuda.is_available():
            Phi_t = Phi_t.cuda()
        self.Phi_t = Phi_t
        self.cnnlayer = nn.Sequential(
            nn.Conv1d(1, 4, 101, bias=True, padding=50),
            # nn.Sigmoid(),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Conv1d(4, 16, 101, bias=True, padding=50),
            # nn.Sigmoid(),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Conv1d(16, 64, 101, bias=True, padding=50),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # # nn.ReLU(inplace=True),
            # nn.Conv1d(16, 32, 101, bias=True, padding=50),
            # nn.Tanh(),
            # # nn.ReLU(inplace=True),
            # nn.Conv1d(32, 64, 101, bias=True, padding=50),
            nn.Tanh()
            # nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            # nn.Tanh(),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnnlayer(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        return x_tt[:, :, None]
    
    
class PhyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PhyRNN, self).__init__()
        n = 3000
        dt = 0.02
        phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
        temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
        temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
        phi2 = temp1 + temp2
        phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
        Phi_t = 1 / dt * np.concatenate([np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
        Phi_t = torch.tensor(Phi_t)
        if torch.cuda.is_available():
            Phi_t = Phi_t.cuda()
        self.Phi_t = Phi_t
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super(PhyRNN, self).__init__()
        self.rnnlayer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(16, 4),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x, (ht, ct) = self.rnnlayer(x)
        x = self.fc(x)
        
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        
        return x_tt[:, :, None]