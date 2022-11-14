import torch
import torchvision
import torch.nn as nn
import numpy as np


class LSTM_basic(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super(LSTM_basic, self).__init__()
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
            Phi_t = Phi_t.cuda().float()
        self.Phi_t = Phi_t
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        # self._initial_parameters()
        

    def forward(self, x):
        x, (ht, ct) = self.rnnlayer(x)
        x = self.fc(x)
        
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        
        return x_tt[:, :, None]
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)


class FC_basic(nn.Module):
    def __init__(self):
        super(FC_basic, self).__init__()
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
            Phi_t = Phi_t.cuda().float()
        self.Phi_t = Phi_t

        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(16, 4),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(4, 1)
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x = self.fc(x)
        
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        
        return x_tt[:, :, None]
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)


class RNN_basic(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super(RNN_basic, self).__init__()
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
            Phi_t = Phi_t.cuda().float()
        self.Phi_t = Phi_t
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnnlayer = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 16),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(16, 4),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(4, 1)
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x, _ = self.rnnlayer(x)
        x = self.fc(x)
        
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        
        return x_tt[:, :, None]
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)


class CNN_basic(nn.Module):
    def __init__(self, kernel_size=101, num_layers=3):
        super(CNN_basic, self).__init__()
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
            Phi_t = Phi_t.cuda().float()
        self.Phi_t = Phi_t
        if num_layers == 1:
            self.cnnlayer = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh()
            )
        elif num_layers == 2:
            self.cnnlayer = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(8, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh()
            )
        elif num_layers == 3:
            self.cnnlayer = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(4, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(16, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh()
            )
        elif num_layers == 4:
            self.cnnlayer = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(4, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(16, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(32, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh()
            )
        elif num_layers == 5:
            self.cnnlayer = nn.Sequential(
                nn.Conv1d(1, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(4, 8, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(8, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(16, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh(),
                nn.Conv1d(32, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
                nn.Tanh()
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
        # self._initial_parameters()
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnnlayer(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        return x_tt[:, :, None]
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.01)
            else:
                nn.init.constant_(p, 0)


class MLP_basic(nn.Module):
    def __init__(self, lens):
        super(MLP_basic, self).__init__()
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
            Phi_t = Phi_t.cuda().float()
        self.Phi_t = Phi_t
        self.lens = lens
        self.MLP = nn.Sequential(
            nn.Linear(self.lens, 2 * self.lens),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(2 * self.lens, 2 * self.lens),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(2 * self.lens, self.lens),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.MLP(x) 
        x = x.permute(0, 2, 1) 
        x = self.fc(x)
        x_t = torch.matmul(self.Phi_t, x[:, :, 0].permute(1, 0))
        x_tt = torch.matmul(self.Phi_t, x_t)
        x_tt = x_tt.permute(1, 0)
        return x_tt[:, :, None]      
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)


class CNN_LSTM(nn.Module):
    def __init__(self, kernel_size=51, LSTM_layers=3) -> None:
        super(CNN_LSTM, self).__init__()
        self.cnnlayer = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.Conv1d(4, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.Conv1d(16, 64, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh()
        )
        self.rnnlayer = nn.LSTM(input_size=64, hidden_size=64, num_layers=LSTM_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.Linear(16, 4),
            nn.Linear(4, 1)
        )


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnnlayer(x)
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.rnnlayer(x)
        x = self.fc(x)
        return x
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.01)
            else:
                nn.init.constant_(p, 0)