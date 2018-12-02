import torch
import torch.nn as nn
import torch.nn.functional as F


class autoenc(nn.Module):
    
    def __init__(self):
        super(autoenc, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),nn.ReLU(True),  # Modify input according to dataset
            nn.Linear(128, 64), nn.ReLU(True), 
            nn.Linear(64, 12), nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),nn.ReLU(True),
            nn.Linear(12, 64),nn.ReLU(True),
            nn.Linear(64, 128),nn.ReLU(True), 
            
            nn.Linear(128, 28 * 28), nn.Tanh())     # Modify output according to dataset
        
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x