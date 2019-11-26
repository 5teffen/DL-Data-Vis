import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from my_classes import PruneLinear


# no_layers = 5
# print(np.logspace(np.log2(3), np.log2(784), num = no_layers, base = 2.0))  # Distributes hidden layers

class Autoenc(nn.Module):
    
    def __init__(self):
        super(Autoenc, self).__init__()
        # old: 784 -> 128 -> 64 -> 12 -> 3
        # new: 784 -> 194 -> 48 -> 12 -> 3
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 194),nn.ReLU(True),  # Modify input according to dataset
            nn.Linear(194, 48), nn.ReLU(True), 
            nn.Linear(48, 12), nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),nn.ReLU(True),
            nn.Linear(12, 48),nn.ReLU(True),
            nn.Linear(48, 194),nn.ReLU(True), 
            nn.Linear(194, 28 * 28), nn.Tanh())     # Modify output according to dataset
        
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x,code

class Autoenc_w_Pruning(nn.Module):
    def __init__(self):
        super(Autoenc_w_Pruning, self).__init__()
        # old: 784 -> 128 -> 64 -> 12 -> 3
        # new: 784 -> 194 -> 48 -> 12 -> 3
        self.encoder = nn.Sequential(
            PruneLinear(28 * 28, 194),nn.ReLU(True),  # Modify input according to dataset
            PruneLinear(194, 48), nn.ReLU(True), 
            PruneLinear(48, 12), nn.ReLU(True), 
            PruneLinear(12, 3))
        
        self.decoder = nn.Sequential(
            PruneLinear(3, 12),nn.ReLU(True),
            PruneLinear(12, 48),nn.ReLU(True),
            PruneLinear(48, 194),nn.ReLU(True), 
            PruneLinear(194, 28 * 28), nn.Tanh())     # Modify output according to dataset
        
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x,code

    def set_masks(self, masks):
        self.encoder[0].set_mask(masks[0])
        self.encoder[2].set_mask(masks[1])
        self.encoder[4].set_mask(masks[2])
        self.encoder[6].set_mask(masks[3])

        self.decoder[0].set_mask(masks[4])
        self.decoder[2].set_mask(masks[5])
        self.decoder[4].set_mask(masks[6])
        self.decoder[6].set_mask(masks[7])


class Autoenc2(nn.Module):
    
    def __init__(self):
        super(Autoenc2, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(784, 392),nn.ReLU(True),  # Modify input according to dataset
            nn.Linear(392, 196), nn.ReLU(True), 
            nn.Linear(196, 98), nn.ReLU(True), 
            nn.Linear(98, 49), nn.ReLU(True), 
            nn.Linear(49, 24), nn.ReLU(True), 
            nn.Linear(24, 12), nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),nn.ReLU(True),
            nn.Linear(12, 24),nn.ReLU(True),
            nn.Linear(24, 49),nn.ReLU(True),
            nn.Linear(49, 98),nn.ReLU(True),
            nn.Linear(98, 196),nn.ReLU(True), 
            nn.Linear(196, 394),nn.ReLU(True),
            
            nn.Linear(394, 784), nn.Tanh())     # Modify output according to dataset
        
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x,code