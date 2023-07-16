import torch
from torch import nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self,d_model,max_rows = 5000,weight = 10000.0):
        super().__init__()
        self.weight = weight
        position = torch.zeros(max_rows,d_model)
        pe = torch.arange(0, max_rows, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(self.weight)/d_model))
        position[..., 0::2] = torch.sin(pe * div_term)
        position[..., 1::2] = torch.cos(pe * div_term)
        position = position.unsqueeze(0)
        # .transpose(0, 1)
        self.register_buffer('position', position)
    
    def forward(self,x):
        x = x + self.position[:,:x.size(1),:]
        return x
