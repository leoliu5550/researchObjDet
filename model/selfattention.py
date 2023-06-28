import torch 
import torch.nn as nn


class selfattention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

    def forward(self,x):
        q_vale = self.w_q(x)

        return q_vale 