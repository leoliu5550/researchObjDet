import torch 
import torch.nn as nn


class selfattention(nn.Module):
    def __init__(self,d_model,num_head):
        super().__init__()
        self.num_head = num_head
        # self.
        self.d_model = torch.tensor(d_model) 
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

    def forward(self,x):
        # 1. dot product with weight matrices
        q_vale = self.w_q(x)
        k_vale = self.w_k(x)
        v_vale = self.w_v(x)

        # 2 dot k_vale @ q_value
        a_vale = q_vale @ k_vale.permute(0,1,3,2)  / torch.sqrt(self.d_model)
        b_vale = a_vale @ v_vale
        return b_vale