import torch 
import sys
sys.path.append('.')
import torch.nn as nn
from model.position_encoding import PositionEmbeddingSine


class selfattention(nn.Module):
    def __init__(self):
        super().__init__()
        # self.postin = PositionEmbeddingSine()

    def forward(self,q_vale,k_vale,v_vale):
        batch_size, length,heads, d_tensor = k_vale.size()
        # q_vale = self.postin(q_vale)
        # k_vale = self.postin(k_vale)
        a_vale = q_vale @ k_vale.permute(0,1,3,2)  / torch.sqrt(torch.tensor(d_tensor))
        b_vale = a_vale @ v_vale
        return b_vale
    
class multhead_position(nn.Module):
    def __init__(self,num_head,d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.selfatten = selfattention()
        self.postin = PositionEmbeddingSine(self.d_model)
        self.w_q = nn.Linear(self.d_model,self.d_model)
        self.w_k = nn.Linear(self.d_model,self.d_model)
        self.w_v = nn.Linear(self.d_model,self.d_model)

    def forward(self,x):
        # 加入位置資訊
        w_q = self.postin(self.w_q(x))
        w_k = self.postin(self.w_k(x))

        q_vale = self.split(w_q)
        k_vale = self.split(w_k)
        v_vale = self.split(self.w_v(x))
        x = self.selfatten(q_vale,k_vale,v_vale)
        x = self.concat(x)
        return x
    
    def split(self,x):
        batch_size,rows, _ = x.size()
        d_tensor = self.d_model//self.num_head
        x = x.view(batch_size, rows, self.num_head, d_tensor)
        return x

    def concat(self,x):
        batch_size, rows,heads, d_tensor = x.size()
        x = x.view(batch_size, rows,self.d_model)
        return x