import sys
sys.path.append('.')
import torch
import torch.nn as nn
from model.selfattention import multhead_position

from model.position_encoding import PositionEmbeddingSine


class encoder_block(nn.Module):
    def __init__(self,num_head,d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.normlayer = nn.LayerNorm(d_model)
        self.multattention = multhead_position(
            self.num_head,
            self.d_model
            )
        self.linear = nn.Sequential(
            nn.Linear(self.d_model,self.d_model),
            nn.ReLU()
            )
        

    def forward(self,x):
        x_ = x
        x = self.multattention(x) + x_
        x = self.normlayer(x)
        x_ = x
        x = self.linear(x) + x_
        x = self.normlayer(x)

        return x
    
D_MODEL = 6
NUM_HEAD = 2

# batch,rows,d_model
x = torch.ones([9,3,D_MODEL])
model = encoder_block(NUM_HEAD,D_MODEL)
print(model(x))