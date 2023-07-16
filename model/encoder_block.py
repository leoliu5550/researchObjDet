import torch
import torch.nn as nn
from model.selfattention import multhead
class encoder_block:
    def __init__(self,num_head,d_model):
        super().__init__()
        self.num_head = num_head
        self.d_model = d_model
        self.normlayer = nn.LayerNorm(d_model)
        self.multattention = multhead(
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