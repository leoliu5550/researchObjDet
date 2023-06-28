import torch 
import torch.nn as nn

class scale_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear()
        