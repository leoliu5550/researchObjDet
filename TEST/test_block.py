import torch 
import torch.nn
import pytest
from model.encoder_block import encoder_block
# import sys
# sys.path.append('.')

num_head = 2
d_model = 6
class Testencoderblock:

    def test_blockenc(self):
        x = torch.ones([9,3,d_model])
        model = encoder_block(num_head,d_model)
        assert model(x).shape == x.size() 

