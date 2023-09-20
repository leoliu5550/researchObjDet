import sys
sys.path.append('.')
import torch 
import torch.nn


from model.rtdetr import rtdetr
from model.rtdetr.hybird_encoder import *

# class Testencoderblock:

#     def test_blockenc(self):
#         num_head = 2
#         d_model = 6
#         x = torch.ones([9,3,d_model])
#         model = encoder_block(num_head,d_model)
#         assert model(x).shape == x.size() 
