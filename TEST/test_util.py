import pytest
import sys,os
sys.path.append('.')
from model.detr.util.misc import *
import torch

class Test_NestedTensor:
    
    im0 = torch.rand(3,200,200)
    mask = torch.rand(3,200,250)
    nt = NestedTensor(im0,mask)
    def test_init(self):
        self.nt.tensors.shape == torch.Size([3,200,200])
        self.nt.mask.shape == torch.Size([3,200,250])

        # x = nested_tensor_from_tensor_list([im0, im1])





#     mt = tt()
#     def test_o1(self):
#         assert self.mt.pp() == 1
# class tt:
#     def __init__(self) -> None:
#         pass
    
#     def pp(self):
#         return 0