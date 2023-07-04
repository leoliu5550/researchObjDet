import pytest
import sys,os
import torch
import torch.nn as nn
sys.path.append('.')
from model.SelfAttention import *



class Testbselfattention:
    def test_selfattention(self):
        # batch,length,heads,d_tensor
        x = torch.ones([1,3,2,3])
        model = selfattention()

        assert model(x,x,x).shape == x.shape
        

class Testmulthead:
    def test_multhead(self):
        x = torch.ones([1,3,6])
        model = multhead(2,6)
        # assert model(x).shape == torch.Size([1,3,2,3])
        assert model(x).shape == torch.Size([1,3,6])
        
