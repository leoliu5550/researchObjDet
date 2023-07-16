import pytest
import math
import sys,os
import torch
import torch.nn as nn
sys.path.append('.')
from model.selfattention import *
from model.position_encoding import PositionEmbeddingSine 
D_MODEL = 6
NUM_HEAD = 2
class Testbselfattention:
    def test_selfattention(self):
        # batch,rows,heads,d_tensor
        x = torch.ones([1,3,2,3])
        model = selfattention()

        assert model(x,x,x).shape == x.shape
        

class Testmulthead:
    def test_multhead(self):
        # batch,rows,d_model
        x = torch.ones([9,3,D_MODEL])
        # num_head and d_model
        model = multhead(NUM_HEAD,D_MODEL)
        assert model(x).shape == x.size()



class Testpostencode:
    # batch,rows,d_model
    x = torch.zeros([9,3,D_MODEL])
    model = PositionEmbeddingSine(D_MODEL)
    weight = model.weight
    def test_SinCos(self):
        # print()
        # print("#"*100)
        # print(self.model(self.x))
        assert self.model(self.x).shape == self.x.shape
        
