import pytest
import sys,os
import torch
import torch.nn as nn
sys.path.append('.')
from model.SelfAttention import selfattention



class Testbselfattention:

    def test_selfattention(self):
        x = torch.ones([1,1,3,5])
        model = selfattention(5)

        assert model(x).shape == x.shape
        

