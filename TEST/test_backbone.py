import pytest
import dynamic_yaml
import sys,os
sys.path.append('.')
import torch

from model.backbone import backbonebase

def get_hyper():
    CFG_PATH = './cfg/hyperparameter.yaml'
    with open(CFG_PATH, 'r') as file:
        para = dynamic_yaml.safe_load(file)
    return para

# print(get_hyper())
class Testbackbonebase:
    def test_resnet18(self):
        x = torch.ones(1,3,256,256)
        model = backbonebase()
        assert model(x).shape == torch.Size([1,512,3,3])
        
x = torch.ones(1,3,256,256)
model = backbonebase()
print(model(x))


