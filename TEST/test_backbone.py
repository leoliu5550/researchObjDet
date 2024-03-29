import pytest
import dynamic_yaml
import logging
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
@pytest.mark.skip(reason="Skipping this CLASS level test")
class Testbackbonebase:
    def test_resnet18(self):
        x = torch.ones(1,3,256,256)
        model = backbonebase()
        y = 1
        assert model(x).shape == torch.Size([1,512,3,3])
        
        
logger = logging.getLogger('main.mod')
x = torch.ones(1,3,256,256)
model = backbonebase()
print(model.children)

# print(model.modules)


