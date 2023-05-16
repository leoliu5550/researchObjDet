import torch
import torch.nn as nn
import torchvision.models as models

class backbonebase(nn.Module):
    def __init__(self,net = 'resnet18',pretrain=True,cust_model=None):
        super().__init__()
        weights = None
        if net == 'resnet18':
            if pretrain:
                weights = models.ResNet18_Weights.DEFAULT
            self.net = models.resnet18(weights=weights)
            self.net = torch.nn.Sequential(*(list(self.net.modules())[:-1]))


        if net == 'resnet50':
            if pretrain:
                weights = models.ResNet50_Weights.DEFAULT
            self.net = models.resnet50(weights=weights)
        
        if net == 'resnet101':
            if pretrain:
                weights = models.ResNet101_Weights.DEFAULT
            self.net = models.resnet101(weights=weights)
            
        if net == 'densenet121':
            if pretrain:
                weights = models.DenseNet121_Weights.DEFAULT
            self.net = models.densenet121(weights=weights)
        
        if net == 'densenet161':
            if pretrain:
                weights = models.DenseNet161_Weights.DEFAULT
            self.net = models.densenet161(weights=weights)
        if net == 'custom':
            assert cust_model != None ,"custom model cannot be None"
            self.net = cust_model
        
        # if net != 'custom':
        #     self.net = torch.nn.Sequential(*(list(self.net.modules())[:-1]))

            
    def forward(self,x):
        x = self.net(x)
        return x

