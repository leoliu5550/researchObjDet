# print(yaml.dump(data))
# print('---')
# print(type(data))
# print(data)
# print('---')
# print(data['name'])
# print(data['age'])
# import torchvision
# print(dir(torchvision.models))

# print(getattr(torchvision.models, name))


# import os,logging

# import logging,logging.config
# import temp2 as mod

# logging.config.fileConfig('cfg/logger.conf')
# root_logger = logging.getLogger('root')

# root_logger.debug('MainProg:Test Root Logger...')
# logger = logging.getLogger('main')
# logger.info('Test Main Logger')

# mod.testLogger()#子模块


import torch
import torch.nn as nn
import torchvision.models as models
# from collections import OrderedDict

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

# weights=ResNet18_Weights.DEFAULT
print(model)

# from torchvision.models import resnet18, ResNet18_Weights


# class Predictor(nn.Module):

#     def __init__(self):
#         super().__init__()
#         weights = ResNet18_Weights.DEFAULT
#         self.resnet18 = resnet18(weights=weights, progress=False).eval()
#         self.transforms = weights.transforms()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             x = self.transforms(x)
#             y_pred = self.resnet18(x)
#             return y_pred.argmax(dim=1)