import argparse
import logging
import sys
from copy import deepcopy
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
import thop

import torch
import torch.nn as nn



class Detect(nn.Module):
    stride = None
    export = False
    end2end = False
    includes_nums = False
    concat = False 

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect,self).__init__()
        # self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        # self.nl = len(anchors)  # number of detection layers
        # self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [torch.zeros(1)] * self.nl  # init grid
        # a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(
        #     nn.Conv2d(x, self.no * self.na, 1) for x in ch
        #     )  # output conv
        
    def forward(self):
        # x = x.copy()  # for profiling
        z = []  # inference output
        
        self.training |= self.export
        print(self.export)
        print(self.training)
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

    #         if not self.training:  # inference
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
    #             y = x[i].sigmoid()
    #             if not torch.onnx.is_in_onnx_export():
    #                 y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:
    #                 xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
    #                 xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
    #                 wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
    #                 y = torch.cat((xy, wh, conf), 4)
    #             z.append(y.view(bs, -1, self.no))

    #     if self.training:
    #         out = x
    #     elif self.end2end:
    #         out = torch.cat(z, 1)
    #     elif self.include_nms:
    #         z = self.convert(z)
    #         out = (z, )
    #     elif self.concat:
    #         out = torch.cat(z, 1)
    #     else:
    #         out = (torch.cat(z, 1), x)

    #     return out

    # @staticmethod
    # def _make_grid(nx=20, ny=20):
    #     yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    #     return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    # def convert(self, z):
    #     z = torch.cat(z, 1)
    #     box = z[:, :, :4]
    #     conf = z[:, :, 4:5]
    #     score = z[:, :, 5:]
    #     score *= conf
    #     convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
    #                                        dtype=torch.float32,
    #                                        device=z.device)
    #     box @= convert_matrix                          
        # return (box, score)

det = Detect()
det.forward()