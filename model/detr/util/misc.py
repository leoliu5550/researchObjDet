from typing import Optional
from torch import Tensor
'''
PyTorch中包含的分布式软件包（即torch.distributed）
'''
import torch.distributed as dist

def is_dist_avail_and_initialized():
    # 分布式訓練設定
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


class NestedTensor(object):
    '''
    定義一個類似{tensor,tensor},的巢狀張量
    '''

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
    def to(self, device):
        # type Device --> NestedTensor
        # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)



