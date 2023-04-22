import torch 
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    def __init__(self,n):
        super().__init__()
        '''
        PyTorch中定义模型时,有时候会遇到self.register_buffer('name', Tensor)的操作,
        该方法的作用是定义一组参数,该组参数的特别之处在于:模型训练时不会更新（即调用 optimizer.step() 
        后该组参数不会变化,只可人为地改变它们的值）,但是保存模型时,该组参数又作为模型参数不可或缺的一部分被保存。
        '''
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))


    def _load_from_state_dict(self, 
                    state_dict, 
                    prefix, 
                    local_metadata, 
                    strict,
                    missing_keys, 
                    unexpected_keys, 
                    error_msgs):
        
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    

