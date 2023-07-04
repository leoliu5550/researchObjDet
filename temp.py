import torch
d_model = 6
# a = torch.arange(1,25)
a = torch.arange(1,13)
# [batch_szie,row_data,data]

a = a.reshape([1,2,d_model])

import torch.nn as nn

tensor = torch.FloatTensor([[1, 2, 4, 1],
                            [6, 3, 2, 4],
                            [2, 4, 6, 1]])

layer_norm = nn.LayerNorm([3, 4],elementwise_affine=False)
norm = layer_norm(tensor)
print(norm)
print(torch.sum(norm[0]))
print(torch.sum(norm[0]))
print()