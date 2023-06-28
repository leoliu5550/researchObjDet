import torch


a = torch.arange(1,13)
b = torch.reshape(a,(1,1,1,12))
print(a)
print(b)
print(b.shape)

c = torch.reshape(b,(1,1,4,3))
print(c)
print(c.shape)