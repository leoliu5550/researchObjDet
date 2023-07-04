import torch
d_model = 6
a = torch.arange(1,25)
[batch_szie,row_data,data]

a = a.reshape([1,1,4,d_model])
print(a.size())
print(a)

head = 2
length = int(d_model/ head)
print(length)

b = a
b = torch.reshape(b,(1,1,4,2,length))
print(b.shape)
print(b)
