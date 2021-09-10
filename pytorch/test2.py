import torch
import numpy as np

print(torch.__version__)


a = torch.empty(3, 4)
b = torch.ones(3, 4)
c = torch.randint(3, 10, (3, 4))
print(b)
print(c)

d = np.arange(1)
print(d)

e = torch.tensor(np.arange(1,3))

print(e)

t1 = torch.Tensor([[[1]]])
t2 = torch.tensor([[[1]]])
print(t1.type())
print(t2.type())
print(t1.item())


a = torch.randn(2,2)
print(a)
a = ((a * 3) / (a - 1))
print(a)