import torch
import numpy as np
import random
from matplotlib import pyplot as plt

# y = 3x +1.2

x = torch.rand([500, 1])

y_true  = x * 3 + 1.2
y_true += 0.01 * torch.rand([500, 1]) * x


w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
leaning_rate = 0.008


for i in range(5000):
    y_hat = torch.matmul(x, w) + b
    loss = (y_hat - y_true).pow(2).mean()
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()
    w.data = w.data - leaning_rate * w.grad
    b.data = b.data - leaning_rate * b.grad

    if i % 100 == 0:
        print("w, b, loss", w.item(), b.item(), loss.item())


# plt.figure(figsize=(20, 8))
plt.scatter(x.numpy().reshape(-1), y_true.numpy().reshape(-1))
y_hat = torch.matmul(x, w) + b
plt.plot(x.numpy().reshape(-1), y_hat.detach().numpy().reshape(-1), c='r')
plt.show()