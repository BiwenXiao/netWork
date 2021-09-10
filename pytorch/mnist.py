import torchvision
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


# mnist = MNIST(root="./data", train=True, download=False)
# testData = MNIST(root="./test", train=False, download=True)
# mnist[0][0].show()

# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081, ))])
#
# mnist = transform(mnist)
#
# train_dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
#
# print(train_dataloader)



def get_dataloader(batch_size = 128, train = True):
    transform_fn = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081, ))
    ])

    data_set = MNIST(root="./data", train=train, download=False, transform=transform_fn)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

class Mnist_NN(nn.Module):

    def __init__(self):
        super(Mnist_NN, self).__init__()
        self.cn1 = nn.Conv2d(1, 6, 5)
        self.cn2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*3*3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.cn1(x), (2, 2)))
        x = F.max_pool2d(F.relu(self.cn1(x), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


data_loader = get_dataloader()
model = Mnist_NN();
print(model)


optimizer = optim.SGD(model.parameters(), lr=0.005)
optimizer.zero_grad()
out = model(data_loader[1])

criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
loss.backward()
optimizer.step()
print(out)