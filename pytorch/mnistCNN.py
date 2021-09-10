import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data

EPOCH = 5  # 训练整批数据多少次
BATCH_SIZE = 64  # 每次批数的数据量
LR = 0.001  # 学习率，学习率的设置直接影响着神经网络的训练效果

train_data = torchvision.datasets.MNIST(  # 训练数据
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
test_data = torchvision.datasets.MNIST(
    root='./test',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
# 批量加载
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# 数据可视化
images, label = next(iter(train_loader))
images_example = torchvision.utils.make_grid(images)
images_example = images_example.numpy().transpose(1, 2, 0)  # 将图像的通道值置换到最后的维度，符合图像的格式
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
images_example = images_example * std + mean
plt.imshow(images_example)
plt.show()

image_array, _ = train_data[0]  # 把一个批数的训练数据的第一个取出
image_array = image_array.reshape(28, 28)  # 转换成28*28的矩阵
plt.imshow(image_array)
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height输入
                out_channels=16,  # n_filters输出
                kernel_size=5,  # filter size滤波核大小
                stride=1,  # filter movement/step步长
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化, padding=，(kernel_size-1)/2 当 stride=1填充
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN().cuda()
print(cnn)  # 显示神经网络
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # Adam优化函数
loss_func = nn.CrossEntropyLoss()  # 损失函数（损失函数分很多种，CrossEntropyLoss适用于作为多分类问题的损失函数）

# training and testing
for epoch in range(EPOCH):  # 训练批数
    for step, (x, y) in enumerate(train_loader):  # 每个批数的批量
        b_x = x.cuda()  # batch x
        b_y = y.cuda()  # batch y
        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        print('epoch: %s   step: %s   loss: %s' % (epoch, step, loss))
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 损失函数的反向传导
        optimizer.step()  # 对神经网络中的参数进行更新

# 在测试集上测试，并计算准确率
print('**********************开始测试************************')

for step, (x, y) in enumerate(test_loader):
    test_x, test_y = x.cuda(), y.cuda()
    test_output = cnn(test_x)

    # 以下三行为pytorch标准计算准确率的方法，十分推荐，简洁明了易操作
    pred_y = torch.max(test_output.cpu(), 1)[1].numpy()
    label_y = test_y.cpu().numpy()
    accuracy = (pred_y == label_y).sum() / len(label_y)

print(test_output)  # 查看一下预测输出值
print('acc: ', accuracy)
