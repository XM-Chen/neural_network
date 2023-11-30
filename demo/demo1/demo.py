import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
# from self_model import *

#定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1、准备数据集
train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(), download=False)


train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

print(train_dataset_size)
print(test_dataset_size)
#2、加载数据集
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=64)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=64)





#3、搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model1(x)
        return x




#4、创建网络模型
net = Net()
net = net.to(device)



#5、设置损失函数、优化器
#损失函数
loss_fun = nn.CrossEntropyLoss()   #交叉熵
loss_fun = loss_fun.to(device)


# 优化器
learn_rate = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)



#6、设置网络训练中的一些参数
total_train_step = 0   #记录总计训练次数
total_test_step = 0    #记录总计测试次数
epoch = 20    #设计训练轮数


#添加tensorboard
writer = SummaryWriter("./logs_self_train")

#7、开始进行训练
for i in range(epoch):
    print("---第{}轮训练开始---".format(i+1))

    net.train()     #开始训练，不是必须的，在网络中有BN，dropout时需要
    for data in train_dataset_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)


        #比较输出与真实值，计算Loss
        loss = loss_fun(outputs, targets)

        #反向传播，调整参数
        optimizer.zero_grad()    #每次让梯度重置
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print("---第{}次训练结束, Loss:{})".format(total_train_step, loss.item()))

    writer.add_scalar("train_loss", loss.item(), total_train_step)






    #8、开始进行测试,测试不需要进行反向传播
    net.eval()   #开始测试，不是必须的，在网络中有BN，dropout时需要
    with torch.no_grad():    #这句表示测试不需要进行反向传播，即不需要梯度变化【可以不加】
        total_test_loss = 0   #测试损失
        total_test_accuracy = 0  #测试集准确率
        for data in test_dataset_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)

            #计算测试损失
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy
    print("第{}轮测试的总损失为：{}".format(i+1, total_test_loss))
    print("第{}轮测试的准确率为：{}".format(i+1, total_test_accuracy/test_dataset_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_test_accuracy/test_dataset_size, total_test_step)


    total_test_step += 1




    #9、保存模型
    torch.save(net, './self_model_{}.pth'.format(i+1))
    print('模型已保存')