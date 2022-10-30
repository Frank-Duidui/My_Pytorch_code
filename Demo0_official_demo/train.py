import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    '''
    transform.ToTensor()
    1. 是将输入的数据shape H，W，C ——> C，H，W   2. 将所有数除以255，将数据归一化到【0，1】
    后面的transform.Normalize()则把0-1变换到(-1,1). x = (x - mean) / std
    即同一纬度的数据减去这一维度的平均值，再除以标准差，将归一化后的数据变换到【-1,1】之间
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    '''
    batch_size : 一次处理36张照片
    shuffle：是否打乱数据集
    '''
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                               shuffle=True, num_workers=0)
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader) #生成可迭代的迭代器
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() #实例化模型
    loss_function = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001) # 优化器

    # 训练集迭代5次
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0 #累加损失
        for step, data in enumerate(train_loader, start=0): # 遍历训练集样本
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data # data包含输入的图像和对应的标签

            # zero the parameter gradients
            optimizer.zero_grad() # 历史损失梯度清零
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels) # 通过网络预测的输出值与真实标签值计算损失
            loss.backward() # 方向传播
            '''优化器optimizer的作用 优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数值的作用 '''
            optimizer.step() # 进行参数的更新

            # print statistics
            running_loss += loss.item() # 将损失累加起来    使用loss.item()直接获得所对应的python数据类型
            if step % 500 == 499:    # print every 500 mini-batches 每隔500步打印一次信息
                '''在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。
                    而对于tensor的计算操作，默认是要进行计算图的构建的，
                    在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建。
                '''
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]  # 输出网络预测最大值的索引  [1] 代表索引
                    # 将预测的值与真实值进行对比，相同为1，不同为0.  通过求和算出预测正确的样本 除以/  总样本数 = 正确率
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth' # 保存模型路径
    torch.save(net.state_dict(), save_path)  # 保存模型所有参数


if __name__ == '__main__':
    main()
