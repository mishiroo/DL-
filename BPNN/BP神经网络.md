# BP神经网络

[TOC]

## 背景知识

### 优化器

**思路：对下降路径和学习率变化规律进行改进**

以梯度下降为基础进行衍生

SGD: 在样本中抽取一部分(一个)作为损失函数期望

数学证明如果直接用梯度下降法性价比不高，且使用SGD在凸问题上存在解，造成的速度下降也可以接受。

另一个方面：

梯度下降路径虽然是最快的方向，但实际情况，点是离散的，所以计算梯度并非完全准确。

我们想到可以降低学习步长进行缓解，但存在大幅增加计算量的问题。

牛顿法考虑使用抛物线作为切线拟合替代 

![image-20230103101943210](C:\Users\YLS\AppData\Roaming\Typora\typora-user-images\image-20230103101943210.png)



我们自然联想到是否可以对每个分量进行研究

比如下面的情况使用梯度下降法，竖直方向分量变化比较剧烈，但水平分量总体向前

所以效果不好。

动量法：构造指数加权 来实现对于过去梯度的考虑（数学上称为指数加权移动平均法）

![image-20230103102843773](C:\Users\YLS\AppData\Roaming\Typora\typora-user-images\image-20230103102843773.png)

Nesterov算法

![image-20230103103956820](C:\Users\YLS\AppData\Roaming\Typora\typora-user-images\image-20230103103956820.png)



从学习率数值的角度来看，学习率不应该是一个不变量

我们提出一种自适应的学习率变化算法 AdaGrad

但是会在平台期移动的慢，所以再次使用指数加权移动平均法-->RMSprop方法

RMSprop方法+动量法=Adam法(很常见)

也有其他的变种



### 损失函数之CrossEntropyLoss

这是深度学习我们常见的一个损失函数

[损失函数：交叉熵详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/115277553)

**功能**：判断实际输出与期望输出的接近程度

**交叉熵**主要刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近。假设概率分布p为期望输出，概率分布q为实际输出， H(p,q) 为交叉熵

## 原理介绍

 BP(Back Propagation)神经网络是一种按误差逆传播算法训练的多层前馈网络，它的学习规则是使用梯度下降法，通过反向传播来不断调整网络的权值和阈值，使网络的误差平方和最小。BP神经网络模型拓扑结构包括输入层(input)、隐层(hiddenlayer)和输出层(output layer)。BP网络的学习过程，由信息的正向传播和误差的反向传播两个过程组成。



## 实现步骤

### 技术细节

在使用pytorch的dataloader载入MNIST数据后

数据格式为[batch-size, 1, 28, 28]的Tensor

所以需要把批数据进行格式变化

最简单的情况是：传入一个28×28的Tensor,将它转为784×1的Tensor

eg: batch-size=64隐藏层只有1层 100个神经元

64×784->64×100->64×10

根据矩阵乘法推断出
hidden: in=784 out=100

output: in=100 out=10

### 网络模板

```python
import torch
import torch.nn as nn


# 训练数据集
x_train = torch.tensor([1], dtype=torch.float)
y_train = torch.tensor([1], dtype=torch.float)


# 定义网络模型
class Model_name(nn.Module):
    def __init__(self):
        super(Model_name, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        # other network layers

    def forward(self, x):
        x = self.conv1(x)
        # others
        return x


model = Model_name()  # 模型实例化

loss_func = torch.nn.MSELoss()  # 定义损失函数

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 定义优化器


num_iter = 100  # 定义最大迭代轮数
for step in range(num_iter):
    # forward
    prediction = model(x_train)            # 前向传播
    loss = loss_func(prediction, y_train)  # 计算损失函数值

    # backward
    optimizer.zero_grad()  # 每次做反向传播前都要将梯度清零，否则梯度会一直累加
    loss.backward()        # 自动微分求梯度
    optimizer.step()       # 梯度反向传播，更新参数
    

```



## 应用案例

采用鸢尾花数据集进行分类

MNIST数据集识别数字

## 代码实现

 [code.ipynb](code.ipynb) 
