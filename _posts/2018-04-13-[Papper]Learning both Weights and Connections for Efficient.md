---
    author: LuckyGong
    comments: true
    date: 2018-03-29 20:27
    layout: post
    title: [Papper]Learning both Weights and Connections for Efficient
    categories:
    - modelcompression
    tags:
    - papper
    - modelcompression
    - prune
---

# 1.摘要

- 目的：减少存储量和计算量而不影响准确性，使网络更适合在移动设备上运行；同时可以自动学习网络结构（即剪枝）；也可以防止过拟合。

- 方法：简单地说是只学习重要的连接，修剪冗余的连接减少模型参数

- 步骤：

  - train，学习哪些连接是重要的，不需要学习权重最终值。
  - prune，剪去重要性低于阈值的不重要连接 ，将密集的全连接转换为稀疏连接网络。
    - 阈值确定：首先这个阈值相关于这一层权重的标准差，同时每一层的阈值的确定也要参考相应层对裁剪的敏感度。
  - retrain，finetune重训练网络微调剩下的连接权重，以便其余连接可以补偿删除连接。
  - prune和retrain可以反复迭代进行

- 生物学意义：就像在哺乳动物的大脑中一样，在儿童发育的前几个月里，突触是在这里形成的，然后逐渐修剪少量使用的连接，成为一个“人”。

- 结果：

  - AlexNet减少模型参数量61million->6.7million，没有精度损失 
  - VGG-16减少模型参数量138million->10.3million，没有精度损失

  ​

# 2.相关工作

- [Vanhoucke et al] 8-bit整数(vs 32-bit浮点) activations，用8位int型的activation代替32位float。
- [Denton et al] 寻找适当的低阶参数近似值并保持原始模型的1％以内的精度
- [Gong et al] 使用矢量量化（Deep Compress论文中介绍的），和剪枝可以同时使用
- [Network in Network和GoogleNet模型]使用global average pooling代替全连接层来减少参数，但在使用ImageNet的参数时，需要另外增加一个线性层。（这是在网络结构方面寻求参数缩减）
- [Optimal Brain Damage和Optimal Brain Surgeon]修剪网络以减少基于损失函数的Hessian的连接数量，并且表明这种修剪比基于权重的修剪（例如权重衰减）更准确。 但是二阶导数需要额外的计算。
- [Shi et al和Weinberger et al]HashNet，通过使用散列函数将连接权重随机分组为哈希桶来减少模型大小，同一个哈希桶中所有连接共享单个参数值，稀疏性使散列冲突最小化。和prune结合可能会得到更好的效果。

# 3.Trick

## 3.1Regularization

- 采用不同的regularization会影响prune和retrain的效果：
  - L1会将更多的参数转换成接近0，这在prune之后retrain之前有很好的准确率，但是留下的参数在retrain后效果不如L2
  - L2会获得更好的剪枝结果

## 3.2Dropout率调整

- 用于防止过拟合，两次训练过程均用到了。
- 在二次训练中，考虑到模型容量的变化，必须调整dropout率。
- 由于prun已经减少模型容量，所以retrain时dropout ratio应该调小一点。
- **dropout中，每个参数在训练期间概率性下降，特征检测器停止工作但在测试的时候会升回来。（？）**
- 有如下公式：
  - Ci：第i层的连接数
  - Cio：原神经网络
  - Cir：重新训练后的神经网络
  - Ni：第i层神经元数目
  - D0：原始dropout ratio
  - Dr：再训练期间的dropout ratio

$$
C_i=N_iN_{i-1} \\
D_r = D_o\sqrt{\frac{C_{ir}}{C_{io}}}
$$

- **由于dropout对神经元起作用，且Ci与Ni是二次变化关系，所以根据公式1，修剪参数后的dropout应遵循公式2**

## 3.3局部修剪和参数协调

- 在第二次训练过程中，最好保留prun后存活的连接的初始训练阶段权重，而不是重新初始化修剪过的layer。
- 原因：
  - **CNNs contain fragile co-adapted features: gradient descent is able to find a good solution when the network is initially trained, but not after re-initializing some layers and retraining them.**
  - **第二次训练过程中降低计算量：因为反向传播不需要通过整个网络。**
- 问题：随着网络变深，容易出现梯度消失，使得修剪错误难以恢复。
- **解决：只训练prune后shallow layer保存下来的params。**

## 3.4迭代修剪

- 反复prune-retrain、prune-retrain步骤。
- 每次迭代都是一次贪心搜索。

## 3.5修剪神经元

- 在修剪连接之后进行。
- 一些0输入或者0输出的神经元也被prune。
- 原因：retrain阶段到达死神经元将具有零输入连接和零输出连接的结果。 这是由于梯度下降和正则化而发生的。