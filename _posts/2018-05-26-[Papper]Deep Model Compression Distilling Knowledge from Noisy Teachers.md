---
    author: LuckyGong
    comments: true
    date: 2018-05-25 20:27
    layout: post
    title: Papper-Deep Model Compression Distilling Knowledge from Noisy Teachers
    categories:
    - modelcompression
    tags:
    - modelcompression
    - papper
---

# Abstract

- 我们扩展了深层模型压缩的师生框架，具有解决运行时间和训练时间复杂度的潜力。
- 从教师网络迁移到学生网络时，给学生网络加入了一个噪音正则项
- 在cifar10、svhn、mnist上都有改进，在cifar10表现最好。

# 1.Introduction

- 往常的方法主要是减少模型的存储空间，但是模型必须在运行时解压缩，因此移动设备上的可部署性问题依然存在。
- 主要贡献：
  - 减少空间占用、预测时间
  - 从单一的教师中提炼知识可能会受到限制，提出一种简单的基于噪声的正则化方法来模拟来自多位教师的学习，以获得更好的深度模型压缩
  - MNIST，SVHN和CIFAR-10上验证方法

# 2.Background and Related Work

- 略
- 参数共享和矩阵分解方法都只关注深层模型的存储复杂性，但它们无法改善运行时（或训练时间）的复杂性。
- 网络修剪方法在存储复杂性任务上表现出非常好的性能，但并不是旨在减少运行时（或训练时间）的复杂性。
- 师生方法在实现所有复杂性压缩的这个方向上显示出很好的前景，但自提出以来没有太多的后续工作。

# 3.Proposed Methodology    

## 3.1Teacher-Student Learning

- 教师输出中存在的“黑暗知识”作为学生模式的强大目标和规则，因为它提供了共享有用信息的软目标。
- 由于软目标有助于训练，所以收敛速度通常比仅使用原始0/1硬标签要快。
- 少量的训练数据通常足以训练学生网络。

# 3.2Student Learning using Logit Regression    

- 训练logits
- 损失函数：其中T是 mini batchsize

$$
L(x,z,\theta)=\frac{1}{2T}\sum_i||g(x^{(i)};\theta) - z^{(i)}||_2^2
$$

## 3.3Noisy Teachers: Student Learning using Logit Perturbation

- 可以看作模拟从多个教师网络中进行学习，也可以看作加入正则项，称之为逻辑扰动
- logits公式：其中ϵ 是均值为0，方差为σ的高斯噪声
  - σ决定了教师原始logit值z(i)的扰动量，越高扰动越高，但是这不必要施加在所有的样本上，我们用固定概率α从小批量样本中选择样本后再使用下式计算（有的加噪声，有的没加）。

$$
z^{'(i)}=(1+\epsilon)z^{(i)}
$$

- 损失函数：

$$
L(x,z^{'},\theta)=\frac{1}{2T}\sum_i||g(x^{(i)};\theta) - z^{'(i)}||_2^2
$$

- 优化方法：SGD

## 3.4 Equivalence to Noise-Based Regularization

- Bishop在文献[2]中指出，在损失函数中增加一个L2正则化项相当于在输入数据中添加高斯噪声。
- 公式推导：见papper

# 5.Discussions and Analysis

