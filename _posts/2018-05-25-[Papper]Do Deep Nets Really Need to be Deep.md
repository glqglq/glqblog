---
    author: LuckyGong
    comments: true
    date: 2018-05-25 20:27
    layout: post
    title: Papper-Do Deep Nets Really Need to be Deep
    categories:
    - modelcompression
    tags:
    - modelcompression
    - papper
---

# Abstract

- 本文内容：

  - 凭经验证明浅前馈网络可以学习深网所学的复杂函数，并达到以前只有深模型才能实现的精度。
  - 在某些情况下，浅层神经网络可以使用与原始深层模型相同数量的参数来学习这些深层功能。

  

# 1.Introduction

- Baseline：1M训练集，86%->91%，可能原因：
  - a）深层网络有更多的参数; 
  - b）在给定相同参数的情况下，深网络可以学习更复杂的功能; 
  - c）深层网络有更好的bias并学习到更有趣/有用的function（例如，因为深层网络更深入，它学习了分层表示）; 
  - d）没有卷积的网络不容易学习卷积网络可以学习的网络; 
  - e）当前学习算法和正则化方法在深层架构下比浅层架构更好地工作; 
  - f）以上全部或部分内容; 
  - g）以上都不是？
- It has been shown that deep nets coupled with unsupervised layer-by-layer pre-training technique[10]/[19] work well. 
- 在[8]中，作者表明深度与预训练相结合为模型权重提供了良好的先验，从而提高了泛化能力。
- 在[4]中，作者证明神经网络宽度足够宽的话，一层网络也可以逼近任何决策边界
- 但是经验说明：浅层网络很难被训练和深层网络的精度一致。
  - 对于cv：[7]表明cnn在参数量相同条件下，越深越好。在[5]：作者通过sift特征训练浅层网络，对imagenet分类，结果表明用浅层网络学习的话，很难学到复杂的模型。
  - 对于语音识别：更深层模型比浅层模型更好。
- 本文工作：
  - 根据经验证据得出：浅层网络能学习与深层网络相同的function，并且在某些情况下具有与深层网络相同数量的参数。
  - 实验：用浅层模型模拟深层模型

# 2.Training Shallow Nets to Mimic Deep Nets

## 2.1Model Compression

- [3]中，小神经网络模型包含的参数小了1000倍，但还是很精确。
- 模型压缩原理：将未标记的数据送入大模型，收集模型生成的分数（概率），用来训练小模型。小模型不是在原始onehot label上训练，小模型从大模型学出来的function中训练。如果小模型完美拟合大模型，则它会和复杂模型有相同的预测结果和误差。
- 通常，小模型不可能像大模型一样精确。

## 2.2Mimic Learning via Regressing Logit with L2 Loss

- 大模型用softmax输出、用交叉熵损失函数。小模型由183个loits训练（183个类，在softmax前），由于logits捕获了概率之间的对数关系，这些关系在概率空间中并不明显，但由教师模型学到，因而直接学logits更加容易。
- 为什么学logits而不学softmax：logits中可能是[10,20,30]，信息更加丰富；softmax输出就可能是：[0.0000001,0.000009,0.999999]，模型会更加注意0.999999这一项。
- 损失函数：回归损失函数，标准误差反向传播+SDG+动量
  - W：input到隐藏层的权重
  - β：隐藏层到输出层的权重

$$
L(W,\beta)=\frac{1}{2T}\sum{}_{t=1}^N||g(x^{(t)};W,\beta)-z^{(t)}||_2^2
$$

-	也尝试了其他损失函数，如：KL散度、概率的L2损失，但是还是平方误差损失最优。
		对logits进行归一化（减均值、除标准差）教师模型可以获得更好的L2损失结果，但是学生模型没关系。

## 2.3Speeding-up Mimic Learning by Introducing a Linear Layer

-	浅层网络的一层需要更多非线性隐藏单元（大权重矩阵），由于这其中有很多相关的参数，浅层网络将大部分计算花费在输入数据向量和大权重矩阵的矩阵乘法，所以sgd收敛缓慢。
		在输入和非线性隐藏层间加入瓶颈线性层（k个线性隐藏单元）可以加速学习。将权重矩阵W分解为U和V，U和V可以被线性层反向传播所学习。

$$
L(U,V,\beta)=\frac{1}{2T}\sum{}_{t=1}^N||\beta f(UVx^{(t)})-z^{(t)}||_2^2
$$

- 这个操作不仅可以加速学习，还可以将空间从O(HD)降到O(k(H+D))，其中H和D是W的长和宽。

