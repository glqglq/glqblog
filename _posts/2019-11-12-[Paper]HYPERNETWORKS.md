---
    author: LuckyGong
    comments: true
    date: 2019-11-12 20:27
    layout: post
    title: Paper-HYPERNETWORKS
    categories:
    - aml
    tags:
    - aml
    - metalearning



---

# 1.Introduction

- Hypernetwork：small，生成main network的weights，输入是：main network的weights相关的结构信息。
- main network：large

# 2.Movitation and Related Work

- 演化计算（evolutionary computing）：很难直接在大搜索空间上跑。
- Schmidhuber（1992）：提出快速权值。一个网络可以为另一个网络产生上下文相关的权值变化。
- Schmidhuber（1993）：快速权值的递归版本。
- Gomez&Schmidhuber（2005）：快速权值的实际应用，进化学。习
- HyperNEAT（2009）：
  - 搜索空间约束在很小的权重空间内。
  - 输入是一堆main network中每个权重的虚拟坐标（virtual coordinates）。
  - CPPNs用来学习main network的权重结构。
- Compressed Weight Search（2010）：
  - HyperNEAT的变种，结构没变
  - 权重通过Discrete Cosine Transform来进化。
- Denil et al., 2013
- Yang et al., 2015
- Bertinetto et al., 2016
- De Brabandere et al., 2016
- Jaderberg et al., 2016
- Andrychowicz et al., 2016
- Differentiable Pattern Producing Networks (DPPNs)(Fernando et al., 2016)
  - HyperNEAT的变种，结构变了。
- ACDC-Networks(Moczulski et al., 2015)
  - 线性层用DCT压缩
- 缺点：训练慢，需要启发式方法。离散余弦变换在压缩权值搜索中的应用过于简单，使用DCT先验可能不适用于很多问题。HyperNEAT中结构和权值进化对于大多数问题都是多余的。
- 本文：
  - 优点：端到端训练、模型灵活和训练简单取得良好平衡。rnn也能用。



# 3.Methods

- input：各个层的权重的嵌入向量（每个层一个embedding），可以是固定，也可以是端到端学习到的，生成可以是动态生成的。允许同层、跨层的近似权重分享。
- 权值共享：
  - rnn：可以被看做强权值跨层共享，使得不灵活、产生梯度消失。
  - cnn：不共享权值，灵活，网络深的时候参数冗余。
  - Hypernetwork：看做轻权重共享，取得了平衡，学习参数的数量大大低于main network。

## 3.1静态

- hepernetwork是两层线性网络：
  - hypernetwork第1层：输入main network第j层编码向量zj。其中：Nin个权重矩阵Wi∈Rd*Nz、偏置Bi∈Rd，d是第1层神经元数，固定d=Nz。
  - hypernetwork第2层：输入ai∈Rd，用Wi∈R fsize * Nout fsize * d、偏置B∈R fsize * Nout fsize。最终的kernel是由Kj们concate起来的。
- 参数量对比：
- 为什么用两层不用一层：一层网络的参数量更多。
- 不具有相同维度kernel：如残差网络=>如果我们需要一个更大的内核用于某一层，我们将把多个基本内核连接在一起，形成更大的内核。例：有16、31、64，则可以学单位为16的。尺寸较大的内核将需要按比例增加embedding向量的数量。

## 3.2动态

- hypernetwork是rnn：生成rnn，relaxe的权值共享（rnn的硬权值共享和cnn的无权值共享的折衷）。
  - 。。。。。



# 未完待续