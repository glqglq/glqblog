---
    author: LuckyGong
    comments: true
    date: 2019-01-15 20:27
    layout: post
    title: Papper-DeepFM A Factorization-Machine based Neural Network for CTR Prediction
    categories:
    - ad
    tags:
    - ad
    - papper
---

# 0.Abstract

- 现有的方法对低阶或高阶的交互有很强的偏向，有的方法需要专业特征工程。本文中我们提出兼顾低阶和高阶特征的端到端学习模型。
- 与Google最新的Wide＆Deep模型相比，DeepFM对其“宽”和“深”部分有共享输入，除不需要特征工程。
- 证明DeepFM相对于基准数据和商业数据的CTR预测的现有模型的有效性和效率。

# 1.Introduction

- 通常，用户点击行为背后的特征交互是高度复杂的，其中低阶和高阶特征交互应该发挥重要作用。根据Google的Wide＆Deep模型[Cheng et al，2016]，考虑到低阶和高阶特征交互同时带来了额外的改进，而不是单独考虑二者其一。
- 然原则上FM可以模拟高阶特征交互，但实际上通常由于高复杂性而仅考虑2阶特征交互。
- 作为学习特征表示的有效方法，深度神经网络具有学习复杂特征交互的能力。一些想法到CNN和RNN用于CTR预测，但基于CNN的模型偏向于相邻特征之间的相互作用，而基于RNN的模型更适合具有顺序依赖的点击数据。 
- FNN：基于FM的NN，在应用DNN前先训练FM，因此受到FM限制。
- PNN：通过在embedding层和全连接层之间引入乘积（product）层。
- PNN和FNN与其他深度模型一样，捕获很少的低阶相互作用特征，但低阶相互作用特征对于CTR预测也是必不可少的。

- Wide&Deep：在该模型中，“宽部分”和“深部分”需要两个不同的输入，“宽部分”的输入仍然依赖于专业特征工程。

- DFM：
  - 集成了FM和DNN，模拟了FM等低阶特征交互和DNN等高阶特征交互。
  - 可以在没有任何特征工程的情况下进行端到端训练。
  - wide部分和deep部分共享相同的输入和embedding。

# 2.Our Approach

## 2.1符号定义

- n：样本数
- m：firld数
- χ ：输入
- y∈{0,1}：输出
- xfieldj：第j个field的变量

## 2.2模型

- 由两个组件构成：FM组件、deep组件。共享相同的输入。
- 对于特征i：wi用于衡量其1阶重要性，潜在矢量vi用于衡量特征i与其他特征相互作用的影响。Vi以FM组件的形式输入以模拟order-2特征交互，并以deep组件的形式输入以模拟高阶特征交互。
- FM组件和DEEP组件共享相同的embedding特征，这有两点好处：
  - 同时学习低阶、高阶特征交互
  - 不需要专门的特征工程
- 所有参数，包括wi、vi等，都是针对组合预测模型联合训练的。

- 公式：y=sigmoid(yFM+yDNN)

- 图示：

  ![](http://5b0988e595225.cdn.sohucs.com/images/20180904/bf5749e152b14794b4f6fe2d529b081d.jpeg)

### 2.2.1FM组件

- 学习低阶特征交互

- 就是个FM模型

- 图示：

  ![](http://5b0988e595225.cdn.sohucs.com/images/20180904/ad2980da927d4cb78ae482f9ea5531fa.jpeg)

### 2.2.2Deep组件

- 学习高阶特征交互
- 与图像、音频等数据作为输入的连续密集神经网络相比，CTR预测的输入完全不同，需要设计新网络架构。
- 输入特点：高度稀疏、超高维度、分类-连续特征混合、根据field分组。所以要有个embedding层将数据转换为：低维、密集、实值向量，否则网络难以训练。
- embedding层：
  - 输入feild向量长度不同，embedding层参数却都有相同的尺寸k。
  - FM的隐向量V现在作为权重用，被学习并用于将输入field压缩到embedding。
  - 我们不再需要通过FM进行预训练，而是以端到端的方式联合训练整个网络。 
  - 输出尺寸为m，输出为：a(0)=[e1,...,em]。ei是第i个field的embedding，m是field数。

- 图示：

  

![](http://5b0988e595225.cdn.sohucs.com/images/20180904/6a109d64aa8948f38d5c38ef968d4867.jpeg)

## 2.2与其他CTR NN关系

- FNN：
  - 简介：FM初始化的前馈神经网络
  - 局限：embedding参数可能受到FM影响、预训练阶段引入开销降低了效率、仅能捕获高阶特征交互。
- PNN：
  - 简介：embedding层和第一个隐藏层之间加product层。
  - 变体：IPNN、OPNN、PNN
  - 局限：仅能捕获高阶特征交互。
- Wide&Deep：
  - 局限：wide需要特征工程
  - 拓展：用FM替换LR

# 3.实验



# 5.结论

- 优点：
  - 不需要pretrain
  - 能学习到高阶、低阶交叉
  - 提出了一种特征embedding共享策略来避免特征工程。