---
    author: LuckyGong
    comments: true
    date: 2019-06-13 20:27
    layout: post
    title: Paper-Deep & Cross Network for Ad Click Predictions
    categories:
    - ad
    tags:
    - ad
    - paper
---



# Absract

- 特征工程困难、需要人工，有时需要穷举。
- DNN能自动学习特征交叉。但是其隐含地生成所有交互，并且不一定有效地学习所有类型的交叉特征。
- 本文：
  - 引入了一种新颖的交叉网络（cross network），可以更有效地学习某些有界度的（bounded-degree）特征交互。
  - DCN明确地在每一层应用特征交叉，不需要手动特征工程，并且为DNN模型增加了可忽略的额外复杂性。
  - 我们的实验结果证明了其在CTR预测数据集和密集分类数据集上的最新算法在模型精度和内存使用方面的优越性。

# 1.Introduction

- 探索频繁的特征、交叉特征是做出良好预测的关键。然而web推荐系统数据大多数是离散的，导致对于特征探索有挑战性，这将大多数模型限制为线性模型，如LR。
- 线性模型简单、可解释、易拓展，但是表现力有限。且不支持交叉特征。需要特征工程。
- 本文提出的网络是cross网络与DNN共同训练的交叉网络。
  - DNN捕捉特征之间的非常复杂的相互作用，但是与cross网络相比，其要更多的一个数量级的参数，无法明确形成交叉特征，可能无法有效学习特征交互。
  - cross网络由多个层组成，最高程度的交互可由层深度确定，每个层基于现有层生成高阶交互，并保持与先前的交互。

## 1.1Related Work

- FM：将稀疏特征投射到低维稠密向量上，学习向量内积的特征交互。大量参数会产生不希望的计算成本。
- FFM：每个向量与field相关联。
- DNN+embedding：由于embedding和激活函数，DNN能学习到非平凡（non-trivial）的高阶特征交互。
- 残差网络使得可以训练非常深的网络。
- Deep Crossing：通过堆叠所有类型的输入（all types of inputs）来拓展残差网络并实现自动特征学习。
- Wide&Deep：将交叉特征作为线性模型的输入，将线性模型与DNN联合训练。其效果取决于正确选择交叉特征，这是一个指数复杂度的问题。

## 1.2Main Contributions

- 提出了DCN，支持稀疏和稠密输入，自动特征学习。有效捕获bounded degree的特征交互，学习高度非线性相互作用，计算成本低。
  - 提出了新颖的cross网络，明确地用每层的特征交叉，有效地学习bounded degree的特征交互，不需要特征工程。
  - 网络简单有效。每层最高多项式degree（the highest polynomial degree）在每层增加，degree最终由层数决定。网络由degree的所有交叉项组成，直到最高层，每层系数都不同。
  - 内存使用效率高，易于实现。
- 实验结果表明：DCN的logloss比DNN更低，参数数量少了接近一个数量级。

# 2.DCN

## 2.1embedding与stacking层

- 输入：连续值、离散值。
- 将连续值和离散值的embedding给stack起来：

![](https://upload-images.jianshu.io/upload_images/4155986-dfc32419b13cac1a.png?)

## 2.2.Cross网络

- 交叉网络的核心思想是以有效的方式应用显式特征交叉。交叉网络由交叉层组成，每个层具有以下公式：

![](https://upload-images.jianshu.io/upload_images/4155986-feacc6f25a1a985d.png)

- 特征高维交叉：交叉网络的特殊结构使交叉特征的程度随着层深度的增加而增大。
- 参数量：
  - 如果用Lc表示交叉层数，d表示输入维度。参数的数量参与跨网络参数为：d * Lc * 2 (w和b)。
  - cross网络的时间和空间复杂度在输入维度上是线性的。 因此，与dnn相比，cross网络引入的复杂性可忽略不计，使DCN的总体复杂性保持在与传统DNN相同的水平。
  - This efficiency benefits from the rank-one  property of x0xTl ,which enables us to generate all cross terms without computing or storing the entire matrix.
- cross网络的参数少限制了模型容量。为了捕捉高阶非线性的相互作用，模型并行地引入了一个深度网络。

## 2.3Deep网络

- Ld：神经网络层数
- m：神经元数
- 参数量： d×m+m+(m +m)×(Ld−1) 。

## 2.4Combination层

- 将Deep网络和Cross网络的输出给concate起来，然后送入标准logits层（就是一个全连接层）。
- 公式：

![](https://upload-images.jianshu.io/upload_images/4155986-3b2e83dee702d12d.png)

## 2.5损失函数

- 加了正则项的logloss

![](https://upload-images.jianshu.io/upload_images/4155986-6a3cad235da5dd61.png)

# 3.Cross网络分析

- 假设bi=0
- 令第j层元素的权重为w_j^{i}，对于多下标（multi-index）：α=[α1,⋅⋅⋅,αd]∈N^d，x=[x1,…,xd]∈Rd，定义|α|=∑_i=1^{d}αi。将xα11xα22⋅⋅⋅xαdd交叉项定义为|α|，多项式定义为它的高阶项。

## 3.1多项式近似

- 由weierstrass逼近定理，平滑假设下任何函数的任何精度都可以通过多项式逼近。

- 我们从相同degree的多项式角度分析Cross网络。Cross网络是一种高效、富有表现力的方式，是比相同degree的多项式类更好地概括为真实世界数据集的方式。

- 用Pn(x)表示n度的多元多项式，如下：

  ![](https://www.wengbi.com/data/attachment/forum/201810/30/091734uvv9eeqo4z55dkf8.jpg)

- 每个多项式在这类上都有O(dn)的参数、交叉网络包含所有的交叉项发生在多项式相同的程度,每一项的系数彼此不同。

- 定理：一个l层交叉网络，i+1层可以定义为xi+1=x0xTiwi+xi。网络的输入为：*x*0=[x1,x2,⋯,xd]T，输出为：gl(x0)=xlTwl。那么多元多项式gl(x0)产生以下情形：

  ![](https://www.wengbi.com/data/attachment/forum/201810/30/091735nw5z0hkqgtokzkuw.jpg)

## 3.2FM的泛化

- Cross网络利用了FM的参数共享思想，并进一步拓展到更深层的结构。
- 在DCN中，xi与{w_k^{i}}_{k=0}^l有关，xixj的权重是{w_k^{i}}_{k=0}^l、{w_k^{j}}_{k=0}^l。
- 两个模型都具有独立于其他特征的每个特征学习的一些参数，并且交叉项的权重是相应参数的特定组合。
- 参数共享不仅使模型更有效，而且使模型能够推广到看不见的特征交互，并且对噪声更具鲁棒性。
- FM是隐层结构且是degree2以内，DCNdegree是无限的，将单层拓展到多层和高度交叉。
- 与高阶FM不同，DCN的参数量只随输入维数线性增长。

## 3.3有效的映射

- 每个cross layer以有效的方式将x0和xl之间的所有的成对交叉（pairwise interaction）退回到输入的维度。

- 如果将cross layer的输入x构建pairwise interaction为xixj，然后映射回d维，最简单的方法也是n3的复杂度。

- cross layer是这么做的 ：x'=xixjw，这就等价于：

  ![](https://img-blog.csdn.net/20181017202658694?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JvZ3Vlc2ly/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)