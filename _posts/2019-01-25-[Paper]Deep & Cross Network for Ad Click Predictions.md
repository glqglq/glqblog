---
    author: LuckyGong
    comments: true
    date: 2019-01-25 20:27
    layout: post
    title: Papper-Deep & Cross Network for Ad Click Predictions
    categories:
    - nlp
    tags:
    - nlp
    - papper
---

# 0.Abstract

- 提出了深度&交叉网络（DCN），引入了一种新颖的交叉网络，可以更有效学习某些有界度的（bounded-degree）特征交互。
- DCN在每一层应用特征交叉，不需要手动特征工程，为DNN模型增加了可忽略的额外复杂度。

- 在CTR预测数据集和密集分类数据集上劲都和内存的优越性。

# 1.Introduction

- CPC模型
- 系数数据对特征探索很有挑战性。
- lr简单易懂、易于拓展；然而表现力有限、需要手动特征工程。

- DCN：
  - 由多个层组成，最高程度的交互可由层深度确定。
  - 每个层基于现有层生成高阶交互，并保持先前层的交互。
  - DNN可以捕获交互特征，但是需要大量的参数。
  - 将cross network和DNN联合训练，可以有效捕获交互特征。

## 1.1相关工作

- FM
- FFM
- DNN
- Deep Crossing 通过堆叠所有类型的输入来扩展Resnet并实现自动特征学习。
- Wide&Deep：交叉特征作为线性模型的输入，并将线性模型与DNN模型联合训练。

## 1.2主要贡献

- DCN支持使用稀疏和稠密输入进行自动特征学习，有效地捕捉特征交互，学习高度非线性交互，不需要手工特征工程。
  - 简单而有效。
  - 参数比DNN少了近一个数量级，内存耗费不多，易于实现。
  - logloss表现比DNN好。

# 2.DCN

图示：

![](https://img2018.cnblogs.com/blog/784062/201811/784062-20181107204830366-1613845981.png)

- 损失函数：
  $$
  loss = -\frac{1}{N} \sum_{i=1}^N y_i log(p_i) + (1-y_i)log(1-p_i) + \lambda \sum_{l} ||w_l||^2
  $$

## 2.1Embedding和Stacking层

- 稀疏特征onehot表示耗空间，索引引入embedding层：

- 公式：

  - xembed,i：embedding向量
  - xi：第i个特征的输入
  - Wembed,i：embedding矩阵
  - ne：embedding size
  - nv：词向量size

  $$
  x_{embed,i}=W_{embed,i}x_i
  $$

## 2.2Cross Network

- 图示：

  ![](https://img2018.cnblogs.com/blog/784062/201811/784062-20181109154440806-596027424.png)

- 公式：

  - xl、xl+1：l交叉层到l+1交叉层的列向量
  - wl：第l层权重
  - bl：第l层偏置

  $$
  X_{l+1} = X_0X_l^TW_l+B_l+X_l=f(X_l,W_l,B_l)+X_l
  $$

- 复杂度分析：Lc表示交叉网络的层数，d表示输入向量的维度，则交叉网络需要的参数为d×Lc×2，乘以2是因为每一层有两个长度为d的参数W和B，从而交叉网络的时空复杂度为O(d)，所以交叉网络相对于DNN引入的复杂度是微乎其微的。得益于X0XTl的一阶性质，使得我们无需计算和存储整个矩阵就能够高效的生成所有交叉项。

## 2.3Deep Network

- RELU激活

- 复杂度分析：为了简单起见假设每个隐藏层的神经元数目相同，Ld表示deep网络的层数，m表示deep层的尺寸。第一层需要的参数量为d×m+m，剩余层需要的参数量为(m2+m)×(Ld−1)。deep层的参数量：

  d × m + m + (m2 + m) × (Ld - 1) 

## 2.4Combination Layer

- 把两个网络的输出concate起来，过一个逻辑斯蒂函数。

- 公式：
  $$
  p=\sigma ([X_{L_d} ^ T,H_{L_m}^T]\cdot W_{logits} + B_{logits})
  $$

# 3.理论分析

- 多项式近似：根据Weierstrass逼近定理，在特定平滑假设下任意函数都可以被一个多项式以任意的精度逼近，所以可以从多项式近似的角度分析交叉网络。dd元nn阶多项式参数量为O(d^n)，交叉网络只需要O(d)参数量就可以生成相同阶数多项式中出现的所有交叉项。

- FM的泛化：交叉网络借鉴了FM共享参数的思想并将它扩展至更深的结构。FM模型中每个特征x_i都有一个相关的权重向量v_i，交叉项x_ix_j的权重通过<v_i,v_j>计算得到。DCN中每个特征x_i都对应一个标量集{w^(i)_k}^l_k=0，也就是每个交叉层权重向量W的第i分量组成的集合，这样交叉项x_ix_j的权重通过{w^(i) _k}lk=0和{w^(j) _k}lk=0计算得到。两个模型中每个特征对应的参数都是独立学习的，交叉项的参数通过对应的特征参数计算得到。参数共享不仅使得模型更高效而且对没见过的组合特征具有更好的泛化能力，同时对噪声更健壮。比如xi和xj在训练数据中没有同时出现过，xixj对应的权重就无法学习到。FM是一个浅层结构，只能表示2价的特征组合。DCN能够学习高阶的特征组合，在特定阶数限制下能够构建所有的交叉项。而且同对FM的高阶扩展相比，DCN的参数量是输入向量维度的线性函数。

- 高效映射：每个交叉层都会创建X_0和X_l各元素之间的两两组合，生成d^2维度的向量，然后将该向量映射到d维的空间中。如果直接进行映射操作需要O(d^3)，而DCN提供了一种高效的映射方式只需要O(d)即可。考虑X_p=XX^TW，假设X和W都是2维列向量，如下所示上面公式是直接计算，下面公式是高效的计算法法:
  $$
  X_p = X \tilde X^T W = \begin{bmatrix} x_1 \\ x_2 \\ \end{bmatrix} [\tilde x_1, \tilde x_2] W = \begin{bmatrix}x_1 \tilde x_1 & x_1 \tilde x_2 \\ x_2 \tilde x_1 & x_2 \tilde x_2 \\ \end{bmatrix} W\\
  
  X_p ^T = [x_1 \tilde x_1,x_1 \tilde x_2, x_2 \tilde x_1, x_2 \tilde x_2]  \begin{bmatrix} W & 0 \\ 0 & W \\ \end{bmatrix}\\
  
  W=\begin {bmatrix} w_1 & 0 \\ w_2 & 0 \\ 0 & w_1 \\ 0 & w_2 \\ \end{bmatrix}
  $$
  

# 5.结论和未来的方向

