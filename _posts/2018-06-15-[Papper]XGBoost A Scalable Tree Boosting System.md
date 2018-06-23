---
    author: LuckyGong
    comments: true
    date: 2018-06-15 20:27
    layout: post
    title: Papper-XGBoost A Scalable Tree Boosting System
    categories:
    - prml
    tags:
    - prml
    - papper
---



# Abstract

- xgb是可拓展的端到端的提升树，对于稀疏数据进行稀疏感知，用分布式加权直方图算法（weighted quantile sketch）来近似树学习。提供缓存访问模式、数据压缩和共享。
- xgb可以使用比现有系统少得多的资源来拓展数十亿个样本。

# 1.Introduction

- 一些大数据分析的成功的两个原因：
  - 有效的统计模型的应用，获取复杂的数据依赖
  - 可脱产的学习系统，从大数据种学习模型
- 梯度提升树是一种在多个应用中大放异彩的技术，LambdaMART 是用于排序、ctr等场景的提升树变体。
- 比现有的框架快10倍以上，可拓展性高！这是来源于一些创新：
  - 一种新颖的处理稀疏数据的树学习算法
  - 一种分布式加权直方图算法，可以在近似树学习的过程中处理样本权重
  - 并行和分布式计算使得学习速度快。
  - 利用out-of-core计算，针对核外树学习提出了一种有效的缓存感知块结构。

# 2.TREE BOOSTING IN A NUTSHELL（回归树）

## 2.1模型

- 模型：用K个函数来进行预测，每个回归树每个叶子上有连续分数，我们使用wi表示第i个叶子上的分数，通过综合对应树叶的得分（由w给出）来计算最终预测。
  - xi是第i个样本的特征，有m个特征
  - yi是第i个样本的标签
  - F是回归树CART的函数空间
  - q是每个树的结构（决策规则），将一个样本的特征映射到（一些）叶子，并将wi求和。
  - T是树上叶子的数量
  - fk对应于一个树结构q和叶子权重们w
  - wi是第i个叶子节点的score（连续值）


![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_1.PNG)

![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_2.PNG)

## 2.2目标函数

- 对正则的objective有小幅改进，用了二阶方法。
- 目标函数1：
  - l是可微分的凸函数
  - Ω是正则项，用于平滑权重，包括：
    - 叶子节点数
    - 叶子权重

![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_3.PNG)

- 目标函数2：

  - 目标函数1包括函数作为参数，在欧式空间中不能使用传统的优化方法优化，使用加性方式训练。与adaboost相似，假设^ yi（t）是第t次迭代中第i个样本的预测，我们需要加上ft来最小化以下目标：

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_4.PNG)

  - 由于：

  $$
  \begin{split}
  f(x+\Delta x)\simeq f(x)+f'(x)\Delta x+\frac12f''(x)\Delta x^2
  \end{split}
  $$

  - 二阶近似可以用来快速优化目标，将上式进行二阶泰勒展开：
    - gi是一阶梯度
    - hi是二阶梯度

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_5.PNG)

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_6.PNG)

  - 去掉与待求参数无关的常数项，从而得到新的优化目标为：

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_7.PNG)

  - 将上式变形，将关于样本迭代转换为关于树的叶子节点迭代：
    - Ij={i|q(xi) = j}是叶子j的样本集合

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_8.PNG)

- 目标函数3：正规方程法求权重对应的最优值，该式可以用来衡量树的质量：

![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_10.PNG)

- 目标函数4：记分到左子树的样本集为IL,分到右子树的样本集为IR，则分裂该节点导致的损失减少值如下，我们希望找到一个属性以及其对应的大小，使得下式取值最大： 

![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_11.PNG)

## 2.3算法

- 算法1：正规方程法求权重

  - 对于每个叶子j的wj，有以下计算最优解的公式  

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_9.PNG)


- 算法2：贪心算法求分割点。不能枚举所有树结构，只能用贪心算法来分裂节点， 从单个叶子开始，遍历所有属性，遍历属性的可能取值， 并反复将分支添加进来。

  - 精确：列举所有特征上的所有可能分割。 为了有效地做到这一点，算法必须首先根据特征值对数据进行排序，然后按排序顺序访问数据，以上式中的梯度统计量。

  ![](http://7xiegr.com1.z0.glb.clouddn.com/xgboost_12.PNG)

  - 近似：

## 2.4收缩和采样

- 收缩：在每棵树加进来后，将新增加的w按因子α缩放，类似于优化中的学习率，收缩减少了每棵树的影响，并为未来的树留出空间来改进模型。
- 特征列采样：也加速了稍后描述的并行算法的计算。