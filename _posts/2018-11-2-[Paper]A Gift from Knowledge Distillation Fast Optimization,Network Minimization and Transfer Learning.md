---
    author: LuckyGong
    comments: true
    date: 2018-11-02 20:27
    layout: post
    title: Papper-A Gift from Knowledge Distillation:Fast Optimization,Network Minimization and Transfer Learning
    categories:
    - modelcompression
    tags:
    - modelcompression
    - papper

---

# Abstract

- 我们根据层之间的流量（flow between layers）定义要传输的蒸馏知识，这通过计算两层特征之间的内积来得到。
- 这种方法有三种重要现象：
  - 学生DNN学习蒸馏知识的优化速度比原始模型快得多
  - 学生DNN优于原DNN**（什么优于？性能嘛？为什么？）**
  - 学生DNN可以从在不同任务中训练的教师DNN学习提取的知识，并且学生DNN优于从头开始训练的原始DNN。

# 1.Introduction

- KD方法：难以优化非常深网络。
- Fitents：加入中间层，优化非常深的网络。
- Gatys等人使用Gramian矩阵来表示输入图像的纹理信息（texture information）。 因为Gramian矩阵是通过计算特征向量的内积而生成的，所以它可以包含特征之间的方向性，这可以被认为是纹理信息。
- 本文：
  - motivation：
    - 知识迁移的结果对所蒸馏知识的定义非常敏感（即：迁移什么）。
    - 考虑到真正的老师教给学生如何解决问题的流程，我们将高级蒸馏知识定义为解决问题的流程。因为DNN按顺序使用许多层来从输入空间映射到输出空间，所以解决问题的流程可以定义为来自两个层的要素之间的关系。
  - 方法：
    - 我们通过使用由两层特征之间的内积组成的Gramian矩阵来表示解决问题的流程。
    - Gatys等人计算层内的Gramian矩阵，我们计算跨层的Gramian矩阵
  - 实验度量（知识的有用性）：
    - 快速优化：理解解决问题的流程的DNN可以是解决主要任务的良好初始权重，并且可以比普通DNN更快地学习。
    - 改善小型网络的性能：
    - 迁移学习：

# 2.Related Work

- 知识迁移：Hinton的工作、Fitnets、Net2Net
- 快速优化：
  - SGD等。
  - 在早期，具有零均值和单位方差的高斯噪声初始化非常流行。
  - 其他各种初始化技术如Xavier初始化也被广泛使用。
  - 出现了一些基于数学方法的新技术[18,22,14]。
- 迁移学习：微调

# 3.Method

## 3.1提出的蒸馏的知识

- DNN每层都生成特征图，越是高层越是有用的特征。
- 如果我们将DNN的输入视为问题而输出作为答案，我们可以将DNN中间生成的特征视为解决方案过程中的中间结果。
- 学生DNN不一定必须在输入特定问题时学习中间输出，最好是当遇到特定类型的问题时可以学习解决方法。

## 3.2蒸馏的知识的数学表达

- 表示为两个层之间的特征方向，FSP矩阵由两层之间的特征生成，
- 具体公式：见论文

## 3.3FSP矩阵的损失

- 学生网络和教师网络的矩阵数一致。
- 使用L2损失。
- 具体公式：见论文

## 3.4学习过程

- 基本条件：
  - 教师网络应该由一些数据集预先训练，该数据集可以与学生网络学习的数据集相同或不同。
  - 教师网络可以比学生网络更深或更浅。但是，我们认为教师网络与学生网络相同或更深。
- 学习过程：
  - 根据大模型的 FSP 矩阵调整小模型参数，使得小模型层间关系也和大模型的层间关系类似；
  - 直接用原损失函数（如交叉熵）继续精调小模型参数

# 4.实验

# 5.结论

