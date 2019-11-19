---
    author: LuckyGong
    comments: true
    date: 2019-11-19 20:27
    layout: post
    title: Paper-Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    categories:
    - ad
    tags:
    - ad
    - paper


---



# 1.Introduction

- lexical matching：不准确
- 隐式语义模型：将类似上下文中的不同term聚到一个语义群里。
  - LSA：用SVD将doc-term矩阵分解，$\hat{D}=A^TD$，A是映射矩阵，最后用余弦距离进行检索。
- 概率主题模型：无监督，与评估函数松散耦合。
  - PLSA：
  - LDA：
- ctr数据训练得出的模型：
  - BLTMs：生成模型，query和doc要共享同样的topic分布，每个topic包含同样的factions of words，最大化log似然，是次优的结果。
  - DPMs：用S2Net算法，遵循pairwise learning-to-rank模式，优化过程有很多矩阵乘法（跟随词表大小激增）。
  - 用auto-encoders扩展语义模型：无监督，优化目标是文档的重构。

# 3.DEEP STRUCTURED SEMANTIC MODELS FOR WEB SEARCH

## 3.1DNN for Computing Semantic Features

## ![](.\img\9.png) 

- 模型假设：如果点击的话，query和doc是相关的。
- tanh作为激活函数。
- 最终是用余弦相似度。

- 优化目标：最大化文档点击率

## 3.2Word Hashing

- 目的/优点：
  - 大大降低维度
  - 减少训练集中未出现的词的影响
  - 减少词性变化的影响。
- 做法：可以看做是固定线性转换（fixed linear transformation）。
  - word头部尾部加上ending mask
  - 将word分为ngram
  - 同一个单词进行统计，出现过的ngram在对应位置+1，最后形成一个multi-hot的编码
- 问题：碰撞，但是碰撞概率极低。

## 3.3训练

- 模型：条件（后验）似然概率P(D|Q)，通过softmax计算得到。系数gamma>0用于平滑，当它大于1时，概率之间的差异会被放大，当它小于1时，概率直接的差异会被缩小。

  ![](.\img\10.png)

- 目标函数：最大化极大似然。度量D个Doc（包括D+和D-），其中D-和D+按照4：1采样取得，实验中发现正负样本的比例其实没那么重要。

  ![](.\img\11.png)

- 算法：sgd算法

## 3.4实现细节

- 数据分为训练集、验证集
- 初始化：均匀分布$\sqrt{(-\frac{6}{fanin+fanout},\frac{6}{fanin+fanout})}，其中franin是输入单元数，franout是输出单元数$
- pretrain没有用。
- batchsize为1024
- 20个epoch会收敛

# 4.实验

## 4.1Datasets & Evaluation Methodology

- eval数据集有16510个query，每个query平均要预估15个web doc的ctr
- label是人工生成的，0-4（4是最相关的，0是最不相关的）
- 预处理：white-space tokenized、lowercased、保留数字、不词干化、不词尾化。
- 2-fold交叉验证
- metric指标：
  - NDCG
  - 使用paired t-test进行显著性检验，p值小于0.05时被认为有统计学意义。
- train数据集有1亿个数据集，只用title和query做。

## 4.2结果

- 词汇匹配方法：
  - TF-IDF：doc和query都用TF-IDF term weighting表示为term vector。
  - BM25
- word翻译模型：侧重解决文档语言差异
  - WTM：比TF-IDF和BM25好。
- 语义模型（无监督）：DNN的隐式语义模型比线性映射模型（如LSA）表现好。LSA和DAE都是在doc上无监督学习的，所以没有词汇匹配方法好。
  - LSA：使用PCA而不是SVD来计算linear projection matrix
  - PLSA：只用doc训练，使用MAP估计进行学习
  - DAE：模型训练复杂，词表减少为40K。4层隐藏层，每层300个node，中间有个bottleneck层（128个node）。
- 语义模型（ctr数据监督）：比词汇匹配方法好。
  - BLTM-PR：EM算法优化，约束要求query和title有相同的fractions of terms分配给每个隐topic。PLSA的升级版。
  - DPM：线性判别映射模型，映射矩阵用S2Net学习到。LSA的升级版。