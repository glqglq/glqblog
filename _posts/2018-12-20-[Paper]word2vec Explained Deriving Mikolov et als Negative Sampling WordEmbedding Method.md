---
    author: LuckyGong
    comments: true
    date: 2018-12-20 20:27
    layout: post
    title: Papper-word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method
    categories:
    - nlp
    tags:
    - nlp
    - papper
---

# 0.Abstract

- 解释负采样。

# 1.skip-gram模型

## 1.0符号定义

- 当前词汇：w
- 当前词汇的某个上下文词汇：c∈C
- 当前单词的所有可用上下文：C
- 语料库：Text
- 模型参数：θ
- 词汇集合：D
- 词汇集合随机采样正样本置为负样本的词汇集合：D′  
- lookup表维度：d
- lookup表：v
  - 上下文的向量表示（待优化的参数）：vc∈Rd
  - 当前词汇的向量表示（待优化的参数）：vw∈Rd
- p(D=1|w,c)：(w,c)来自训练数据（即c是w的上下文）的概率

## 1.1skip-gram模型及其参数化

- 模型：

$$
\arg\max_\theta \prod_{(w,c)\in D}p(c|w;\theta)
$$

- 概率化：使用softmax
  $$
  p(c|w;\theta) = { e^{v_c \cdot v_w} \over \sum_{c’ \in C}e^{v’_c \cdot v_w}}
  $$

  $$
  \arg \max_\theta \sum_{(w,c)\in D}logp(c|w) = \sum_{(w,c)\in D}(loge^{v_c \cdot v_w}-log\sum_{c’}e^{v_{c’}\cdot v_w})
  $$


- 这里以下式子计算量特别大，可以用层次softmax解决。
  $$
  \sum_{c’ \in C}e^{v’_c \cdot v_w}
  $$


# 2.负采样

- 负采样基于skip-gram，事实上其优化了一个不同的目标函数。

- 目标函数：
  $$
  \begin{align} 
  &\arg\max_\theta \prod_{(w,c)\in D}p(D=1|w,c;\theta) \\ 
  = & \arg\max_\theta log \prod_{(w,c)\in D}p(D=1|w,c;\theta)\\ 
  = & \arg\max_\theta \sum_{(w,c)\in D}logp(D=1|w,c;\theta) 
  \end{align}
  $$

- 由sigmoid，其中：
  $$
  p(D=1|w,c;\theta)={1 \over {1+e^{-v_c \cdot v_w}}}
  $$

- 则：
  $$
  \begin{align} 
  \arg \max_\theta \sum_{(w,c)\in D}log{1 \over {1+e^{-v_c \cdot v_w}}} 
  \end{align}
  $$


- 这个目标函数存在一个问题，如果我们设定θ使得每一对(w,c)的p(D=1|w,c;θ)=1，那这个目标函数就无意义了。只要设置θ，使得vc=vw且vc⋅vw足够大，则上述这种情况就很容易出现（在Goldberg[1]的实验中当vc⋅vw≈40时，概率就为1了）。

- 因为为了避免所有向量都是相同的值，可以去掉某些(w,c)的组合，即可以随机选择(w,c)对中的一部分作为负例。

- 目标函数变为：
  $$
  \begin{align} 
  & \arg\max_\theta \prod_{(w,c)\in D}p(D=1|c,w;\theta) \prod_{(w,c)\in D’}p(D=0|c,w;\theta) \\ 
  = & \arg \max_\theta \prod_{(w,c)\in D}p(D=1|c,w;\theta) \prod_{(w,c)\in D’}(1-p(D=1|c,w;\theta)) \\ 
  = & \arg \max_\theta \sum_{(w,c)\in D}logp(D=1|c,w;\theta) + \sum_{(w,c)\in D’}log(1-p(D=1|c,w;\theta)) \\ 
  = & \arg \max_\theta \sum_{(w,c)\in D}log{1 \over {1+e^{-v_c \cdot v_w}}}+ \sum_{(w,c)\in D’}log(1-{1 \over {1+e^{-v_c \cdot v_w}}}) \\ 
  \\ 
  =& \arg \max_\theta \sum_{(w,c)\in D}log\sigma(v_c \cdot v_w)+ \sum_{(w,c)\in D’}log\sigma(-v_c \cdot v_w) 
  \end{align}
  $$


- 这个目标函数表面的含义也可以理解为要尽量增大正例的(vc⋅vw)数据对，而尽量降低负例的(vc⋅vw)数据对。词与词之间，若其上下文很相近，则他们本身也很相似。
- 与skip-gram不同，本公式不对p（c | w）建模，而是模拟与w和c的联合分布相关的数量。

# 参考

- http://qiancy.com/2016/08/24/word2vec-negative-sampling/