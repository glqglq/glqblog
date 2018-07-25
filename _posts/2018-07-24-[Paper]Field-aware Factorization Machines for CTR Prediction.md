---
    author: LuckyGong
    comments: true
    date: 2018-07-25 20:27
    layout: post
    title: Paper-Field-aware Factorization Machines for CTR Prediction
    categories:
    - ad
    tags:
    - ad
    - paper
---



# Absract

- 在本文中，我们建立了FFM作为一种有效的方法来分类大型稀疏数据，包括来自CTR预测的数据。 
- 首先，我们提出了有效的FFM训练方法。 然后我们全面分析FFM并将此方法与竞争模型（competing models）进行比较。 实验表明，FFM对于某些分类问题非常有用。 最后，我们发布了一套供公众使用的FFM。

# 1.Introduction

- 如论文例子所示，关联特征对CTR预测至关重要，然而线性模型难以学习这些信息。
- 基于FM的PITF方法被提出来用于个性化标签推荐。在2012年KDD杯中，PITF的轮廓（被称为“因子模型”）由“Team Opera Solutions”提出。 由于该术语过于笼统并且很容易与因子机混淆，因此本文将其称为“场感知因子机”（FFM）。
- PITF考虑三个特殊字段：user、item、tag，FFM更加通用。PITF中有以下几个结论：
  - 用SGD来优化，为避免过拟合，之训练一个epoch
  - FFM在他们尝试的六种模型中最好。
- 本文成果
  - 将FFM与Poly2和FM进行比较，首先进行概念性的比较，再做实验查看准确性和训练时间的差异。
  - 提出了训练FFM的计数，用于FFM的有效并行化算法以及使用early stop来避免过拟合。
  - 发布了一个开源软件。

# 2. POLY2 AND FM

- POLY2：d=2的多项式映射通常可以有效地捕获特征组合连接信息。通过在d=2的显式映射上使用线性模型，训练和测试时间都比用核方法快得多。poly2模型会为每一个特征对学习一个权重： 

  ![](https://pic4.zhimg.com/v2-2ba646eec9a0eca44166aa521312a12d_r.jpg)

  - h(j1,j2)是一个把j1和j2编码成一个自然数的函数
  - 公式的计算复杂度是O(n^2)，n表示每个样本的非0数目的平均。

- FM：FMs是为每个特征学习一个隐向量表示，每个隐向量包含k维，这样特征交互的影响就被建模成2个隐向量之间的内积：

  ![](https://pic3.zhimg.com/v2-1d5c97f436cf64567294ca51f171ecb8_r.jpg)

  - 通过一些计算技巧可以把计算复杂度降到O(nk)

- 在稀疏数据集上，FMs模型要比poly2模型好一些，比如对于论文的例子，对于pair(ESPN,Adidas)只有一个唯一的负样本，通过poly2模型会学习到一个大的负向权重对于这个pair，然而对于FMs来说，因为它是学习ESPN和Adidas的隐向量表示，所有包含ESPN的样本和所有Adidas的样本都会被分别用来学习这2个隐向量，所以它的预测会更准确一些。

# 3.FFM

## 3.1模型

- FFM的想法源于PITF[7]，PITF用于具有个性化标签的推荐系统。 在PITF中，它们假定三个可用字段，包括user，item和tag，以及在单独的隐空间中分解（user，item），（user，tag）和（item，tag）。 在[8]中，他们将PITF推广到更多字段（例如，AdID，AdvertiserID，UserID，QueryID），并将其有效地应用于CTR预测。 因为[7]的目标是推荐系统，并且仅限于三个特定域（user，item和tag），[8]缺乏关于FFM的详细讨论，在本节中我们提供了关于CTR预测的FFM的更全面的研究。 对于大多数CTR来说，特征可以被group为field。 在上述例子中ESPN，Vogue，NBC可以被group成Publisher，而Nike，Gucci，Adidas属于Advertiser，FFM会充分利用group的信息。 

- FM中每个特征只有一个隐向量表示，这个隐向量被用来学习与其他任何特征之间的影响。 考虑ESPN，w(ESPN)被用来学习与Nike的隐性影响w(ESPN)*w(Nike)，还被用来学习与Male的影响w(ESPN)*w(Male),然而由于Nike和Male属于不同的域，它们的隐性影响是不一样的。 在FFMs里面，每个特征会有几个不同的隐向量，上述例子的FFMs表示如下：

   ![](https://pic4.zhimg.com/80/v2-5fb0cda9cf5d4a192d275809a70ad1d1_hd.jpg)

- 我们看到，为了学习（ESPN，NIKE）的潜在影响，使用wESPN,A是因为Nike属于广告商场。使用了wNike,P是因为ESPN属于字段Publisher。

  再次，为了学习（EPSN，男性）的潜在影响，使用WESPN; G是因为男性属于性别领域，而wMale; P被使用是因为ESPN属于字段Publisher。在数学上，

- 公式：

  - f1、f2：是j1和j2的域
  - f：域的数目
  - nfk：FFM的变量数

![](https://pic3.zhimg.com/v2-e48fb501a67acef55d9e35cf854b7965_r.jpg)

- 上式可以在线性时间O(非n^2k)内计算。由于FFM中每个隐变量通常只需要通过特定域来学习，通常有：kFFM<<kFM    

## 3.2目标函数

- 除了φLM（w; x）被φFFM（w; x）替换之外，该优化问题与LR相同。

## 3.3参数估计

- 使用随机梯度法（SGD）。 

  - 因为x非常稀疏，所以只更新具有非零值的维度。
  - 使用AdaGrad [10]，因为[12]已经证明了矩阵分解的有效性，矩阵分解是FFM的一个特例。
  - 算法描述：
    - η：学习率
    - w：初始化从均匀分布[0,1/根号k]中取样
    - G：

  ![](http://ot0qvixbu.bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20180725163112.png)

  ![](http://ot0qvixbu.bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20180725162938.png)

  ![](http://ot0qvixbu.bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_201807251631121.png)

  ![](http://ot0qvixbu.bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_2018072516311212.png)

- 将每个样本标准化以具有单位长度使得测试精度稍微更好并且对参数不敏感。

- 最近还提出了一些自适应学习速率调节表，如[10,11]所示，以促进SGD的训练过程。 

- 并行化：用Hogwild [13]，它允许每个线程独立运行而不需要任何锁定。

- 输入数据加入域信息：为每个特征标明属于那个域。

  - 离散特征：通常转换为onehot特征
  - 数值型特征：
  - 自己单独一个域的特征：

# 参考资料

- 论文
- https://blog.csdn.net/jediael_lu/article/details/77772565