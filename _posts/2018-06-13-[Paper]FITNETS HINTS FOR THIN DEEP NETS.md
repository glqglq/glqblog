# Abstract

- 网络加深会提升性能，但会难以训练
- 本文在知识精馏的工作上，不仅使用输出，还使用教师的中间层表示进行学习。这样允许教师网络更深、学生网络更浅。
- 由于学生网络的隐藏层少，所以引入附加参数来将学生隐藏层映射到教师隐藏层

# Introduction

- 之前的工作——知识蒸馏：
  -  Bucila et al. (2006)：训练一个神经网络来模拟一个大网络/集成学习模型输出，用集成网络（ensemble）来标记未标记的数据，并用集成网络标记的数据训练网络。
  - Ba＆Caruana（2014）：
  - Hinton＆Dean（2014）：
- 之前的工作——中间层学习：
  - 根据中间层target，隐藏层一层接一层地被训练
  - Weston et al.（2008）：半监督embedding为中间层提供了指导，帮助学习特别深的网络。
  - Cho et al（2012）：在无监督场景中每隔一层用另一个模型的激活来缓解DBM优化。监督通过在隐藏层上stacking具有softmax层的监督MLP来引入的。
  - Bengio（2009）：Curriculum Learning strategies (CL)通过修改训练分布来解决优化问题，逐渐接受增加适当难度的例子。其实就像一个continuation method，加速训练收敛
- 本文提出了Fitnets方法来训练小模型，从而压缩宽又深的模型。
  - 从教师隐藏层引入“中间层提示”来指导学生网络，希望学生网络学到中间表示。Hints允许训练更细、更深的网络。

# Method    

- Hint是教师网络的中间层输出，用于负责指导学生网络的中间层输出Guided。希望这两个输出一致！
- Hints是正则化的一种，guided layer越深，网络的灵活性越差，Fitnets更容易受过度正则化的影响。
- 鉴于教师网络通常比FitNet宽，所选Hint层可能会有比Guided层更多的输出。所以加一个regressor 函数到guided层前面，使其输出与Hint层大小匹配。
- 在引导层和提示层是卷积的情况下，使用完全连接的regressor会显着增加参数的数量和内存消耗。
- 这里使用卷积regressor，其被设计为使得学生网络guided与教师网络hints大致相同的特征图，regressor的输出与教师hint具有相同的大小。
- 损失函数：
  - uh：教师网络hint层的深层嵌套函数
  - vg：学生网络guided层的深层嵌套函数
  - Whint：教师网络参数
  - Wguided：学生网络参数
  - r：guided层前的regressor函数
  - Wr：regressor函数r的参数

![](https://upload-images.jianshu.io/upload_images/1770756-4beb985c64a66c32.png)

- 算法描述：
  - 教师网络的训练参数WT，FitNets的随机初始化参数WS，以及分别对应于hint层、guided层的两个参数h和g。Wr作为regressor的参数。
  - 根据教师hint层的预测误差LHT训练学生网络guided层
  - 用KD方法训练整个网络