# Part2 深度学习基础

## 1.深度学习基本步骤

### 1.1 前向传播

前向传播是已知网络参数，输入数据，求得对应的结果和损失函数值的过程。

**非最后一层：**每一层一般是先带入网络求值，然后再带入激活函数求值。比如输入为X，隐藏层为MLP，参数为W,b，激活函数为ReLU，则输出为Y=ReLU(WX+b)。激活函数为了提供非线性。

**最后一层：**对于最常见的分类问题来说，最后一层通常是**全连接层+softmax+求损失函数**。

全连接层将输入映射到一个n维向量，n是类别数目；**n维向量最大值所在位置就是预测的结果。**

之后对n维向量做softmax，结果还是n维向量，不过实现了“赢者通吃”，原先n维向量的结果差异被显著放大了。
$$
g(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{k}e^{z_j}}\\
因为softmax可能导致数值溢出，常常分子分母都除以e^{z_{max}}
$$
最后用n维向量结果和标签y做交叉熵求损失函数。标签需要经过one-hot encoding（独热编码），也就是这样：如果有n个类别，某个结果y属于第i类，则这个结果的one-hot encoding有n维，其第i维是1，其余维是0。交叉熵的结果是一个数值，就是损失函数。
$$
-\sum_{j=1}^{k}q_jlog(p_j)
$$
而因为深度学习常常一个batch一个batch，而不是一个数据一个数据输入，所以最终损失函数和结果也是以batch形式呈现的。batch之间没有关系，但是带上batch，损失函数就这样了(m为batch size）：
$$
J(\theta,b)=\frac{1}{m}\sum_{i=1}^{m}J(\theta,b;x^i,y^i)
$$


### 1.2 反向传播

**反向传播的数学本质是链式法则，算法本质是动态规划（填表算法）。**

![diff](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\diff.png)

比如求上图J对theta1的导数，就要先知道J对z1的导数（残差），再带上z1对theta1的导数。

而这样的话，为了计算方便，就最好把残差先求出来记录下来。而求残差又需要求后一层的残差。因此，应该从后往前求残差并且记录，然后根据残差推导数。

**求残差：**因为正向传播公式如下
$$
z^{(l+1)} = \theta_l(g(z^{(l)}))
$$
因此有反向传播递推公式
$$
\sigma_i^{l} = \frac{\partial J}{\partial z^{i}} = \frac{\partial J}{\partial z^{i+1}}\frac{z^{i+1}}{z^{i}} = \sum_{j=1}^{s_{l+1}}\sigma_j^{(l+1)}\theta_{ji}^{(l)}g'(z_i^{(l)})
$$
**根据残差求梯度：**
$$
\frac{\partial J}{\partial \theta_{ij}^{(l)}} = a_{j}^{(l)}\theta_i^{(l+1)}，其中a为第l层的z经过激活函数的结果（z没经过激活函数）
$$
**优化：**自动求导--建立导数计算图

### 1.3 梯度更新

$$
\theta'=theta - \eta \Delta^{(t)}(\frac{\partial J}{\partial \theta})，其中\eta 为学习率，\Delta为优化算法
$$

优化算法有多种策略，有一阶的梯度下降（只考虑一阶导数），还有二阶的牛顿法，Quasi牛顿法，共轭梯度法（考虑二阶导数也就是Hesse矩阵）。

## 2.深度学习常见技巧

### 2.0 对症下药：常见的问题

梯度消失：梯度太小训练不动

梯度爆炸：梯度过大

欠拟合：训练准确率太低

过拟合：训练准确率很高，但是测试准确率很低，模型泛化能力不行

### 2.1 初始化策略

**Xavier初始化：**初始化为方差为1/n，均值为0的正态分布

**Kaiming初始化（适合ReLU）：**初始化为方差为2/n，均值为0的正态分布（因为ReLU小于0的一半值为0了）

**迁移学习：**把其他问题的预训练参数拿过来初始化，如果自己的训练样本少，就只训练最后一个全连接层，否则就全都训练。

### 2.2 激活函数选择

常用：Sigmoid，tanh，ReLU，leakyReLU

Sigmoid和tanh容易梯度消失，ReLU小于0容易梯度消失

一般最常用ReLU，RNN常用tanh

### 2.3 Batch Normalization

（对应操作：Weight Norm（RNN常用），Layer Norm）

对整个batch的所有数据x：
$$
x_i = \gamma\frac{x_i -\mu_B}{\sqrt{\theta_B^{2}+\epsilon}} + \beta
$$
用途：防止梯度消失和梯度爆炸，增加训练速度，改进效果

### 2.4 Dropout

**在训练时**，随机设置某些结点值为0（测试的时候不能这样做）

对于每个节点，有p概率被设置为0，其余情况被设置为h/(1-p)，这样是为了保证仍是无偏估计

用途：引入随机性，防止过拟合

### 2.5 Weight Decay

损失函数加入正则项，变为
$$
J(D;\theta) = L(y,f(x);\theta) + \Omega(\theta)
$$
有两种方法：一阶（参数值的绝对值求和），二阶（参数值的平方求和）

用途：限制参数的大小，防止参数绝对值太大导致过拟合

### 2.6 Learning Rate Decay

因为网络训练一般是前面较容易，后面较难，所以学习率也要动态调整。

有几种调整方法：指数衰减，线性衰减，阶梯衰减（平时不变，到达某个epoch瞬间除以m）

用途：动态调整学习率，使得训练更快，更稳定

### 2.7 优化算法

见3，常用SGD（0.1这种较大的学习率），Adam（0.001这种较小的学习率）

### 2.8 训练时的一些好习惯

**Early Stopping：**当测试准确率开始下降的时候，及时停止--过拟合了

**先过拟合少量数据：**搭建网络之后，在训练前先过拟合少量数据，如果能过拟合，说明模型至少能收敛，不是错的

### 2.9 训练-测试集的选择和度量方法

**holdout：**固定训练集和测试集

**k折验证：**数据均分为k份，每次k-1份训练1份测试，交叉验证k次

**度量：**准确率，精度，召回率，F1等度量

## 3.优化算法

### 3.1 SGD--随机化

一般的梯度下降使用所有训练数据进行下降，这是很慢的。

SGD只随机选用m大小的batch下降，m << n，而且每次会shuffle训练数据集。

一方面，这样更快；另一方面，带来了随机性，不容易掉入局部极小。

### 3.2 Momentum，Nestrov Momentum

动量的引入能让算法更容易逃离局部极小值。
$$
\Delta_t = \beta \Delta_{t-1} + \bigtriangledown J^t(\theta^t)
$$
![momentum](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\momentum.png)

而Nestrov Momentum是先求动量再求梯度，是改进版Momentum，使用其的NGD如下：
$$
{\theta^t{’}} = \theta^t - \beta \Delta_{t-1}\\
\Delta_t = \beta \Delta_{t-1} + \bigtriangledown J^t(\theta^t{’})
$$


### 3.3 Adam系列

有一种重要的优化思路是自适应学习速率，这种思路的想法是**“中和”不同分量的学习速率**，使得梯度较大的分量学习率较低，梯度较小的分量学习率较高，从而达到均衡。

Adagrad是在SGD基础上改进的，采用了如下的自适应更新方法
$$
r^t=r^{t-1} + \bigtriangledown J^t(\theta^t)^2\\
h^t = \frac{1}{\sqrt{r^t}+\sigma}\\
\Delta^t = h^t \bigodot \bigtriangledown J^t(\theta^t)
$$
但是这样，r会不断快速上升，导致学习率h不断下降，最终梯度消失

因此，RMSprop更新了r的更新公式
$$
r^t= \rho r^{t-1} + (1-\rho)\bigtriangledown J^t(\theta^t)^2
$$

再加上Momentum，以及去掉bias，就变成了Adam
$$
r^t= \rho r^{t-1} + (1-\rho)\bigtriangledown J^t(\theta^t)^2\\
r^t{'}=r^t/(1-\rho^t)\\
h^t = \frac{1}{\sqrt{r^t{'}}+\sigma}\\
momentum:\\
s^t=\epsilon s^{t-1}+(1-\epsilon)\bigtriangledown J^t(\theta^t)\\
s^t{'}=s^t/(1-\epsilon^t)\\
\Delta^t = h^t{'} \bigodot s^t{'}
$$
使用Nestrov Momentum就是RAdam

但是Adam有不收敛的问题：没有保证h单调递减，当仅当h单调递减才收敛

因此AMSGrad改进了Adam，用
$$
r^t{'}=max(r^t,r^{t-1}{'})\\
$$
保证收敛

### 3.4 优化的思维导图

最常用的是SGD+Momentum和Adam

![optimize](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\optimize.png)

## 4.最基本的单元：MLP

### 4.1 定义和基本性质

$$
y=sgn(\sum \theta_i x_i - T)
$$

当x为bool表达式，可以表达逻辑

![logic](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\logic.png)

单层感知机可以表达且，或，非；多层感知机可以表达任意命题逻辑；

当x为实数，可以表达超平面

![line](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\line.png)

当感知机用实数和逻辑组合的时候，可以表达很复杂的图形

![MLP](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\MLP.png)

### 4.2 表达能力

深度和宽度对MLP表达能力都很重要。深度增加，表达能力指数增长；宽度增加，表达能力多项式增长；二者缺一不可，但是深度增加更重要

## 5.整体思维导图

![mindmap](E:\Learning\THU\2020 Spring\深度学习\总结\Part2-Basic\mindmap.jpg)