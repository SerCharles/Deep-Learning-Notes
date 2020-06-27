---
typora-root-url: ./
---

# Part3 CNN

## 1.CNN基本单元

### 1.1 正向传播

**卷积层：**
$$
S(w,h,d')=(I*K)(w,h,d')=\sum_d\sum_i\sum_j I(w+i,j+j,d)K(i,j,d) + b
$$
其实就是对于图片，每个K * K * C的，和卷积核大小一样的块，和卷积核对位相乘，加到一起，再加上一个常数，作为一个数放在结果里。（滑动内积）

卷积核的大小是K * K * C，其channel和输入图片channel应该一样，参数为K * K * C + 1个。

![conv](/conv.png)

再加上padding（图片周围一圈补0），stride（卷积跳步），卷积结果的大小为
$$
\lceil(size - filter + 2*pad)/stride \rceil + 1
$$
卷积结果的channel个数为卷积核个数D

**池化层：**

和卷积类似，也是对于图片，每个K*K的部分，取最大值（max pooling)或者平均值（average pooling）作为结果的值。

池化层没有参数，结果大小和卷积结果大小一样，channel个数不变。

### 1.2 反向传播

**卷积层：**

卷积层的残差求解方法是反卷积：
$$
\sigma^{(l)} = \sum_d(\sigma_d^{(l+1)} * rot180(\theta_d^{(l)}))\bigodot g'(z^{(l)})
$$
![transconv](/transconv.png)

具体操作的时候要先对l+1做一样的stride，然后进行padding保证维度匹配，之后反卷积即可

**池化层：**

池化层进行上采样，也就是扩张l+1到l应有的大小，然后做max和average的逆运算即可。

### 1.3 基本的CNN单元

卷积层：卷积+ReLU激活

池化层：只有池化

## 2.CNN扩展单元

### 2.1 3D卷积

用途：处理视频，三维模型等

单元：3D卷积层-D * D * D，和2D唯一区别是不要求第三个维度和channel数一样

### 2.2 图卷积

用途：图的操作，比如节点分类，图分类，丢失链接猜测

单元：
$$
h_i^{l+1} = \sigma(h_i^l W_0^l + \sum_{邻居}\frac{1}{c_{ij}}h_j^lW^l)
$$
W为全局的共享参数

### 2.3 非局部卷积

用途：获取图片中较远距离位置的相关性（self attention）
$$
y_i = \frac{1}{l(x)}\sum_{\forall j}f(x_i,x_j)g(x_j)\\
其中g(x_j)=Wx_j（参数共享，线性映射）\\
f(x_i,x_j)=e^{x_i^Tx_j}(Gaussian)\\
或者e^{\theta(x_i)^T\phi(x_j)}(Embedded Gaussian)，用于度量相似性\\
l(x)=\sum_{\forall j}f(x_i,x_j),正则化因子
$$

## 3.CNN网络架构

### 3.1 VGG

- small filters，deeper network：3个3*3的卷积核感受野和7 * 7一样，但是参数数量，运算速度和效果都更好
- 1*1卷积：用于改变channel大小
- 少用全连接：速度和大小都是瓶颈，只在最后一层用就行了

### 3.2 GoogleNet

- 增加网络的宽度很有必要：可以分成几条路径，然后最后concencate
- 每条路径可以有不同的感受野大小/卷积核大小：增加特征的丰富程度
- small filters，deeper network：1 * 3与3 * 1结合=3 * 3,2个3 * 3 = 5 * 5 

### 3.3 ResNet

- Highway Network：将不同层（ResNet里是每隔2个3*3卷积）结果直连，让更深的网络梯度传播更快，训练很深的网络成为可能（进一步改进：densenet，每个模块里的每一层结果都直连）
- Resnext：也是增加网络的宽度，每个模块被分为32条路径，最后concencate到一起

### 3.4 压缩

- 剪枝：

  - 全部完成后剪枝：训练完毕后剪掉绝对值很小的节点
  - 部分完成就剪枝：lottery ticket 假设，循环进行初始化--训练一部分--剪枝小节点

- 压缩：用k-means将网络的值分类（比如2.01,1.89都分类2；3.21，2.99都分类3），huffman编码压缩

- 组卷积：将channel分组，只有组内的channel才在卷积的时候一起计算（逐通道卷积）。然后通过1*1卷积（逐点卷积），以及shuffle来让组间的channel产生关系。

  如果原来的卷积核是K * K * M * N，那么逐通道和逐点卷积能压缩到K * K * M + M * N

  ![group_conv](/group_conv.png)

## 4.CNN分析与思考

### 4.1 CNN的基础：平移不变性和局部连接

**局部假设：**局部的信息足够识别了

**平移不变性假设：**如果一个特征在位置（x,y)有效，那么在(x',y')也应该有效

因为这两个假设，才会采用卷积网络。首先卷积网络的感受野大小有限，一个像素只和其周围能一起卷积的像素有直接关系（多层感受野会扩大，但是这种连接更加间接了）。其次卷积网络全局参数共享，如果在不同位置效果不一样那肯定不对。

### 4.2 数据增强，不变性

我们希望CNN是不变和等变的，但是卷积网络既不是不变的，也不是等变的。比如一个数据发生细微变化，网络识别可能就不对了，这是不行的。

目前缓解这种问题的方法主要是数据增强：通过生成一组数据的各种变换，来让网络学到不同的情况，增加其泛化能力。

但是根本上解决的方法还是构造更好的网络。一种想法是改进max pooling为反锯齿max pooling。把原来的dense pooling + subsampling改造成dense pooling + 用Blur kernel卷积 + subsampling来增加平移不变性。另一种想法是capsule网络，考虑长度，夹角等。

CNN还有其他问题，比如一个人的眼睛和嘴换位子，因为pooling也能正确识别之类的。

![problem](/problem.png)

### 4.3 CNN的其他特性

- Weight Decay对CNN没用
- 在大数据集上预训练的CNN网络可以迁移到其他问题，但是有监督学习在跨任务（比如图片分类-物体检测）不好用，无监督学习更好

### 4.4 AutoML

AutoML是**自动学习网络结构和超参数**的方法

一般用贝叶斯优化或者强化学习等

强化学习：

- Reward：准确率R
- Action：预测的超参数
- 期望的最大Reward：最低的损失函数
- 不能有监督学习
- Policy Gradient算法

## 5.整体思维导图

![mindmap](/mindmap.jpg)