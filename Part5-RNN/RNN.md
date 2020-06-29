---
typora-root-url: ./
---

# RNN

## 1.RNN基础

### 1.1 基本单元

**一般：**
$$
h_t = tanh(Wh_{t-1} + Ux_t + b)
$$
**LSTM:**
$$
f_t = sigmoid(W_{xf}x_t + W_{hf}h_{t-1} + b_f) 遗忘门，控制是否遗忘之前的信息，接近1就记忆，接近0就遗忘\\
i_t = sigmoid(W_{xi}x_t + W_{hi}h_{t-1} + b_i)写入门，控制是否写入新的数据，接近1就写入，接近0就不写入 \\
o_t = sigmoid(W_{xo}x_t + W_{ho}h_{t-1} + b_o)输出门，控制是否输出结果，接近1就输出，接近0就不输出\\
g_t=tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)和基本架构一样，结果\\
c_t=f_t\bigodot c_{t-1} + i_t\bigodot g_t 记忆，从第一个时刻传下来，包括先前记忆与新的输出\\
h_t=o_t\bigodot tanh(c_t) 输出结果
$$
**GRU:**改进版LSTM，效果和思路类似，快
$$
r_t = sigmoid(W_{xr}x_t + W_{hr}h_{t-1} + b_r)重置门，就记忆，0就遗忘1\\
r_t = sigmoid(W_{xz}x_t + W_{hz}h_{t-1} + b_z)更新门，1就不更新，0就更新\\
c_t'= tanh(W_{xc}x_t + W_{hc}(r_t\bigodot h_{t-1}) + b_c)新的结果\\
h_t = c_t = (1-z_t)\bigodot c_{t-1} + z_t \bigodot(c_t')最终结果
$$
**反向传播：BPTT**
$$
\frac{\partial L_t}{\partial U} = \sum_{s=0}^{t}\frac{\partial L_t}{\partial y_t}\frac{\partial y_t}{\partial h_t}\frac{\partial h_t}{\partial h_s}\frac{\partial h_s}{\partial U},其中\\
\frac{\partial h_t}{\partial h_s} = \frac{\partial h_t}{\partial h_{t-1}}....\frac{\partial h_{s+1}}{\partial h_s}，序列可能很长，因此在求导一定次数后就停下\\
性质：||\frac{\partial h_t}{\partial h_{t-1}}|| \leq \sigma_{max}\gamma，||\frac{\partial h_t}{\partial h_{s}}||\leq(\sigma_{max}\gamma)^{s-t}
$$
**因此，这种求导容易梯度爆炸和消失。解决梯度爆炸：gradient clipping。解决梯度消失：LSTM,GRU，各种记忆机制。**

### 1.2 基本结构

#### 1.2.1 通用架构

**多层RNN：**每层的结果输入下一时刻和下一层，最后用全连接层，softmax和交叉熵

![network](/network.png)

**双向RNN：**双向，其中正向反向都依赖输入，影响输出

![bidirect](/bidirect.png)

#### 1.2.2 解决不同问题的具体架构

![structure](/structure.png)

**多对一：**多个输入，一个输出，用于情感分析，文本分类等

**一对多：**一个输入，多个输出，用于图片描述等

**多对多（同构）：**每个时刻的输入都直接能输出，要求输入输出同构，用于语言建模等

**多对多（异构）：**输入编码成一个状态（encoder），再一对多输出（decoder），用于机器翻译等

多对多的输出阶段，前一个输出作为下一个时刻输入。使用**beam search**技术，分类讨论上一个时刻输出的几种较为可能的结果作为下一个时刻输入。

### 1.3 训练技巧

**Gradient Clipping：**如果梯度大小大于threshold，则
$$
g = \frac{threshold}{||g||^2}g，将g范数设置为threshold
$$
作用：防止梯度爆炸

**Curriculum Learning:**先训练简单的，再训练复杂的，防止误差积累

**Variational Dropout：**不同时序的dropout一样

**Layer Norm：**对各个channel减掉均值除以方差

**Weight Norm：**对y=g(wx+b)，将每个w正则化

**为什么不用batch norm？**一方面,RNN batch数量少，均值方差效果不好；另一方面，这样参数无法在各个时刻内共享

## 2.Attention

### 2.1 一般形式

Query，Key，Value三个要素
$$
通过计算相似度来求得权重（对齐）：W = a(Q,K)\\
对值加权平均：output = \sum_{j=1}^{T}w_{ij}v_j
$$
整体思路和神经网络内存读写类似

**用于解决问题：RNN，尤其是异构sequence to sequence的误差累加问题**

### 2.2 Temporal Attention

**目标：将encoder向量和decoder的每个时序对齐**

encoder阶段：用双向循环网络得到h=[h正，h反]

attention：通过求h与decoder网络时序i-1的记忆s(i-1)的相关性，得到权重

decoder阶段：将h加权求和，输入第i个时序中
$$
e_{ij} = a(s_{i-1}, h_j)\\
\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T}exp(e_{ik})}\\
c_i = \sum_{j=1}^{T}\alpha_{ij}h_j\\
s_i = f(s_{i-1},y_{i-1},c_i)
$$
![temporal](/temporal.png)

**改进：**

多种对齐方法
$$
h_T^Th_s\\
h_T^TW_ah_s\\
v_a^Ttanh(W_a[h_t;h_s])
$$
Local Attention:对齐考虑位置，假设正态分布
$$
a(s)=align(h_t,h_s)exp(-\frac{(s-p_t)^2}{2\sigma^2})
$$
**最终结果：**NMT机器翻译系统

**问题：**encoder-decoder模型只能是串行的，而且过于复杂，太慢

### 2.3 Self Attention，Transformer

**描述：**K,Q,V均是自身，本质是求待考虑位置与其他位置相关性

**公式：**
$$
a(q,k)=q^TWk\\
a(q,k)=q^Tk\\
a(q,k)=q^Tk/\sqrt(d_k)，最常用，其中d_k为数据维数\\
最终结果：Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt(d_k)})V
$$
几种技巧：

**Mask：**类似dropout，也就是无效化部分单词，用于decoder阶段，用于保证decoder阶段还是autoregressive的（因为decoder阶段的序列信息可能已经被严重破坏了）

**Multi-Head：**类似增加channel数，给V,K,Q各做不同的线性变换，然后求attention。本质是得到不同的子特征的相关性。
$$
head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)\\
MultiHead(Q,K,V)=Concat(head_1,....head_h)W^o
$$
![multihead](/multihead.png)

**Positional Encoding:**将输入数据编码上位置信息，以考虑被attention忽视的相对位置信息

#### **最终结果：Tranformer：**

**encoder阶段：**输入是x经过positional encoding的结果。网络是MultiHead Attention，LayerNorm， Feed Forward（全连接），LayerNorm组成的6层网络

**decoder阶段：**输入是encoder结果经过positional encoding的结果。网络是Masked MultiHead Attention， LayerNorm，MultiHead Attention，LayerNorm， Feed Forward（全连接），LayerNorm组成的6层网络

![transformer](/transformer.png)

**Feed Forward：**两层全连接，对每个位置进行操作（先变化成（batch*seq_len,fea_size)形式），类似1 * 1卷积

**Residual机制：**引入Resnet类似的直连机制，用于保留输入的位置等信息，防止梯度消失

### 2.4 GPT，BERT

#### **GPT：**

用迁移学习方法优化Transformer

**阶段1：**无监督预训练，任务是语言建模(为了完成这个任务，强行把transfomer改成单向的了)

**阶段2：**有监督优化，任务是你要实现的具体任务

阶段2实际采用**多任务学习**
$$
L_3(C) = L_2(C) + \lambda L_1(C)，其中L_2是目标任务，L_1是预训练的语言建模任务
$$

#### BERT：

双向的，进一步优化的Transformer

**阶段1：**无监督预训练

​	**任务1：**是Masked语言建模（完形填空），把k%的单词mask上预测。

​		问题：有监督优化阶段没有mask。

​		解决：80%用一个单词[MASK]代替，10%用随机单词代替，10%用	原来单词，让有无监督阶段接轨

​	**任务2：**预测下一个句子

**阶段2：**有监督优化

**相比GPT的三点改进：**

- ​	双向的，更符合self attention的双向特性
- ​	增加了segment特征，用了特征工程方法（原来只有单词和位置特征）
- ​	多任务学习，预训练2个任务

### 2.5 Spatial Attention

#### 2.5.1 Temporal Attention解决图片描述问题

**实质：求向量和特征每个位置的对齐**

先经过CNN提取特征a（196 * D），然后用global average pooling得到1 * D的特征，将这个特征线性变换，得到LSTM的记忆c0和用于attention的h0。**然后求h0与特征a每一个位置的相似度**，softmax得到权重，用这个权重作用在a上得到z1，再用z1得到新的h1。循环往复，不断用新的h和a每个位置求相似度，得到新的z，新的h。

![space_temp](/space_temp.png)

（其中y是用于训练的描述词，d是输出描述词，alpha是相似度）

进一步改进：直接用CNN-LSTM来完成任务

#### 2.5.2 图片的Self Attention

**实质：求特征图片每两个位置间相似度，也就是非局部卷积**
$$
先将特征图片分别做两个1*1卷积（类似不同的线性变换）\\
y_i = \frac{1}{C(x)}\sum_{\forall j}f(x_i,x_j)g(x_j)\\
g(x_j)=W_gx_j\\
f(x_i,x_j)=e^{x_i^Tx_j}(Gaussian)\\
f(x_i,x_j)=e^{\theta(x_i)^T\phi(x_j)}(Embedded Gaussian)\\
C(x) = \sum_{\forall j}f(x_i,x_j)正规化因子
$$
得到的结果可以被用来加权修改图片

![picture](/picture.png)

#### 2.5.3 其他

**Channel Attention：**用attention求得每个channel的重要性，加权更新channel

![channel](/channel.png)

**Class Activation Mapping：**在分类网络中，利用最终的分类结果（**最终softmax得到的权重**），加权修改图片，得到每个类别的重要程度

![class](/class.png)

改进：用梯度信息替代结果信息

## 3.RNN时空建模

RNN想要建模现实世界，一方面需要感知时空变化，另一方面需要记忆（sensory，短期，长期）。利用无监督学习和预测方法可以让RNN实现这一点。

**Convolutional LSTM:**

首先，LSTM本身就能实现短期记忆。通过把x，h等向量从单词改造成图片（压到一维），进行sequence to sequence预测（异构，encoder decoder参数共享）就能初步实现预测学习了。一种很好的方法是进行多任务学习---预测的同时重建原有序列，能重建原有序列说明模型不错，这样能更好优化模型。

![short_memory](/short_memory.png)

而为了更好地空间建模，把LSTM的所有全连接操作换成2D或者3D卷积就更好了。

**Spatiotemporal LSTM:**

之后，因为RNN网络的深度不是特别大，而卷积操作的感受野增加缓慢，RNN的深度不够，因此，我们需要增加卷积操作的深度，也就是空间记忆。这样，我们引入了**Zigzag机制**，也就是时刻t的最后一层传入时刻t+1的第一层。这样就能大大改进空间记忆。

![zigzag](/zigzag.png)

但是，因为RNN网络在时间很长的时候会梯度消失，长期记忆效果不太好。因此，需要引入**highway机制**

Cascaded ST-LSTM引入了GHU单元，是一种highway机制。这样网络可以通过门结构选择走慢路径（一层一层走，zigzag）还是走快路径，能够有效缓解梯度消失。

**Eidetic LSTM:**

现实世界中，很多数据是non-stationary，变化较快。要很好学习这种数据，必须有记忆机制。的这种网络引入了memory pool机制来记忆之前的状态，因为之前的状态多数有用，因此抛弃了forget gate，但是引入了recall gate来引入memory pool。

![eidetic](/eidetic.png)

这样就实现了三种记忆形式：

短期：3D卷积，实现了即时记忆

中期：Zigzag flow和LSTM，实现了时空记忆

长期：memory pool和recall gate实现了较长期记忆

**神经图灵机与电脑：**

实现长期记忆的方法，最直接的就是仿照传统计算机，用内存/外存实现数据存储。

神经图灵机引入了内存机制（全局共享），用LSTM模块作为控制器，用LSTM的门来控制读写。

![turing](/turing.png)

LSTM控制器会生成读key，写key，读vector三个数据（这个和经典LSTM的input，forget，output等类似）。用读key读取内存并且获取reading vector向下传播；用写key和写vector修改内存。

不过神经图灵机和计算机的读写不同，它的寻址，读，写都不是确定唯一的。寻址阶段，用key和各个内存的key做相似性比较，再做softmax得到权重；
$$
K[u,v] = \frac{u·v}{||u||·||v||}\\
w(i)=\frac{exp(\beta K[k,M(i)])}{\sum_j exp(\beta K[k,M(j)])}
$$
读的时候，根据权重，读取每个元素，加权求和作为最终结果
$$
r_t = \sum_i^R w(i)M(i)
$$
写的时候，根据权重来更新每个元素
$$
M_{erased}(i) = M_{old}(i)(1-w(i)e)\\
M_{new}(i) = M_{erased}(i)+w(i)a
$$
之后，还有Neural Computer（用冯诺依曼架构来进一步模仿计算机）和Memory Network（用外存代替内存）等其改进版，不过万变不离其宗，都是用LSTM的门结构作为控制器，然后尽可能模拟计算机的存储机制，以实现计算机的长期记忆。进而结合神经网络与计算机，做出一个“可自主学习”的人工智能。

## 4.整体思维导图

![mindmap](/mindmap.jpg)