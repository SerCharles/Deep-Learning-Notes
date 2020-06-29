---
typora-root-url: ./
---

# GAN

## 1.GAN基础

### 1.1 Generative Modeling

假设所有的数据来自一个分布，那么一个重要任务就是根据部分数据，得到这个分布。但是由于分布非常复杂，很难直接写出来，因此采用另外一种手段：用能表示出来的分布来近似这个实际分布，使得对于某个统计距离D，两个分布距离最小。
$$
\theta = min_{\theta} D(p_{\theta}(x),P_{x}(x))\\
得到的分布P_{\theta}可以用来生成和P_{x}这个实际分布几乎同分布的数据，也就是生成和数据图片近似的图片
$$

### 1.2 GAN

思路：两个网络，一个辨别器，一个生成器相互对抗。

辨别器用于区分真实数据和生成的数据，越强越好（很多改进都是加强这个）

生成器用于将Gauss分布的数据转化成和真实数据近似的数据，骗过辨别器

![GAN](/GAN.png)

整体的损失函数：
$$
min_{G}max_{D}[E_{x~P_{data}}logD(x) + E_{z~P_z}log(1-D(G(z)))]
$$
具体流程：

连续训练k次辨别器，损失函数如下，使用梯度上升：
$$
max_{D}[E_{x~P_{data}}logD(x) + E_{z~P_z}log(1-D(G(z)))]
$$
训练1次生成器，损失函数如下，使用梯度上升：
$$
max_{G}[E_{z~P_z}log(D(G(z)))]
$$
原因：

- ​	辨别器是熵减少过程，较难；生成器是熵增加过程，较容易；因此训练判别器k次，更多次，以平衡二者
- ​	如果使用log(1-D(G(z)))训练生成器，当D(G(z))接近0时，也就是一开始模型较差的时候，梯度较小；反之，D(G(z))接近1，也就是模型最后较好的时候，梯度较大。这样不利于快速训练和收敛。取相反的损失函数，则能够让一开始梯度较大，最后梯度较小。

![gradient](/gradient.png)

### 1.3 CGAN

对于图片任务，GAN需要辨别器是CNN分类网络，生成器是反卷积生成网络。

具体：

- ​	生成器使用反卷积网络，以从小数据生成图片；辨别器使用带stride的卷积网络分类
- ​	和CNN的网络一样，使用BatchNorm，除辨别器最后一层外不使用全连接层
- ​	生成器使用RELU激活函数（除最后一层），最后一层使用tanh函数，因为这种有界激活函数对生成同样有界的图片数据好。辨别器使用LeakyRELU激活函数，因为RELU<0时会梯度消失，让依赖D梯度的生成器训练无法进行
- ​	为了防止数据损失，不用任何pooling

### 1.4 改进损失函数

为什么？对于部分分布的情况，交叉熵损失函数会导致梯度消失

**常用改进：Wasserstein Distance**
$$
W_p(u,v)=(inf_{P\in \pi(u,v)\int\int D(x,y)^p P(dx,dy)})^{1/p}\\
其中，u,v是待比较的两个分布，\pi(u,v)是两者的联合分布，它使得后面的距离函数最小
$$
直观理解：推土机距离---把同样体积的土堆u填到土堆v，需要最少的运输量

![wasserstein](/wasserstein.png)

实际情况：限制Lipchitz常数在1以内（梯度不能太大，符合神经网络的情况）
$$
min_{G}max_{D}(E_{x: P}D(x)-E_{y: G(N(0,I))}D(y))
$$
此时D是一个critic--评价好不好

**更广义的改进：F-distance**
$$
d_{F,\phi}(u,v)=sup_{D\in F}|E_{x:u}\phi(D(x)) + E_{x:v}\phi(1-D(x))|-2\phi(1/2)\\
其中\phi为凹函数\\
性质：当u=v时，根据凹函数性质，损失函数D_{F}=0，此时D(x)=1/2\\
使用这种损失函数，能保证GAN的泛化性\\
|D_{F,\phi}(u',v')-D_{F,\phi}(u,v)|\leq \epsilon
$$
Wassenstein距离就是F距离在phi=1的特殊情况

**对于损失函数：最好找一个最大的，使得只要实际分布和生成分布有一点区别，损失函数就很大**

### 1.5 评价指标

**IS：**

给定待模仿的单一数据x，在GAN中生成得到图片，放进Inception网络里分类，结果为P(y|x)，这种单一数据的结果，分类越清晰越单一越好。

对于各种类别的x，将其对应的GAN结果图片放到Inception网络里分类，结果为P(y)，这种各种类型数据分类结果，越多样性越均匀越好。

因此用这两个的KL散度来度量
$$
IS = exp(E_{x:G}[KL(p(y|x)||p(y))])
$$
**IS越大越好**

**FID:**

同时将待模仿数据x和生成数据g扔到Inception网络里，假设对于网络某一层的结果，待模仿的数据x的结果遵循高斯分布，那么用GAN生成的数据g的结果也应该符合类似的高斯分布。对于一大堆x，g，统计出这些结果（**layer features**）的高斯分布，用Wasserstein距离来评价即可
$$
真实数据Gaussian:(\mu_x,\Sigma_x),生成数据Gaussian:(\mu_g,\Sigma_g)\\
FID(x,g) = ||\mu_x-\mu_g||^2_2 + Tr(\Sigma_x + \Sigma_g - 2((\Sigma_x \Sigma_g)^{1/2}))
$$
**FID越小越好**

## 2.模式坍塌避免

### 2.1 模式坍塌

理想的GAN结果：真实分布和生成分布完全匹配

不理想的GAN结果：部分情况下，真实分布和生成分布完全匹配；其他情况下，生成分布没有对应的生成分布。也就是生成分布只是拟合了部分真实分布

![collapse](/collapse.png)

### 2.2 一般方法：SN-GAN

**观察1：**模式坍塌情况下，通常拟合较好的部分，D取值为1/2，Lipchitz较小；拟合较差的部分，D取值为1，Lipchitz较小；两者之间：D取值在1/2-1之间，Lipchitz较大。

**基本思路：**控制Lipchitz系数

控制方法1：**Gradient Penalty**，用梯度的定义控制，只能控制局部，不够好
$$
(\frac{|D(x)-D(x')|}{|x-x'|}-1)^2
$$
控制方法2：限制判别器的W为正交矩阵，能全局控制Lipchitz系数（特征值为1），但是会破坏判别能力，不够好
$$
||W^TW-I||_F
$$
控制方法3：**矩阵的Lipchitz系数=其最大|特征值|**

而对于整个网络：
$$
f=(W_{L+1}(a_L(W_L(...a_1(W_1x))))\\
||f||_{lip} \leq ||W_{L+1}||_{lip} * ||a_L||_{lip} * ... * ||W_1||_{lip}
$$
而激活函数都是ReLU，lipchitz系数都是1

**所以，控制整个网络lipchitz系数-->控制每个矩阵的lipchitz系数**

解决方法：**Spectral Normalization，**将每个矩阵都除以自己最大|特征值|
$$
W_{SN} = W/\sigma(W)
$$

### 2.3 带标签的方法：C-GAN

很简单的思路：之前的模式坍塌，一般是**某些类生成的很好，某些类生成的不好**。这样只要有类别标签，就能辅助消除模式坍塌

**CGAN：**在生成器，判别器中，同时输入标签y

![CGAN](/CGAN.png)

**ACGAN：**通过在判别器中做**多任务学习（判别和分类）**，增强判别器

![ACGAN](/ACGAN.png)

**Projection Discrimiator：**也是多任务学习，另一个任务是让学到的特征与标签embedding尽量接近
$$
D(x,y)=\theta(\phi(x)) + e_y·\phi(x)
$$
![project](/project.png)

**Class-Condition BN：**对于每一类，Batch Normalization的r和b不同

### 2.4 带配对的方法

**图片重建任务：**给一张input输出一张output(style transfer)：

![paired](/paired.png)

增加Reconstruction loss，保证用生成器生成的图片和正常该生成的接近
$$
|G(x_i,z_i)-y_i|_1
$$
**视频重建任务：**类似，不过逐帧操作/类似视频处理，引入光流

**图片重建任务，但是是两个集合间的transfer，没有一一对应关系：**

![unpaired](/unpaired.png)

**使用两个GAN，**一个将集合X变换到Y，尽量和Y接近；另一个把集合Y变换到X，尽量和X接近

![cyclegan](/cyclegan.png)

**增加cycle loss**(也是一种reconstruction loss），保证X变换成Y再变回去和X一样，Y同理
$$
L_{cyc}=||X-F(G(x))||_1 + ||y-G(F(y))||_1
$$
最终loss：
$$
L = L_{GAN}(G,D_Y) + L_{GAN}(F,D_X)+\lambda L_{cyc}
$$

## 3.GAN的改进

#### **SAGAN：**

引入Attention，用图片的self-attention来捕捉待模仿图片的一些相关性（比如鼻子和眼睛的相对位置等等），给GAN网络添加更强的约束

#### BigGAN：**大力出奇迹**

- ​		大Batch--2048（普遍有利于提高效果）
- ​		Truncated Noise：对于用来生成图片的噪声z，把极端的z(||z|| > R)扔掉，虽然会损失图片多样性，但是能提高图片质量
- ​		模式坍塌：用Gradient Penalty和SN结合，同时用eigenvalue clip来减小最大特征值带来的波动，让训练更平稳

$$
W = W - max(0,\sigma_0 - \sigma_{clamp})v_0u_0^T，其中\sigma_{clamp}是超参数，这就是eigenvalue clip
$$

#### Progressive GAN：

从小尺度图片不断增加图片尺度，结合双线性插值和反卷积网络，获得高清图片

![progressive](/progressive.png)

#### StyleGAN：

模仿风格迁移，将待模仿样本（风格图片）和位置噪声（内容图片）结合，生成细节丰富的人脸

## 4.对抗样本

### 4.1 定义

对于一个图片分类网络，给图片加上一个很小的（不影响视觉）的噪声，就能扰动网络,使得分类错误
$$
max_{x + \delta}J(\theta, x + \delta, y), s.t ||\delta||_{\infty} \leq \epsilon\\
解得\delta = \epsilon sign(\nabla_xJ(\theta,x,y))
$$
![adv](/adv.png)

产生原因：当一个sample靠近分类边界，而非线性不够强（RELU等），就容易再附近产生对抗样本

![example](/example.png)

### 4.2 攻击方法

#### **白盒攻击：**

已知网络参数，发动攻击

**损失函数：**
$$
min_{\delta}J(\theta, x + \delta, y'),有目标，把y混淆成y'\\
max_{x + \delta}J(\theta, x + \delta, y)，无目标，只是混淆\\
$$
**具体方法：**

**FGSM：**最高效
$$
\delta = \epsilon sign(\nabla_xJ(\theta,x,y))
$$
**PGD：**迭代攻击

![PGD](/PGD.png)

#### 黑盒攻击：

学习一个对图片的变换即可（比如旋转，增加遮挡等，利用CNN网络平移不变性不好等方法）

![attack](/attack.png)

#### 性质：对抗样本攻击可迁移，这样，随意用一个常见网络（Resnet）来找到对抗样本，就可以攻击很多网络

### 4.3 防御方法

**最简单：生成对抗样本，加入训练集**

**Unbiased Adversial Defense：**

​	将对抗样本防御看做一个对抗问题
$$
min_{\theta}E_{(x,y)~in~P}max_{\delta \in \Delta}J(\theta, x + \delta, y)
$$
​	

​	得到损失函数
$$
min_{\theta}E_{(x,y)~in~P}\alpha J(\theta, x , y) + (1-\alpha)max_{\delta \in \Delta}J(\theta, x + \delta, y)
$$
​	用PGD得到高质量对抗样本，来训练

**TRADE：**

​	一般的准确率和对抗准确率有trade off,因此构造损失函数找到他们的最佳平衡点
$$
min_{\theta}E_{(x,y)~in~P}[J(f_{\theta}(x) , y) + sup_{\delta \in \Delta}J(f_{\theta}(x), f_{theta}(x + \delta)/\lambda)]
$$
**进一步改进：生成对抗样本的值域**

## 5.整体思维导图

![mindmap](/mindmap.jpg)