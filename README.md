# 深度学习笔记总结

本repo是清华大学软件学院研究生课程《深度学习》的笔记和总结

## 知识点

**深度学习理论：**

- ​	泛化误差界
- ​	Rademacher复杂度
- ​	随机标签问题---Margin泛化误差理论
- ​	随机标签问题中，训练越快，网络效果越好---算法稳定性理论
- ​	过参数优化网络性能--过参数理论
- ​	对抗样本：对抗样本是什么，怎么生成对抗样本，对抗样本可迁移，怎么防御对抗样本

**深度学习基础：**

- ​	深度学习流程：前向传播，激活函数，softmax，损失函数，反向传播
- ​	深度学习技巧：初始化，各种Normalization，dropout，weight decay，学习率策略
- ​	优化算法：随机化的SGD+Momentum，自适应的Adam
- ​	MLP网络

**CNN：**

- ​	CNN基础：卷积层，池化层，stride/padding等，反向传播（转置卷积）
- ​	复杂CNN结构：3D卷积，图卷积
- ​	CNN网络架构：AlexNet--VGG--Inception--Resnet
- ​	CNN网络压缩：剪枝，压缩，分组卷积
- ​	CNN思考：平移不变性，局部连接性，数据增广

**CNN应用：**

- ​	人脸识别：人脸识别（封闭集，开放集---改进loss高内聚低耦合），人脸确认（三元组）
- ​	图像分割：encoder-decoder，特征融合
- ​	物体检测：RPN候选框生成，R-CNN识别+回归，ROIAlign进一步对齐，YOLO/SSD一步到位，FPN多尺度融合
- ​	风格迁移：风格图片（Crammer损失）+内容图片（Instance Norm），用encoder-decoder加速训练
- ​	视频：图片+光流，fast-slow融合
- ​	3D：View，Volumetric，点云。点云：PointNet（变换不变性，旋转不变性），点云重建的chamfer distance

**RNN：**

- ​	RNN基本单元：原版，LSTM, GRU
- ​	RNN基本结构：多层，双向，one to many，many to one，同构many to many， 异构many to many（encoder-decoder）
- ​	RNN反向传播：BPTT，梯度爆炸--Gradient Clipping，梯度消失--LSTM/GRU，技巧（weight/layer norm）
- ​	Attention：Self Attention，Temporal Attention，Attention在图片的应用（temporal attention图片描述，图片self attention，channel attention等）
- ​	Transformer：Transformer本体（MultiView Attention，Feed Forward，位置编码，直连，Layer Norm），GPT,BERT
- ​	RNN与时空建模：RNN+CNN，Zigzag与GHU，Eidetic RNN（memory pool），RNN图灵机（内存机制，LSTM门结构控制）

**RNN应用：**

- ​	核心：embedding+网络特征学习
- ​	自然语言处理：embedding（Wordvec，NCE正负例，Hierarchical Softmax），文本分类，摘要，问题回答，词标签
- ​	图：图embedding（deepwalk，unbiased deepwalk），图神经网络
- ​	推荐系统：传统方法，DNN，Tree-Based DNN

**GAN：**

​	GAN基础：基本流程，DCGAN，损失函数改进（Wassenstein Distance），评价（IS,FID)

​	模式坍塌：主要解决方法（SN-GAN，Gradient Penalty），带标签解决（Conditional），有配对解决

​	GAN高级网络：self attention，BigGAN

**RL：**

- ​	RL定义：马尔科夫决策过程MDP
- ​	基于Q-value的方法-DQN：Monte Carlo方法，Temporal Difference方法，Q-value改进（记忆重放，double网络，拆分）
- ​	基于策略的方法-Policy Gradient：基本流程，Actor-Critic改进，比DQN的优势
- ​	AlphaGo，AlphaGo Zero：AlphaGo=MCTS（蒙特卡洛树搜索）+策略网络（监督学习+强化学习）+值网络，AlphaGo Zero一个网络，全靠生成数据，大幅改进
- ​	RL的局限性还很大

## 思路总结

**大力出奇迹：**网络越深越好（只要你不梯度消失，不过拟合），越宽越好，batch越大越好（只要你有卡），但是最好还是**simpler is better**

**参数共享：**减小参数量，提高泛化性，利用局部相关性等。

CNN在图片各个区域参数共享; RNN在所有时间参数共享；图卷积网络同一层都参数共享；

**损失函数：**分类的cross-entrophy，回归的L2，要限制网络为某种性质（比如正交矩阵）||WW^T-I||,encoder-decoder重建有重建损失函数||重建的-需要的||_1(cycle loss是特殊形式)。无监督的NCE损失函数（正负例，用于各种embedding），Hierarchical Softmax等，度量分布相似性的推土机距离等。其余具体任务设置具体损失函数，比如点云重建，度量两个点云相关性的chamfer distance等。

**各种Norm：**目的基本都是让网络更smooth，训练更快，防止过拟合等。

BatchNorm对batch平均，layer norm对channel平均，weight norm直接修改参数网络范数。Instance Norm对图片平均，目的是白化，消除内容图片的对比度影响。spectral norm是网络除以最大|特征值|，控制lipchitz系数来防止模式坍塌

**随机化：**目的都是引入随机性，避免陷入贪心的局部最优解。

优化中SGD引入随机性，改进SGLD引入随机的noise。RL中Policy Gradient引入随机性以实现E-greedy，探索利用的均衡。

**融合：**将多尺度的数据融合在一起，便于学习/利用多个层次的特征，网络拓宽。

Inception网络不同感受野/尺度特征融合，Resnext也是类似的思路。RNN图像分割需要低级特征（细节）和高级特征（骨架）的融合。CNN目标检测需要不同尺度特征融合（FPN,SSD）（高级特征检测大物体，低级特征检测小物体）。视频识别中fast-slow融合，slow为主fast为辅。

**拆分：**可以将CNN网络进行拆分，实现压缩，以及增加深度提高效果

7*7卷积=3个 3 * 3 卷积；3 * 3卷积= 1 * 3 + 3 * 1卷积；正常卷积=分组/逐channel卷积 + 1*1卷积融合channel

**多任务学习：**多个不同但是高度相关的任务一起学习，一方面相互促进改进网络效果，另一方面实现端到端简化训练流程

CNN目标检测中，物体识别和边框回归优化多任务学习；CNN实例检测，图像分割与目标检测多任务学习；GPT的第二阶段，预训练任务和finetune多任务学习；摘要学习的BertSUM，关键句提取和摘要生成多任务学习；Conditional GAN，生成和分类多任务学习；

**Actor-Critic：**用于RL，一个网络Critic评判好不好，另一个网络Actor根据Critic结果来有倾向的选择步骤

DQN做Critic，Policy Gradient做Actor；Alphago/Zero中，Value网络做critic，policy网络做Actor

**记忆：**让神经网络拥有一定的记忆，这样能够处理复杂的时空建模问题，是强人工智能必须的

LSTM，记忆机制和遗忘门实现短期记忆；Memory Pool/神经图灵机模拟内存实现长期记忆；强化学习DQN，记忆重放来防止训练数据相关性太强，利用之前学到的经验

**树型结构：**将神经网络和二叉树等树形数据结构结合，加速

RNN的机器翻译，使用beam search树搜索。使用Hierarchical Softmax解决embedding时softmax分母计算太复杂的问题；树形DNN+beam search解决推荐算法的商品过多，复杂度过高问题；蒙特卡洛树搜索+Value Network与线性回归快速下棋，解决AlphaGo中下一个落子位置众多的问题

**压缩：**要想将神经网络向手机，物联网等终端移植，需要在保证准确率基本不变的前提下减少参数大小，提高计算速度

剪枝（值接近0就剪掉）；拆分与组卷积；K-means近似与huffman编码

