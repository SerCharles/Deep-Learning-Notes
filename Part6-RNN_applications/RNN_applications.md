---
typora-root-url: ./
---

# RNN应用

## 1.NLP

### 1.1 基本框架

Word Embedding词处理 --- RNN/BERT模型提取特征 -- 根据具体问题构造具体网络

### 1.2 Word Embedding

**Word2Vec：**

​	One hot encoding： 向量维数高而且稀疏，计算慢；无法度量词之间的关系

​	需要用MLP对词one hot encoding进行映射，降低向量维数，稠密化，而且能度量词的关系

- ​	Bag of words：用周围词预测中心词
- ​	Skip-Gram：用中心词预测周围词（独立性假设：假设预测每个词这件事相互独立），主要使用这个

$$
w_c:中心词，w_o:待预测词\\
v_c = Vx~(x为中心词one~hot~encoding,v_c为embedding)\\
u_o = Uy_o~(y_o为周围某个词预测结果的one~hot~encoding,u_o为embedding)\\
P(w_o|w_c) = \frac{exp(u_o^Tv_c)}{\sum_{w \in Vocab}exp(u_w^Tv_c)}~(softmax)
$$

**NCE：**

​	用softmax：词表很大，计算分母太慢

​	解决：NCE近似softmax（正例-负例模式）
$$
log\sigma(u_o^Tv_c)+\sum_{k=1}^{K}log\sigma(-u_K^Tv_c)\\
其中u_o为正例embedding，也就是上下文出现的正确结果\\
u_K为负例embedding，负例是从词表分布的变形P_n(w)中随机筛选的（K为几十上百，远小于几万几十万的词表）\\
P_n(w)=U(w)^{3/4}/Z，其中U(w)为词表分布
$$
**Hierarchical Softmax：**

​	用二叉树来表示词表，每个叶子结点是词，非叶子结点是词高级特征
$$
p(w|w_I) = \prod \sigma(f · u_{n(w,j)^T}v_{w_I})，其中如果是左孩子，f=1，否则f=-1\\
概率=二叉树根节点到最后一个非叶子结点的每个高级特征，与目标词的相似度的乘积\\
有\sum_k p(w_k|w_I) = 1（sigmoid性质），所以可以替代softmax，还可以大大降低复杂度
$$
![hier](/hier.png)

**Subword Information：**

​	引入词根词缀等词的局部信息，一个词的embedding包括词本身和词局部信息的embedding，需要hierarchical softmax

**Glove：**

​	需要构造co-occurrence matrix X，代表词之间的共同出现关系，是全局统计信息。\\
$$
Q_{ij}=exp(u_j^Tv_i)\\
J= \sum_{i=1}^{W}\sum_{j=1}^W f(X_{ij})(logQ_{ij} - logX_{ij})^2\\
目标：让全局统计信息与词本身相似性足够吻合
$$

### 1.3 具体问题处理

**文本分类：**

最后一层网络改成softmax分类器即可

**摘要：**

- 提取关键句：转化成分类问题

- 自己写摘要：转换成机器翻译问题，用encoder-decoder架构

  评估：BLEU,ROUGE

  Copying：因为转换成机器翻译生成摘要，可能难以注意到低频重要词汇，所以要copy（用attention找到文中重要词汇）。
  $$
  P_{final}(w) = P_{gen}P_{vocab}(w) + (1-P_{gen})\sum_{i:w_i=w}a_i^t
  $$
  BERTSUM：最终的网络需要提取关键句与生成摘要，用**多任务学习**方法优化问题

**问题回答：**

​	基本思路：用RNN/BERT从d（背景信息）和q（问题）中提取回答，与标准答案a对比
$$
P(a|q,d) = \lambda exp(W(a)g(q,d))
$$
​	Open-Domain:背景信息需要网上获取(wiki等)，也就是获取和q接近，而且包含答案a的d

​	![question](/question.png)

​	SQUAD任务：回答包括开始，结束位置

​	解决方法：SpanBERT：在经过BERT后，把一个span的**开始，结束，中间关键信息**组合起来进行操作

**词标签（标注每个词的类别，比如person，location等）：**

​	基本思路：分类问题

​	思路：

- ​		embedding阶段包括字母，词，全局信息
- ​		用传统方法（CRF）代替简单的softmax分类网络作为最后的网络

## 2.图

### 2.1 基本思路

与自然语言处理类似，Node Embedding（给图每个点学习到合理的值） -- 图学习

### 2.2 Node Embedding

**目标：**给图每个点学习到合理的值，使得他们的相似性可以用内积度量
$$
similarity(u,v) = z_v^Tz_u
$$
**Adjacency-Based：**只考虑直接相邻节点
$$
L= \sum||z_u^Tz_v - A_{u,v}||^2，A为邻接矩阵
$$
**Multi-Hop：**考虑间接相邻节点
$$
S_{u,v}用于度量u与v邻居的重叠程度，比如Jaccard~Score\\
J(A,B) = \frac{|A \bigcap B|}{|A \bigcup B|}\\
L= \sum||z_u^Tz_v - S_{u,v}||^2
$$
**Deep-Walk：最常用的方法**

用随机游走机制更好度量相似性，随机游走会给每个节点随机生成一个邻域，**保证邻域内相似性尽量大，邻域外尽量小**。
$$
L = \sum_{u \in V} \sum_{v \in N_{R}(u)}-log(\frac{exp(z_u^Tz_v)}{\sum_{n \in V}exp(z_u^Tz_n)})
$$
问题：节点数太大，代价太大

解决：NCE，Hierarchical Softmax
$$
NCE:L = log(\sigma(z_U^Tz_v) - \sum_{i=1}^k log\sigma(z_u^Tz_{n_i}))
$$
**Biased Walks：**

​	经典的随机游走：到任何相邻节点等概率

​	改进：不同类型节点概率不同,走回原来节点概率 * =1/p，走到和原来有边相连概率不变，走到和原来无边相连概率 * = 1/q

![biased](/biased.png)

- ​	q较大：倾向于走自己周围的节点--**广度优先**--倾向于找到自己周围的节点
- ​	q较小：倾向于走没去过的节点--**深度优先**--倾向于找到距离较远但是结构相似的节点

![DFSBFS](/DFSBFS.png)

（2个图似乎反了，q=2是图2,0.5是图1）

### 2.3 Graph Learning

#### GNN：

类似图卷积，**用周围节点的平均值来代表自身节点的值**。同一层全局参数共享
$$
h_v^k = \sigma(W_k\sum_{u \in N(v)}\frac{h_u^{k-1}}{N(v)} + B_kh_v^{k-1})
$$
![GNN](/GNN.png)

训练：既可以有监督训练（比如节点分类），**也可以用Random walks来无监督训练**

性质：有**推理能力**，因为**参数共享**，可以轻易将网络用到新的图，或者图的新节点中

#### 改进：

**GraphSAGE：**改进了求值方法，从平均变成加权平均（类似attention思路，但是没做softmax）

**Graph Attention：**用attention机制给每个邻居加权平均

**Gated GNN：**引入RNN来加深图网络

- ​	各层也参数共享
- ​	**把一个节点的所有邻居标号，当做一个序列**，用GRU来建模

$$
m_v^k = W\sum_{u \in N(v)}h_u^{k-1}\\
h_v^k = GRU(h_v^{k-1},m_v^k)(输入同时考虑邻居(在GRU上一层的值的）均值与GRU上一层)
$$

![gated](/gated.png)

## 3.Recommendation

### 3.1 传统方法

**编码：One-Hot encoding**

Logistic 回归：线性模型，难以捕捉非线性特征，依赖特征工程

Factorization Machine：引入二次线性特征，用矩阵低秩变换降维到O(NK)

### 3.2 深度学习方法

**核心：embedding+神经网络（MLP,Attention）特征学习，损失函数仍然用正负例近似**

DNN：最后用logistic损失函数

DIN：引入Attention机制

#### Tree-Based：

思想：将树和深度学习模型结合，降低复杂度到O(logN)

基本操作：构造一个树，树的叶子结点是商品，非叶子结点是抽象的商品种类。推荐时自顶向下做Beam Search+剪枝找到概率最大的几个商品，降低复杂度到logN。

![TDM](/TDM.png)

概率计算：
$$
p^{(j(n|u)} = \frac{max_{n_c \in n的子节点}p^{(j+1)}(n_c|u)}{\alpha^{(j)}}\\
父亲节点的概率由子节点最大的那个概率决定\\
$$
损失函数计算：NCE正负例近似，不过需要所有结点都取正负例--正例是已经点击的商品和其ancestor，负例随机生成。

![tree](/tree.png)

## 4.整体思维导图

