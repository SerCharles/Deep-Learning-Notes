---
typora-root-url: ./
---

# Theory

## 1.统计学习理论

问题引入：当模型复杂的时候，测试准确率不断提高，但是训练准确率先提高后降低，过拟合

![overfit](/overfit.png)

### 1.1 泛化误差界

模型h是在一个分布中IID采样训练出来的

**期望误差：**  **测试误差**，随机在这个分布取样放到h中，损失函数期望
$$
\epsilon(h) = E_{(x,y):D(x,y)}l(h(x),y)
$$
**经验误差： ** **训练误差**，一般是期望误差的无偏估计
$$
\epsilon_{D_n}(h)=\frac{1}{n}\sum_{i=1}^{n}l(h(x_i),y_i)
$$
**泛化误差：**|经验误差-期望误差|，是一个随机变量
$$
|\epsilon_{D_n}(h)-\epsilon(h)|
$$
实际上，有多种可能的模型，我们无法保证每个模型的泛化误差都被控制住，只能保证绝大多数模型都被控制住，也就是控制不住的模型不超过很小的数目
$$
P(sup_{h \in H}|\epsilon_{D_n}(h)-\epsilon(h)| \geq \epsilon) \leq \delta
$$
**泛化误差的控制：**最简单的公式

​	
$$
err_{test}(f) = err_{train}(f) + O(\sqrt{\frac{plog_2r}{n}})\\
r是每个参数取值个数，p是模型参数，n是训练样本个数\\
更简单：err_{test}(f) = err_{train}(f) + \sqrt{\frac{complexity}{n}}\\
$$
**训练样本越多越好，模型也要尽量简单**

### 1.2 Rademacher复杂度

对于每个样本空间，有2^n种对分方法

对于每种对分方法，可以把一边全设置成正例，一边全设置成负例，然后用模型去分类，得到Rademacher复杂度
$$
\frac{1}{2^n}\sum_y sup\frac{1}{n}\sum_{i=1}^ny_ih(x_1)\\
y_i：第i个样本是正例就是1，否则-1；h(x_i):第i个样本分为正例就是1，否则-1\\
右面的y_ih(x_i)求和部分最多为1，代表全部分对。这是对所有分类结果的平均\\
$$
更形式化：

![rademacher](/rademacher.png)

**度量泛化误差界：**
$$
err_{test}(f) = err_{train}(f) + R_n(F) + \sqrt{\frac{log2/\delta}{2n}}\\
第三项很小，第二项就是Rademacher复杂度\\
R_n(F^L) \leq \sqrt{\frac{C(x)}{n}}(\rho M)^L\\
其中C(x)代表复杂度，\rho是激活函数lipchitz（1），L是模型深度，n是样本个数\\
$$
**样本个数大更好。复杂度越高越不好，模型深度越大越不好（指数正比），而这和深度学习层数越多越好矛盾，统计学习理论无法解释深度学习问题。**

### 1.3 统计学习理论无法解释深度学习问题



## 2.深度学习理论

### 2.1 Margin泛化误差

**现象：**用有noise的标签来训练深度网络，此时网络复杂度，数据n都没变，训练准确率都很高，但是测试准确率随着noise增加下降了，不符合统计学习理论之前得到的结论

![random](/../../../../2020%20Spring/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E6%80%BB%E7%BB%93/Part1-Theory/random.png)

**解决：**

可能原因：深度学习的网络实际可达函数空间远小于整个假设空间

思路：引入margin---对一个样本，距离最近的其他类的距离---margin越大分的越确信

![margin](/margin.png)

此时有：
$$
err_{test}(f) = err_{train}(f) + 2/\gamma R_n(F) + 3\sqrt{\frac{log1/\delta}{2n}}\\
当\gamma增加，也就是margin增大的时候，泛化误差界减小了，这可以解释深度学习的情况
$$
进一步推导，通过控制谱范数（最大特征值）也能控制泛化误差界

![sn](/sn.png)

压缩网络也是一种方法，通过对矩阵做低秩压缩（奇异值分解等），使得函数空间变小，也能控制泛化误差界

![compress](/compress.png)

### 2.2 算法稳定性

**现象：**用真实label比带noise的label快，泛化误差界小--训练越快泛化越好

![fastgood](/fastgood.png)

**解决：**
$$
sup_x|f(x)-f'(x)|\leq \epsilon_{stability}，用这个来控制泛化误差界\\
\epsilon_{test} \leq \epsilon_{train} + \epsilon_{stability}\\
$$
**凸函数：**


$$
\epsilon_{stability} \leq \frac{2\rho^2}{n} \eta T\\
其中\rho 为lipchitz，n为样本数量，T为训练轮数\\
这个是合理的，训练轮数多，慢，不好；lipchitz大，不收敛，不好
$$
**一般情形：**

（用的是SGLD优化函数，通过加一个noise，大小与B成反比，来优化）
$$
\epsilon_{stability} \leq \frac{2\rho M}{n}(\beta \sum_{k=1}^{T}\eta_k)^{1/2}\\
其中\eta_k是第k个epoch学习率，\beta是优化函数噪声等级\\
这个也是合理的，学习率越大越不收敛，噪声越大越慢，lipchitz越大越难优化
$$

### 2.3 过参数

**现象：**用过参数（网络很宽，参数比数据都多）能优化网络

![overparam](/overparam.png)

**解决：**2层网络引入新的泛化误差

![over1](/over1.png)

（初始化部分还能解释迁移学习，迁移学习可能使得泛化误差界下降）

**现象：**Teacher-Student网络，Teacher宽度50,50宽度的学生学不好，100宽度的学生学得很好

![teacherstudent](/teacherstudent.png)

**解决：**2层网络，将网络看成Kernel，将梯度下降连续化，

有当宽度足够大时，Kernel H(0)接近H无穷

而在H是H无穷的时候，得到以下结论，说明宽度增加能够大大改善网络性能（过参数也是如此）
$$
||y - f_t(x)||^2 \leq e^{-\lambda_0t}||y-f_0(x)||^2
$$

## 3.整体思维导图

![mindmap](/mindmap.jpg)

