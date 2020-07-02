---
typora-root-url: ./
---

# RL

## 1.基础：马尔科夫决策过程 MDP

**强化学习本质：**Agent和Environment交互的过程，Agent在某个state，然后执行某个action（有下棋这种有限的，也有开车这种无限的）后，转移到下个状态并且获得reward（有Atari这种即时的，也有围棋这种非即时的）。强化学习就是要最大化整体reward。因为状态转移和reward可能是随机的，因此问题变成最大化整体reward期望。

![RL](/RL.png)

**形式化定义--MDP：**

MDP是一个五元组<S,P,A,R,r>

S是状态集合，A是动作集合，R是给定状态和动作(s,a)后得到的奖励，P是给定状态和动作(s,a)后状态转移的概率分布，r是discount factor---奖励的衰减系数

MDP的目标是找到一个policy，也就是对于每个状态，应该采用的动作，这个动作能使得整体奖励的期望最大
$$
\pi(a|s)\\
有确定的a = \pi(s)和随机的\pi(a|s) = P(a|s)两种形式
$$

$$
Objective：整体奖励的期望\\
E[\sum_{t\geq 0}\gamma^t r_t|\pi]\\
Value~Function：给定初始状态s后，整体奖励的期望\\
V^\pi(s) = E_{s_{1:T},a_{1:T}}[\sum_{t\geq 0}\gamma^t r_t|s_0 = s, \pi]\\
Q-Value~Function：给定初始状态s和初始动作a后，整体奖励的期望\\
Q^\pi(s,a) = E_{s_{1:T},a_{1:T}}[\sum_{t\geq 0}\gamma^t r_t|s_0 = s,a_0=a, \pi]\\
Bellman~Equation:Q-Value的递推公式\\
Q^\pi(s,a) = E_{s',a'}[r + \gamma Q^\pi(s',a')|s,a,\pi]\\
最优Q函数的递推公式\\
Q^*(s,a) = E_{s'}[r + \gamma max_{a'} Q^*(s',a')|s,a]
$$

![MDP](/MDP.png)

## 2.基于Value的方法：DQN

用Q-network来近似最优Q函数

**Monte Carlo方法：**

把s和a当做输入，用Monte-Carlo方法得到labels（Q(s,a,w)）进行训练

具体方法：不断sample得到若干s，a序列，然后对于每个状态-动作(s,a)，用 总奖励/总次数R(s,a)/N(s,a)来估计标签Q

![network](/network.png)

**Temporal Difference：**

本质：填表算法，DP

建立一个s-a表，表初始状态Q0(s,a)都是0，之后动态规划更新表
$$
Q_{t+1}(s_{t},a_{t})=Q_{t}(s_{t},a_{t}) + \eta(R_{t+1} + \gamma max_{a}Q_t(s_{t+1},a) - Q_{t}(s_{t},a_{t}))
$$
一般情况：损失函数为
$$
l = (r + \gamma max_{a'}Q^*(s',a',w) - Q(s,a,w))^2\\
$$
w是神经网络，可以理解为广义的s-a表，用梯度下降更新神经网络参数w即可

**用于深度学习的问题和改进：**

问题：数据相关性高，non-stationarity

改进：

- 引入参数E，用E-greedy策略来平衡探索和利用（前期优先探索，后期优先利用）
- 引入记忆重放机制（把运行中遇到的情况存储起来，定期调用），目标是消除数据间相关性
- 固定记忆重放的参数w’以消除non-stationarity

$$
记忆重放：
l = (r + \gamma max_{a'}Q(s',a',w) - Q(s,a,w))^2
$$

**DQN的一些具体改进：**

Double DQN：用当前网络w选择动作，老网络w‘来评估动作
$$
l = (r + \gamma Q(s', argmax_{a'}Q(s',a',w),w') - Q(s,a,w))^2\\
$$
Prioritized replay：记忆重放的时候有优先级

Dueling network：把神经网络拆成两部分，减小方差
$$
Q(s,a) = V(s,v) + A(s,a,w)\\
V是value function，和动作无关；A是advantage function
$$

## 3.基于Policy的方法：Policy Gradient

流程：

While True(对时间迭代):

​	根据当前策略选择action，然后得到对应的reward以及是否结束，存起来

​	如果结束：用整个episode的数据，代入policy network来训练网络

![policy](/policy.png)


$$
损失函数：\sum_t J(\tau;\theta)logp(a_t|s_t)\\
梯度：E[J(\tau;\theta)\triangledown_\theta logp(a_t|s_t;\theta)]\\
其中J为一个episode的reward，p为对应的概率
$$
相比Q-learning，方差较大，但是因为有随机性（sample一个轨迹）可以进行探索-利用权衡，更关注整体而不是局部得失，也能学到一个策略概率而不是确定性的策略。Policy Gradient收敛性有保障，但是Q-learning无。

**优化：结合Q-learning和Gradient policy---Critic-Actor网络**

用Q(s，a，w)替代reward函数J，能够有效减小方差

流程：sample出一个action，用Q代替reward函数去更新策略，然后去更新Q，如此迭代（从一整个轨迹变成单步了）

![actorcritic](/actorcritic.png)

## 4.AlphaGo，AlphaGo Zero

#### AlphaGo

AlphaGo = MTCS + Policy Network（Actor） + Value Network（Critic）

**Policy Network：**主网络，用于下棋

​	用棋谱数据监督学习训练，用了人工特征，CNN网络，得到p_sigma

​	然后用强化学习方法训练,迁移学习，以p_sigma为初始化数据，用policy gradient算法，用self play（自己和监督学习的AI对下）数据训练，得到p_p

**Value Network：**用于评估当前局面的胜负，**能减少层数**

​	用之前的self_play数据进行监督学习，也是CNN网络

**MTCS（蒙特卡洛树搜索）：**用于选择下棋路径

​	选择：
$$
a_t = argmax_a(Q(s_t,a) + u(s_t,a))\\
u(s,a) = \frac{P(s,a)}{1+N(s,a)}
其中Q是该节点reward，P是监督学习policy~network的结果p_\sigma，N是节点访问次数\\
Q和P保证好的策略优先选到，N保证不会经常遍历一个节点，提高速度，也实现了\epsilon - greedy
$$
​	扩展（叶子结点）：当叶子结点的N足够大，就会扩展出新的叶子结点

​	评估：用value network与直接下棋得到结果（用一个小线性回归网络，虽然不准但是很快）相均衡，降低了层数
$$
V(s_L) = (1-\lambda)v_\theta(s_L) + \lambda z_L
$$
​	存储：
$$
Q(s,a) = \frac{\sum_{i=1}^{n}1(s,a,i)V(s_l^i)}{N(s,a)}，用于后续搜索\\
N(s,a)用于选择步骤
$$
![AlphaGO](/AlphaGO.png)

#### AlphaGo Zero

$$
(p,v) = f_\theta(s), l = (z-v)^2 - \pi^T logp + c||\theta||^2\\
将策略和价值网络合二为一，损失函数三项分别是value，policy，正则化
$$

改进：1.不用人类棋谱数据（缺乏多样性），全用对弈数据

​			2.不用人类特征

​			3.多个网络合为一，而且用了更深的resnet

​			4.MCTS只用于训练网络，真正测试直接使用神经网络（policy gradient）给出决策

![zero](/zero.png)

## 5.强化学习思考

难题：

- ​	对于reward，转移概率等未知的情况处理的不好
- ​	很多问题上效果并不好，比如机器人等复杂现实问题
- ​	debug难度过大

## 6.整体思维导图

![mindmap](/mindmap.jpg)