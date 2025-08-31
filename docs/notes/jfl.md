# Dealer: An End-to-End Model Marketplace with Differential Privacy

## 相较于MBP的改进

1. 完善了整个模型的架构
    1. 允许多个数据提供者提供他们的数据集，并根据shapley和隐私泄露情况进行分配。
    2. 对买家有多个价格点进行调研，更符合实际情况。
    3. broker真正承担了枢纽的作用。

2. 从查分隐私角度思考了噪声的作用

## 各方利益探讨

### 数据提供者

提供数据会获得补偿，天经地义，让我们看看这是如何刻画的。

$$
c_i(\epsilon) = b_i \cdot s_i(\epsilon)
$$

其中$b_i$是和shapley正比的一个数值，$s_i(\epsilon)$是和噪声（差分隐私）相关的量。

> Tip:论文中的差分隐私是通过$(\delta,\epsilon)$对来控制的，但是文中其实是固定了$\delta$。所以$\epsilon$越大，噪声就越小，因而这里的$\epsilon$就和MBP的$\frac{1}{\delta}$类似。

### 买家

$$
P(B_j, M) = V_j \cdot 
\frac{1}{1 + e^{-\delta_j \big( CR(M) - \theta_j \big)}} \cdot
\frac{1}{1 + e^{-\gamma_j (\epsilon - \eta_j)}}
$$

可以认为$V_j$是卖家的原始估值，后面两个影响因子分别是shapley coverage和噪声。但是本文一直都在
强调shapley coverage对模型的影响相较于噪声很小。所以在后面的调查和动态规划中shapley coverage事实上都没有出现。

### 中间商

\begin{equation}
\arg\max_{\langle p(\epsilon_1), \dots, p(\epsilon_l) \rangle} 
\prod_{k=1}^{l} \prod_{j=1}^{m'} 
p(\epsilon_k) \cdot I(t_{mj} = M_k) \cdot I\big(p(\epsilon_k) \le v_j\big),
\end{equation}


\begin{equation}
\text{s.t. } \; p(\epsilon_{k_1} + \epsilon_{k_2}) \le p(\epsilon_{k_1}) + p(\epsilon_{k_2}), 
\quad \epsilon_{k_1}, \epsilon_{k_2} > 0,
\end{equation}


\begin{equation}
0 < p(\epsilon_{k_1}) \le p(\epsilon_{k_2}), 
\quad 0 < \epsilon_{k_1} \le \epsilon_{k_2}.
\end{equation}


与MBP不同的是，它有多个的采样点。从这里可以再次说明，这里只考虑“噪声”无套利，而没有shapley coverage无套利，因为文中也说难以处理。


## 利益分配

利益分配主要包含broker对买家的收费和broker对数据提供者的补偿。

### 收费

文中的动态规划颇为巧妙，比MBP的好很多。利用到子结构的最优解只可能会在多项式个离散点当中，所以就把
状态存在这些子结构当中（并且能保证最优解一定会用这些子结构构成），把所有子结构计算一遍就得到了最终的结果，非常值得学习。

### 补偿



\begin{equation}
\arg\max_{S \subseteq \{D_1, \dots, D_n\}} \;
\prod_{i : D_i \in S} SV_i,
\end{equation}



\begin{equation}
\text{s.t. } \; \prod_{i : D_i \in S} c_i(\epsilon) \le MB.
\end{equation}


其实就是一个0-1背包问题。作者分别使用伪多项式算法进行精确解求解，用先装入单位价值最高物品的方法求解，以及一种结合随机性和单位最高价值的方法求解。

> Tip:这里的shapley coverage与论文前面定义的shapley coverage有所不同，这里只要求“单个人的shapley之和最大”，而前面要求的是联盟的shapley最大，这是买家的要求。