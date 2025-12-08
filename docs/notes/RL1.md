# Reinforcement Learning for Variable Selection in a Branch and Bound Algorithm

<span style="color:red">我们可以认为这是B&B与RL开始的起点</span>

## 前言

区别于以往的IL或者其他启发式选择分支变量的方法，RL实际上聚焦于从0开始学习如何对分支变量进行选择（事实上也可以在已有参数上不从0开始，会收敛更快）。总之，很重要的一点是，不同MILP问题是由不同**特征**的，因此不同类问题P相当于从不同分布$D$当中抽样（电力、某些背包都相当于从不同D中抽样），标准形式即：
$$
p \in \mathcal{P}:
\begin{cases}
\min\limits_{x \in \mathbb{R}^n} \; c^\top x \\
\text{s.t.} 
& Ax \le b, \\
& x_J \in \{0,1\}^{|J|}, \\
& x_{-J} \in \mathbb{R}^{\,n-|J|}.
\end{cases}
$$
