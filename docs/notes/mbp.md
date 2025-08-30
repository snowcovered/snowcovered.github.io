# Model-Based Pricing

## 动机确认

数据市场究竟应该**怎么卖**数据？

以往的传统操作方法：

* 直接售卖数据集：缺点在于买下相同数据集的价格是固定且高昂的，一些买家可能无法负担。从另一个角度，卖家的收益就少了。

* 售卖查询：缺点在于，仅仅售卖查询不足以让买家进行复杂的决策。

**以前训练模型的任务都是交给买家完成**，但事实上卖家（或者中间商）就可以做到这一步，之所以要直接卖模型，正是因为添加不同量模型的噪声对应不同的价值，这样我们就能对不同需求的买家收费(版本化)。

## 模型构建(案件还原)


### 线索1： $\delta \leftrightarrow \epsilon$

添加噪声必然使得模型输出变得不准（分类模型or回归模型），关键是如何刻画。文中我们可以简单得到如下关系：

Let \( \epsilon \) be convex as a function of the model instance \( h \).  
Let  

$$
\hat{h}^{\delta}_{\lambda}(D) = KG(h^{*}_{\lambda}(D), w).
$$  

Then, for any two parameters \(\delta_1, \delta_2\), we have  

$$
\mathbb{E}\big[ \epsilon(\hat{h}^{\delta_1}_{\lambda}(D), D) \big] 
\;\;\ge\;\; 
\mathbb{E}\big[ \epsilon(\hat{h}^{\delta_2}_{\lambda}(D), D) \big]
$$

**if and only if** \(\delta_1 \ge \delta_2\).  

If \(\epsilon\) is additionally **strictly convex**, the above holds with **strict inequality** (\(>\)).

### 线索2： $\epsilon \leftrightarrow p$

在（买家提供的）误差函数$\epsilon(.,.)$检验下，显然更低的误差意味着更高的价值，即：

$$
\mathbb{E}\big[ \epsilon(\hat{h}^{\delta_1}_{\lambda}(D), D) \big] 
\;\;\le\;\; 
\mathbb{E}\big[ \epsilon(\hat{h}^{\delta_2}_{\lambda}(D), D) \big]
$$  

**意味着**  

$$
p_{\epsilon,\lambda}(\delta_1, D) \;\;\ge\;\; p_{\epsilon,\lambda}(\delta_2, D).
$$

如果仅仅到此为止，这非常简单（我们可以让噪声越高的模型价格越低就可以），但关键就在于，***两个低价格，高噪声的模型，可能合成出一个低噪声的模型，其成本可能低于直接购买该模型***，即套利。


###关键推论：无套利条件

作者依据Cramer-Rao不等式（以及其他Fisher Information相关推论）提供的下界，证明了对于平方误差下无套利条件有这样的性质：
(1) If 

$$
\frac{1}{\delta_1} = \frac{1}{\delta_2} + \frac{1}{\delta_3},
$$  

then  

$$
p_{\epsilon^s,\lambda}(\delta_1, D) \;\;\le\;\; 
p_{\epsilon^s,\lambda}(\delta_2, D) + p_{\epsilon^s,\lambda}(\delta_3, D).
$$  

(2) If 
$$
\delta_1 \le \delta_2,
$$  

then  

$$
p_{\epsilon^s,\lambda}(\delta_1, D) \;\;\ge\;\; 
p_{\epsilon^s,\lambda}(\delta_2, D).
$$

简单记为：

$$
\overline{p}(x) \;=\; p_{\epsilon^s,\lambda} \left(\tfrac{1}{x}, D\right).
$$

满足次可加性和单调两个条件。

当然，平方误差只是一个特例，任何严格凸的误差函数，在$\delta$和$\epsilon$之间都构成双射，我们只需要构造一个逆函数

$$
\delta \;=\; \phi\!\left( \; \mathbb{E}\left[\, \epsilon \big(\hat{h}^{\delta}_{\lambda}(D), D \big) \,\right] \;\right).
$$

只要

$$
\overline{p}(x) \;=\; p_{\epsilon,\lambda}\left(\tfrac{1}{\phi(x)}, D\right).
$$

满足次可加性和单调两个条件就实现了无套利，**显然，对于平方误差，$\phi(x)\equiv x$**。

现在为止，我们真正建立了$\epsilon \leftrightarrow p$的约束关系。


## 简化并解决

### 原始问题分析

我们通过市场调研得到了不同买家对于不同质量模型的需求，那么问题实际上就是一个带约束的线性规划问题：

$$
\begin{aligned}
\max_{\hat{p}} \quad & T\big(\hat{p}(a_1), \ldots, \hat{p}(a_n)\big) \\
\text{subject to} \quad 
& \hat{p}(x+y) \leq \hat{p}(x) + \hat{p}(y), \quad x,y \geq 0, \\
& \hat{p}(y) \geq \hat{p}(x), \quad y \geq x \geq 0, \\
& \hat{p}(x) \geq 0, \quad x \geq 0.
\end{aligned}
$$

另外，我们容易发现求解最大收益实际上是一种次线性插值问题。然而，这是一个CO-NP问题。

>由于无界子集和问题是一个NP-hard问题，作者已经证明证明：当且仅当不存在值为 $K$ 的（无界）子集和时，才存在一个次可加且单调的函数能够插值这些点 $(a_j, P_j)$。

**我们往往可以近似解决或者解决一个相关问题，然后分析相关问题的精确解与原问题的解的关系，本文采用了后者。**

### 退一步讲

我们可以缩小搜索空间，将原问题缩小为如下问题（这里指的是下面问题的解满足原问题约束，反之不然）：

$$
\begin{align*}
\max \;& \hat{q}^T \big(\hat{q}(a_1), \dots, \hat{q}(a_n)\big) \\
\text{subject to } \;& \frac{\hat{q}(y)}{y} \le \frac{\hat{q}(x)}{x}, && y \ge x > 0, \\
& \hat{q}(y) \ge \hat{q}(x), && y \ge x \ge 0, \\
& \hat{q}(x) \ge 0, && x \ge 0.
\end{align*}
$$

***之所以要提出这个缩小版问题，有两个原因，第一个原因***，那就是他有与原问题最优相比优良的性质,当“小问题”有可行解$\hat{q}$时，原问题就有可行解$\hat{p}$,且

$$
\frac{\hat{p}(x)}{2} \le \hat{q}(x) \le \hat{p}(x)
$$

最重要的是***第二个***原因：***该问题多项式时间内可解***

在解决最终问题前，我们可以首先把这个“小”问题再简化一下，从函数的关系直接到点的关系（等价性证明略）：

$$
\begin{align*}
\max \;& T(z_1, \dots, z_n) \\
\text{subject to } \;& \frac{z_j}{a_j} \le \frac{z_i}{a_i}, && a_j \ge a_i, \\
& z_j \ge z_i, && a_j \ge a_i, \\
& z_j \ge 0, && 1 \le j \le n.
\end{align*}
$$

###最后一块拼图——动态规划求解

作者的动态规划颇为奇怪，但我们最终一定能够拨云见日。

比如下面只有4个点的情况，假设从$A_1$到$A_4$每个点的权重(人数)分别为2,3,4,1。

简单证明可知，我们在每一个x下最优定价必然会出现在斜率约束处或者单调约束处。***由于本论文出现的所有点默认就是满足单调性约束的***（这是相对合理的，因为人们往往会对高精度产品做出高估值，尤其是在信息差很小的情况），那么，我们的所有约束其实就是斜率约束。

![p11图](./pic/301.png)

如下所示，我们做出约束线：

![p12图](./pic/302.png)

按照斜率从大到小我们可以编号为$l_1,l_2,...,l_4$，斜率可以编号为$\Delta_1,\Delta_2,...,\Delta_4$,注意的是，斜率的编号与点的编号是不对应的。

#回顾梳理

据我的回忆，机制执行过程简单概括如下：

1. 中间人市场调查，得到模型需求，得到一个噪声-价值的点对的集合。
2. 利用动态规划计算最优价格。
3. 一手交钱，一手交货。

<!--简单来说就是调研市场，绘制价格-噪声曲线，然后卖-->

<!--我们可以大致认为经纪人需要训练模型-->

<!--可以认为版本化就是对一个东西卖出不同的版本-->

<!--买家可以挑选自己喜欢的误差函数-->


<!--定价依赖的东西还挺多的-->

<!--我们可以认为中间商最终呈现error主要是为了方便买家选择-->

<!--w就是从协方差矩阵当中生成的向量-->

<!-- Lemma 1
注意error-monotone的定义，然后构造一个非单调函数与常数函数超过1个交点就可以证明套利

 -->

 <!-- lemma2 是无偏的也很好证明 因为我们添加的噪声就是均值为0的高斯噪声  -->


 <!-- lemma 3 噪声的构造本身就可以说明问题 -->

<!-- th 4   这个证明中先证明了严格凸函数的性质，但是这是显然的  
然后用这个严格凸函数的性质证明了


我们需要注意到th 4 的刻画是任意的，并不只是局限在s误差函数上
-->

<!--  th5 构造一个反例就可以证明其中一边的方向的正确性  另一边可以用数学归纳法证明,先证明1-套利违反假设，然后递推出去即可   核心在cramer 

但是th5 只是在平方损失上衡量的
-->

<!--
th6 可以视为对th5的推广 ，它利用到了th4的最后严格凸条件， 还有一个非常非常重要的insight就是 \delta和         

还有一个关键的观察是 \delta---w----E(\epsilon),严格凸意味着这个映射只用考虑从w----E(\epsilon)即可，次可加应该是从这里来的 （这里描述得不准）

关键还得看逆映射


-->
总之，1-6已经说明了简单的充要条件是什么了

<!--th 7相对来说具有一定的技术性，其实就是证明一个K-子集问题和次可加性插值之间是Co 问题即可，前者时NP的-->

<!--th8 非常简单 -->


---
此后就要开始强化条件，证明下界

<!--
这里主要表明新提出的2在证明上是具有优势的
-->

从prop1开始，3和4就已经基本上等价了，所以作者聚焦于解决3的问题

prop2从函数值的关系转向对最优值的关系的探讨(其中prop2 解决的是对前两个要求的近似，也就是说解决4很大程度上也解决了问题2)

那么prop3实际上解决的就是3

<!--最后的核心，保存每个\delta的状态,从最后一个点更新至前面的点

还有一个很关键的点是大的\delta可以很好地兼容小delta的情况>