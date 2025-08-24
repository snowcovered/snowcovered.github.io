## 基本定义
$x^{(i)}$：输入变量

$y^{(i)}$:输出变量；或者叫做目标变量

$(x^{(i)} ,y^{(i)})$:训练样例

n个训练样例的列表：$\{(x^{(i)} ,y^{(i)});i=1,2,...,n\}$称为训练集

$\mathcal{X},\mathcal{Y}$:输入空间，输出空间

h:监督学习(supervised learning)的目标，即一个函数:$\mathcal{X}\mapsto \mathcal{Y}$,使得给定x能够较好地预测到y。

regression problem:学习问题中目标变量是连续变量的。

calssification problem:目标变量(y)是离散的。

# 线性回归

![p8图](./pic/1.png)

比如对于上面这张图，我们采取：
$$
h_\theta(x)=\theta_0+\theta_1 x_1+\theta_2 x_2
$$

这里的$\theta$称为parameter或者weights。为了方便表示，可以令$x_0=1$,实际上就是消除它乘上$\theta_0$的影响，因此有：
$$
h(x)=\sum_{i=0}^{d} \theta_i x_i=\theta^T x
$$
即将其视为向量内积，d表示维度。

---
**一个重要的问题是：我们如何找到合理的h呢？**

我们可以认为一个好的h总能使$h(x^{(i)})$接近对应的$y_i$，那么应该如何衡量呢？比如有下面一种衡量方式：

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$



