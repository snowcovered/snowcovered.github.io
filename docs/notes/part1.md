# 基本定义
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


## LMS方法

前面已经定义了一个损失函数（最小二乘损失函数），我们优化的过程当然就希望优化这个损失函数——沿着损失函数梯度的反方向优化自然是一种不错的方法，所以有：
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

带入前面对$J(\theta)$的定义中，我们自然就能得到对于只有1个训练样本的情况，更新规则为：
$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

对于有n个样本的情况：
$$
\theta_j := \theta_j + \alpha \sum_{i=1}^{n} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

其中j=0,1，...,d。每一轮需要对每个d更新直到收敛。

>  This method looks
at every example in the entire training set on every step, and is called batch
gradient descent.

由于我们定义的损失函数是一个**凸二次函数**，局部最优解就是全局最优解，所以可以避免进入局部最优解。

![p12图](./pic/2.png)

上图就表示了一个收敛轨迹（圈圈是等高线）

