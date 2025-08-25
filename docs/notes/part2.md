# 第二章

## 逻辑回归

现在我们要解决的分类问题，先从二分类问题看起，此时我们的目标函数空间只有两个离散的取值$\{0,1\}$，其中1叫***positive class***,0叫***negative class***。在前一节中我们的$\theta^T x$取值范围在$R$,我们可以在外面套一层:
$$
g(z)=\frac{1}{1+e^{-z}}
$$
来使值域在(0,1)上，得到：
$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$
该函数被叫做逻辑函数或者激活函数(sigmoid function)。

**还有一个尚未被点明的点，就是sigmod函数非常适合用来建模概率，以后会说明（这也就是上下思路突然跳跃的原因，因为事实上我们有多种方法来将整个实数域映射到(0,1)）**

总之，我们选择用该函数刻画y的条件概率：
$$
\begin{aligned}
P(y = 1 \mid x; \theta) &= h_\theta(x) \\
P(y = 0 \mid x; \theta) &= 1 - h_\theta(x)
\end{aligned}
$$

由于y只有两个孤立的取值点，所以它自然可以表示为：
$$
p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}
$$

将x,y等视为已知量，我们可以表示出对于n个样本的似然函数为：

$$
L(\theta)=p(\vec{v}| X ;\theta)
$$

用对数展开求解：
$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log (1 - h_\theta(x^{(i)})) \right]
$$

由于我们需要最大化，所以采用梯度上升求解（单一样本情况）：
$$
\theta := \theta + \alpha \nabla_\theta l(\theta)
$$

省略求解过程，我们不难可以求出$\alpha \nabla_\theta l(\theta)$（单一样本情况）：

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

多样本只用加一个$\sum$符号即可，可见其形式与线性函数的非常相似。


## 感知机算法

感知机算法对于二分类问题更加naive,直接以$h_\theta(x)$正负分类：
$$
h_\theta(x)=g(\theta^T x)=\begin{cases}
1 \ \ if \ \ z\geq 0\\0 \ \ if \ \ z < 0
\end{cases}
$$

更新规则依然是：

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$
