# 一、对流扩散方程
对流扩散方程（convection diffusion equation）具有以下的形式:

$$
\frac{\partial c}{\partial t} + u \frac{\partial c}{\partial x} = D \frac{\partial^2 c}{\partial x^2}
$$

或写作：

$$u_t + c u_x - D u_{xx}=0$$

其中：
- $c$：浓度（或温度、动量等物理量）；
- $\mathbf{u}$：流体速度场（对流项）；
- $D$：扩散系数（扩散项）；
- 对流项（$`u \frac{\partial c}{\partial x}`$）描述流体运动导致的物质输运；
- 扩散项（$`D \frac{\partial^2 c}{\partial x^2}`$）描述浓度梯度驱动的分子扩散。

如下图所示：
![[对流扩散问题示意图.png]]

图中是一个3D空间中的扩散示意图，在t=0.5时，物质在空间中的浓度如图所示。
## 物理条件
本次研究旨在解决对流扩散方程的某一特定物理情景，由于对流扩散方程适用于众多物理问题，故本研究只聚焦于一种物理情形，并且在该物理条件下，在方程的边界点和内部通过随机采样的方式得到训练集，条件如下：
### 1. 空间域和时间域
- 空间：$`x \in [0, 1]`$
- 时间：$`t \in [0, 1]`$
### 2. 初始条件
在$`t=0`$时，初始分布为高斯函数：

$$u(x, 0) = e^{-10x^2}$$

这表示在$`x = 0`$附近有一个初始浓度峰值（或热量集中）。
### 3. 边界条件
- 左边界$`x = 0`$：$`u(0, t) = 1`$
- 右边界$`x = 1`$：$`u(1, t) = 0`$

这代表一种物理情景：**左边持续注入浓度为1的物质，右边自由扩散至0浓度**。
### 4. 参数设置
- $`c = 0.5`$：对流速度（convection coefficient）；
- $`D = 0.01`$：扩散系数（diffusion coefficient）。

在物理世界中，上述条件常常存在于如下情境：
- 热传导问题（热源在左边）
- 污染物输运问题（左边持续排放，右边自然清除）
# 二、数据集
在PINNs中，**训练数据集**和**验证数据集**的划分方式与传统监督学习有所不同。PINNs不依赖大量真实标签数据，而是通过**物理方程（PDE）和少量边界/初始条件数据**共同构建损失函数进行训练。下面具体说明：
## 1. 训练数据
训练数据由两个部分组成：
### 第一部分 **边界条件 & 初始条件点（BC&IC）**——有监督训练数据
这是代码中 `self.X_train` 和 `self.y_train` 的部分，代表网络应该在这些点上**精确拟合已知解值**。其构造方式如下：
#### 初始条件 (IC):
```python
ic = np.stack(np.meshgrid(x_bc_res, [self.t_min], indexing='ij')).reshape(2, -1).T
```
- 在$`t=0`$的全体$`x \in [0, 1]`$上采样了 50 个点。
- 标签为：

$$u(x, 0) = e^{-10x^2}$$

#### 边界条件 (BC):
```python
bc1 = np.stack(np.meshgrid([self.x_min], t_bc_res, indexing='ij')).reshape(2, -1).T
bc2 = np.stack(np.meshgrid([self.x_max], t_bc_res, indexing='ij')).reshape(2, -1).T
```
- 左边界$`x=0`$，取不同时间点$`t \in [0, 1]`$，共采样 50 个点，标签为$`u(0,t)=1`$。
- 右边界$`x=1`$，共采样50个点，标签为$`u(1,t)=0`$。

最终将三类点合并为训练输入`X_train`和对应标签`y_train`：

### 第二部分 **PDE 约束点（无标签）**——无监督方程残差点
```python
x_train_pde_res = np.linspace(self.x_min, self.x_max, 100)
t_train_pde_res = np.linspace(self.t_min, self.t_max, 100)
pde_points_x, pde_points_t = np.meshgrid(x_train_pde_res, t_train_pde_res, indexing='ij')
self.pde_points = Tensor(np.stack([pde_points_x.flatten(), pde_points_t.flatten()], axis=1), mindspore.float32)
```
- 在整个$`(x, t) \in [0, 1] \times [0, 1]`$的区域上，生成$`100 \times 100 = 10000`$个均匀采样点。
- 这些点没有标签，但在这些点上，模型预测的结果应**满足 PDE 方程残差为 0**：

$$f_{\text{PDE}}(x, t) = u_t + c u_x - D u_{xx} \approx 0$$

**这些数据用于“物理损失”（PDE Loss），约束模型符合对流扩散方程。**
## 2. 验证方法
PINNs求解PDE不是使用传统意义的验证集。PINNs 的训练过程依赖于：
- 有标签的边界条件、初始条件；
- 方程残差作为正则化。

验证效果的评估通过**可视化模型预测**（`plot_results()`）的方式间接完成。
# 四、模型训练与推理
## 1. 网络结构
本次实验使用了一个**全连接前馈神经网络（Feedforward Neural Network，简称 FNN）**，具体结构如下：

|层编号|类型|输入维度|输出维度|激活函数|
|---|---|---|---|---|
|1|`Dense(2, 20)`|2|20|Tanh|
|2|`Dense(20, 20)`|20|20|Tanh|
|3|`Dense(20, 20)`|20|20|Tanh|
|4|`Dense(20, 1)`|20|1|无|
## 2. Loss
对于PINNs来说，其独特的Loss构建方式使得其能够在解决PDE上发挥作用，本次实验的Loss函数由2部分构成，分别是**DataLoss**和**PDELoss**，具体形式如下：
- **DataLoss**：来自初始条件(IC)和边界条件(BC)：

$$\mathcal{L}\_{data} = \frac{1}{N} \sum_{i=1}^{N} \left( u_{\text{pred}}(x_i, t_i) - u_{\text{true}}(x_i, t_i) \right)^2$$

- **PDELoss**:基于 PDE 方程：

$$u_t + c u_x - D u_{xx} = 0$$

定义物理残差为：

$$f_{\text{pde}} = u_t + c u_x - D u_{xx}$$

残差损失为：

$$\mathcal{L}\_{pde} = \frac{1}{M} \sum_{j=1}^{M} \left( f_{\text{pde}}(x_j, t_j) \right)^2$$

最后，PINNS的Loss函数由DataLoss和PEDLoss相加得到。
# 五、结果
可视化预测的loss如图所示：![[loss.png]]
可视化预测结果的热力图如下：![[heatmap.png]]
可视化计算结果的时间线图如下：![[linemap.png]]



