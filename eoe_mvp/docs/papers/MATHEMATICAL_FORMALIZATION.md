# EOE 数学形式化方案 (Mathematical Formalization)

> 版本: v1.0 | 最后更新: 2026-03-14
> 本文档将系统拆解为 6 个相互耦合的数学模块，可直接映射为 OOP 类属性和 PDE 数值求解器

---

## 全局状态空间定义 (Global State Space)

系统 $\Omega$ 由连续/离散混合时间 $t$ 驱动，包含环境场 $\mathcal{F}$ 和智能体种群 $\mathcal{P}$：

$$\Omega(t) = \langle \mathcal{F}(t), \mathcal{P}(t) \rangle$$

---

## 模块一：智能体与脑结构形式化 (Agent & Brain Topology)

### 定义

种群 $\mathcal{P}(t) = \{ A_1, A_2, ..., A_{N(t)} \}$，其中 $N(t)$ 为动态种群规模。

智能体状态: $A_i = \langle x_i, G_i, E_i, a_i \rangle$

### 1.1 物理状态 $x_i$

位置 $\mathbf{p}_i \in \mathbb{R}^2$ 和速度 $\mathbf{v}_i \in \mathbb{R}^2$

### 1.2 脑图结构 $G_i$

大脑是有向加权图 $G_i = (V_i, E_i, W_i)$:

- **节点集** $V_i = V_{sens} \cup V_{hid} \cup V_{act}$ (感知/隐藏/执行)
- **边集** $E_i \subseteq V_i \times V_i$
- **权重矩阵** $W_i$, $w_{jk}$ 为节点 $j \to k$ 连接强度
- **网络推理**: $\mathbf{u}_i(t) = \Phi_{G_i}(\mathbf{s}_i(t))$

---

## 模块二：多通道物理场系统 (Multi-Channel Physics Fields)

环境 $\mathcal{F}$ 是多层空间标量/矢量场矩阵，空间坐标 $\mathbf{p}$。

### 2.1 能量场 $F_{energy}(\mathbf{p}, t)$

$$\frac{\partial F_{energy}}{\partial t} = \phi_{regen} - \sum_{i=1}^{N} \delta(\mathbf{p} - \mathbf{p}_i) \cdot c_{gather}(\mathbf{u}_i)$$

- $\phi_{regen}$: 自然恢复率
- $c_{gather}$: 智能体采食函数
- $\delta$: 狄拉克函数

### 2.2 痕迹场 $F_{stigmergy}(\mathbf{p}, t)$

$$\frac{\partial F_{stigmergy}}{\partial t} = D_s \nabla^2 F_{stigmergy} - \lambda_s F_{stigmergy} + \sum_{i=1}^{N} \delta(\mathbf{p} - \mathbf{p}_i) \cdot e_{secrete}(\mathbf{u}_i)$$

- $D_s$: 扩散系数
- $\lambda_s$: 挥发衰减率
- $e_{secrete}$: 分泌量

### 2.3 阻抗场 $F_{impedance}(\mathbf{p}, t)$

静态/动态摩擦力/障碍物场，影响移动能量损耗

### 2.4 压力场 $F_{stress}(\mathbf{p}, t)$

动态危险区域（如严冬），额外能量扣除

---

## 模块三：大脑热力学与能量循环 (Brain Thermodynamics)

### 3.1 个体代谢函数 $M(A_i)$

$$M(A_i) = \underbrace{m_{base} + k_{stress} F_{stress}(\mathbf{p}_i)}_{\text{基础与环境}} + \underbrace{\alpha |V_i| + \beta |E_i| + \gamma \sum |W_i|}_{\text{大脑热力学惩罚}} + \underbrace{\eta \|\mathbf{v}_i\| F_{impedance}(\mathbf{p}_i)}_{\text{运动做功}}$$

### 3.2 个体能量积分方程

$$\frac{d E_i}{dt} = I_{gain}(A_i, \mathcal{F}) - M(A_i)$$

### 3.3 生命上限约束

$$Condition_{death} = (E_i \le 0) \lor (a_i \ge a_{max})$$

### 3.4 尸体反哺机制

$$F_{energy}(\mathbf{p}_i, t^+) = F_{energy}(\mathbf{p}_i, t^-) + \rho \cdot \max(E_i, E_{base\_meat})$$

---

## 模块四：具身动力学与交互 (Embodied Kinematics)

### 4.1 空间感知 (Perception)

$$\mathbf{s}_i^{(k)}(t) = \int_{\|\mathbf{r}\| \le R} \mathcal{K}(\mathbf{r}) \cdot F_k(\mathbf{p}_i + \mathbf{r}, t) d\mathbf{r}$$

- $\mathcal{K}$: 感受野核函数
- $F_k$: 能量、痕迹等各类场

### 4.2 运动更新 (Kinematics)

$$\mathbf{v}_i(t+\Delta t) = \mathbf{v}_i(t) + \Delta t \left[ \mathbf{u}_{move} - \mu F_{impedance}(\mathbf{p}_i)\mathbf{v}_i(t) \right]$$

$$\mathbf{p}_i(t+\Delta t) = \mathbf{p}_i(t) + \mathbf{v}_i(t)\Delta t$$

---

## 模块五：演化与鲍德温机制 (Evolution & Baldwin Effect)

### 5.1 繁殖触发器

当 $E_i > E_{reproduce}$ 且满足冷却时间，触发分裂

### 5.2 拓扑变异 (NEAT-logic)

- $P_{add\_node}$: 插入新节点
- $P_{add\_edge}$: 连接两个节点
- $P_{mutate\_weight}$: 微调权重

### 5.3 鲍德温表型可塑性

$$\frac{d W_{i}}{dt} = \eta \cdot \mathbf{u}_i(t) \cdot \mathbf{s}_i^T(t)$$

遗传同化:

$$W_{child\_init} = (1 - \kappa) W_{parent\_genotype} + \kappa W_{parent\_phenotype}$$

---

## 模块六：宏观复杂性度量 (Complexity Metrics)

### 6.1 拓扑信息熵

$$H(G) = -\sum_{k} P(deg(k)) \log P(deg(k))$$

### 6.2 超节点封装条件

子图 $g \subset G_i$ 满足:
- 内部连接密度 $\rho_{internal} \gg \rho_{external}$
- 种群中出现频率 $f(g) > \theta_{freq}$

---

## Python 架构映射

| 模块 | 类 | 核心属性/方法 |
|------|-----|--------------|
| 一 | `Agent` | `position`, `velocity`, `brain: Graph`, `energy`, `age` |
| 一 | `Brain` | `nodes`, `edges`, `weights`, `forward()` |
| 二 | `EnergyField` | `grid`, `regen_rate`, `update()` |
| 二 | `StigmergyField` | `grid`, `diffusion`, `decay`, `update()` |
| 三 | `MetabolismSystem` | `base_cost`, `node_cost`, `edge_cost`, `calculate()` |
| 四 | `PerceptionSystem` | `receptive_field`, `convolve()` |
| 四 | `KinematicsSystem` | `update_position()`, `apply_impedance()` |
| 五 | `EvolutionSystem` | `reproduce()`, `mutate()`, `hebbian_update()` |
| 六 | `ComplexityTracker` | `calculate_entropy()`, `detect_supernodes()` |

---

## 核心耦合关系

```
Simulation Step:
  1. Perception (Module 4) → 获取 s_i(t)
  2. Brain.forward() (Module 1) → 计算 u_i(t)
  3. Kinematics (Module 4) → 更新 p_i, v_i
  4. Energy Gathering (Module 2) → 更新 E_i
  5. Metabolism (Module 3) → 扣除 M(A_i)
  6. Check Death (Module 3) → 移除死亡个体
  7. Reproduction (Module 5) → 可能的繁殖
  8. Hebbian Learning (Module 5) → 更新权重
  9. Mutation (Module 5) → 拓扑变异
  10. Complexity Tracking (Module 6) → 记录结构
  11. Field Update (Module 2) → 能量/信息素扩散
```

---

*Generated from Mathematical Formalization v1.0*