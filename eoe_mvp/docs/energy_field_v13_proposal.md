# EOE v13.0 能量场物理规则改动方案

> 设计者：陆正旭  
> 版本：v13.0 Draft  
> 核心思想：将能量从"奖励数值"还原为"物理流体"

---

## 一、核心理念升级

### 旧范式：离散奖励
- 食物 = 空间中的离散对象
- 进食 = 直接能量数值增加
- 代谢 = 基于大脑复杂度的静态扣除

### 新范式：热力学流体
- 能量 = 连续标量场 E(x,y)，像温度/浓度一样分布
- 进食 = 能量从环境场流入Agent
- 代谢 = 能量从Agent流出到环境（废热排放）
- 移动 = 做功，消耗能量
- 掠夺 = 渗透膜驱动的能量自发流动

---

## 二、能量场系统 (The Scalar Field)

### 2.1 场定义

```python
# 环境能量场：二维矩阵 E(x, y)
class EnergyField:
    resolution: float = 1.0  # 网格分辨率
    width: int = 100         # 场宽度
    height: int = 100        # 场高度
    field: np.ndarray        # shape (width, height), 能量浓度
    
    # 物理常数
    diffusion_rate: float = 0.05    # 扩散系数 D
    decay_rate: float = 0.001       # 熵增衰减 k
```

### 2.2 扩散方程 (Diffusion)

能量场遵循**扩散方程**：

$$\frac{\partial E}{\partial t} = D \nabla^2 E - kE$$

其中：
- $D$ = diffusion_rate (扩散系数)
- $k$ = decay_rate (衰减系数)

物理意义：
- 能量自发向周围蔓延（浓度差驱动）
- 能量随时间缓慢消失（熵增）

### 2.3 能量源 (Sources)

不再是随机刷新离散食物，而是**固定的能量泉眼**：

```python
class EnergySource:
    position: Tuple[float, float]
    injection_rate: float    # 能量注入速率 (单位/帧)
    radius: float            # 影响半径
    
# 示例：3个能量泉眼
sources = [
    EnergySource((25, 25), injection_rate=0.5, radius=15),
    EnergySource((75, 75), injection_rate=0.5, radius=15),
    EnergySource((75, 25), injection_rate=0.3, radius=10),
]
```

这会自然产生**空间竞争**——Agent必须争夺高能区域。

---

## 三、热力学闭环 (The Cycle)

### 3.1 能量守恒

能量不会凭空消失，而是转化：

```
┌─────────────────────────────────────────────────────────────┐
│  Agent内部能量                                              │
│                                                             │
│    ┌─────────────┐    移动做功      ┌─────────────────┐   │
│    │  代谢输入   │ ──────────────→  │  移动能量消耗   │   │
│    │ (从环境吸收)│                  │  F² × c         │   │
│    └─────────────┘                  └─────────────────┘   │
│          ↑                                      │          │
│          │                                      ↓          │
│    ┌─────────────┐                  ┌─────────────────┐   │
│    │  废热排放   │ ←───────────────  │  大脑运行成本   │   │
│    │ (排入环境)  │                  │  nodes×α+edges×β│   │
│    └─────────────┘                  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │   环境能量场 E(x,y)  │
                    │   (吸收 + 扩散 + 衰减) │
                    └─────────────────────┘
```

### 3.2 移动做功 (Work)

Agent移动时对外做功，消耗能量：

$$E_{move} = c \times |F|^2$$

其中：
- $F$ = 左右输出力的合力 (motor force)
- $c$ = move_cost_coefficient (能量系数)

**物理意义**：速度越快，阻力越大，消耗呈**非线性增长**。

```python
# 实现
speed = (left_force + right_force) / 2.0
move_cost = MOVE_COST_COEFF * speed ** 2
agent.internal_energy -= move_cost
```

### 3.3 废热排放 (Heat Waste)

Agent的代谢（大脑运行 + 移动）不完全消失，而是以"低价值能量"形式排放到环境：

```python
# 当前代谢消耗
metabolic_cost = n_nodes * ALPHA + n_edges * BETA + move_cost

# 废热排放到Agent所在位置
x_idx = int(agent.x / resolution)
y_idx = int(agent.y / resolution)
field[x_idx, y_idx] += metabolic_cost * WASTE_HEAT_RATIO
```

这形成**正反馈**：高代谢的Agent周围能量反而更高（但需要开启渗透膜才能利用）。

---

## 四、Agent能量端口 (The Membrane)

### 4.1 渗透膜 (Permeability Membrane)

每个Agent有一个**渗透率**参数 $\kappa$：

```python
class Agent:
    permeability: float = 0.0  # 渗透率 [0, 1]
    # 0 = 封闭（不与环境交换能量）
    # 1 = 完全开放（与环境能量场平衡）
```

### 4.2 能量交换

Agent与环境场的能量交换：

$$E_{exchange} = \kappa \times (E_{field}(x,y) - E_{agent})$$

- 当 $E_{field} > E_{agent}$：能量流入Agent
- 当 $E_{field} < E_{agent}$：能量流出到环境

```python
def update_energy_exchange(agent, field):
    field_energy = field[int(agent.x), int(agent.y)]
    agent_energy = agent.internal_energy
    
    # 交换量
    exchange = agent.permeability * (field_energy - agent_energy)
    
    agent.internal_energy += exchange
    # 对应位置的能量变化（简化：直接扣除/添加到场）
    field[agent.pos] -= exchange
```

### 4.3 渗透膜代价

开启渗透膜有代价（防止"永远开着"）：

```python
# 渗透膜维持需要少量能量
permeability_cost = agent.permeability * 0.01
agent.internal_energy -= permeability_cost
```

---

## 五、掠夺机制 (Predation)

### 5.1 物理本质

如果Agent A的 $E_{agent}$ 极高，而Agent B靠近它并开启高渗透率，则：

$$E_{flow} = \kappa_B \times \sigma \times (E_A - E_B)$$

其中 $\sigma$ 是"攻击系数"（演化获得）。

### 5.2 实现

```python
def handle_agent_interaction(agent_a, agent_b):
    distance = toroidal_distance(agent_a, agent_b)
    
    if distance < interaction_range:
        # B从A窃取能量（如果B的渗透率高于A）
        potential_theft = agent_b.permeability * (agent_a.energy - agent_b.energy)
        
        if potential_theft > 0:
            steal_amount = potential_theft * THEFT_EFFICIENCY
            agent_b.energy += steal_amount
            agent_a.energy -= steal_amount
```

### 5.3 涌现行为

- **捕食**：演化出高渗透率 + 高攻击系数
- **防御**：演化出低渗透率或在低能区关闭渗透膜
- **寄生**：演化出在靠近高能体时开启渗透的行为

---

## 六、与旧版本的兼容性

### 6.1 配置开关

```python
class Environment:
    # v13.0 新增
    energy_field_enabled: bool = False
    
    # 如果关闭，回退到旧的食物系统
    food_positions: List[Tuple[float, float]]  # 仍然可用
    
    # 新参数
    energy_sources: List[EnergySource] = []
    diffusion_rate: float = 0.05
    decay_rate: float = 0.001
    move_cost_coeff: float = 0.1
    waste_heat_ratio: float = 0.3  # 30%代谢转化为废热
```

### 6.2 渐进式迁移

建议分阶段实现：

1. **Phase 1**：实现能量场 + 扩散 + 源（保留离散食物）
2. **Phase 2**：实现移动做功 + 废热排放
3. **Phase 3**：实现渗透膜 + 掠夺机制
4. **Phase 4**：移除离散食物，完全基于能量场

---

## 七、预期涌现行为

### 7.1 自发进食策略

Agent必须演化出：
1. 感知能量梯度
2. 移动到高能区域
3. 开启渗透膜吸收能量

这比写死的 "Eat" 端口要**高级得多**。

### 7.2 避免贪婪

因为：
- 开启渗透膜有能量代价
- 在低能区开启渗透会流失能量

Agent必须学会**在不进食时关闭渗透**，自然演化出"消化"或"潜伏"的概念。

### 7.3 空间防御

通过调节渗透率为负值（需要逻辑允许）或调节防御参数，Agent可以防止被他人吸取。

---

## 八、关键参数参考

| 参数 | 建议值 | 说明 |
|------|--------|------|
| resolution | 1.0 | 能量场网格分辨率 |
| diffusion_rate | 0.05 | 扩散系数D |
| decay_rate | 0.001 | 熵增衰减k |
| source_injection | 0.5 | 能量源注入速率 |
| move_cost_coeff | 0.1 | 移动做功能量系数 |
| waste_heat_ratio | 0.3 | 废热排放比例 |
| permeability_cost | 0.01 | 渗透膜维持代价 |

---

## 九、待讨论问题

1. **场分辨率**：1.0格子是否够细？还是需要更精细？
2. **环形世界**：能量场如何处理toroidal边界？
3. **性能**：每帧扩散计算O(W×H)，是否需要降采样或GPU加速？
4. **初始状态**：Agent初始能量如何？环境场初始能量？
5. **死亡判定**：能量 <= 0 死亡，还是需要阈值？

---

> 欢迎审阅后反馈修改意见。