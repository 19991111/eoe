# 构成性环境升级方案 (Compositional Environment)

> **指令来源**: EOE Project Lead via Architect AI  
> **目标**: 引入环境构成性与动态物理拓扑  
> **版本**: v16.0 (Composer)  
> **生成日期**: 2026-03-14

---

## 1. 背景与动机

### 1.1 当前问题

当前 EOE 环境 $\mathcal{F}$ 是**静态的"果冻状态"**：
- 能量场和痕迹场只是标量值的起伏
- 智能体无法改变空间的绝对拓扑
- 环境复杂度天花板极低
- 大脑结构演化停滞（最优策略收敛）

### 1.2 解决方案

引入**构成性生态（Compositional Ecology）**：
- 智能体可将能量转化为**永久物理实体**
- 环境从"静态物理网格"升级为"图灵完备系统"
- 迫使系统长出 Supernode

---

## 2. 数学形式化

### 2.1 物质场 (Matter Field)

定义物质场 $F_{matter}(\mathbf{p}, t)$：
- 与能量场、阻抗场、痕迹场并列的离散场
- 高度稳定，不随时间挥发
- 会**截断其他所有场的动力学方程**

### 2.2 物质生成 (Materialization)

扩展动作空间 $U$：
$$\mathbf{u}_{total} = [\mathbf{u}_{move}, \mathbf{u}_{secrete}, \mathbf{u}_{build}, \mathbf{u}_{break}]$$

智能体 $i$ 在 $\mathbf{p}_i$ 处建造：
$$F_{matter}(\mathbf{p}_i, t+1) = F_{matter}(\mathbf{p}_i, t) + \mathbf{u}_{build}^{(i)}$$

**约束条件**：只有当个体能量 $E_i \gg E_{cost}$ 时允许执行

### 2.3 物质降维打击 (Occlusion)

当 $F_{matter}(\mathbf{p}) > \theta_{solid}$ 时：

1. **绝对阻抗**：
$$F_{impedance}(\mathbf{p}) \to \infty$$

2. **场扩散遮挡**（Stigmergy Occlusion）：
$$D_s \to 0 \quad \text{for } F_{stigmergy}$$

3. **感知遮挡**（Ray-casting）：
$$Sensor(\mathbf{p}_{agent}, \mathbf{p}_{target}) = 0 \text{ if } F_{matter}(\mathbf{p}_{ray}) > \theta_{solid}$$

### 2.4 构成性工具

多个 $F_{matter}$ 像素连接构成：
- 墙壁 (Walls)
- 死胡同 (Dead ends)
- 漏斗/陷阱 (Funnels/Traps)

**预期涌现**：捕食者建造漏斗驱赶猎物 → 大脑拓扑急剧扩张

---

## 3. 工程实现方案

### 3.1 任务总览

| 任务 | 优先级 | 难度 | GPU兼容 | 关键修复 |
|------|--------|------|---------|----------|
| T1: MatterGrid 基础设施 | P0 | 中 | ✅ | |
| T2: 建造/分解执行器节点 | P0 | 中 | ✅ | Review #2, #3 |
| T3: 碰撞检测升级 | P0 | 高 | ✅ | Review #4 |
| T4: Ray-casting 感知 | P1 | 高 | ✅ | |
| T5: 压痕场遮挡 | P1 | 中 | ✅ | Review #1 |
| T6: 风场环境（Phase 3） | P2 | 高 | ✅ | |

---

### ⚠️ Code Review 关键修复摘要

| Review | 问题 | 修复位置 | 影响 |
|--------|------|----------|------|
| **#1** | 压痕场穿墙（量子隧穿） | T5 Step 5.1 | 卷积前后双重掩码 |
| **#2** | 自我活埋 | T2 Step 2.3 | forward_dist > agent.radius |
| **#3** | 能量守恒破坏 | T2 Step 2.3 | 墙壁存储能量，分解全额返还 |
| **#4** | GPU并发竞争 | T3 Step 3.2 | 去重逻辑 + scatter处理 |

---

### T1: MatterGrid 基础设施

#### 目标
在 Environment 中引入持久化物质网格

#### 修改文件
- `core/eoe/environment.py`
- `core/eoe/environment_gpu.py`
- `core/eoe/fields/__init__.py` (新增)

#### 实现步骤

**Step 1.1: 在 Environment 类中添加 matter_grid**

```python
# core/eoe/environment.py - Environment.__init__

# v16.0: 构成性物质场
self.matter_grid_enabled = matter_grid_enabled
self.matter_resolution = matter_resolution  # 默认 1.0

if matter_grid_enabled:
    self.matter_grid_width = int(width / matter_resolution)
    self.matter_grid_height = int(height / matter_resolution)
    # Boolean grid: 0 = empty, 1 = solid matter
    self.matter_grid = np.zeros(
        (self.matter_grid_width, self.matter_grid_height), 
        dtype=np.int8
    )
    # ⚠️ CRITICAL: 能量存储网格 (与 matter_grid 并行)
    self.matter_energy = np.zeros(
        (self.matter_grid_width, self.matter_grid_height),
        dtype=np.float32
    )
    print(f"  [MatterGrid] Enabled: {self.matter_grid_width}x{self.matter_grid_height}")
    print(f"    Energy storage enabled for conservation")
else:
    self.matter_grid = None
    self.matter_energy = None
```

**Step 1.2: 在 GPU 环境类中添加 matter_grid（含能量存储）**

```python
# core/eoe/environment_gpu.py - EnvironmentGPU 类

# 添加参数
matter_grid_enabled: bool = False,
matter_resolution: float = 1.0,

# 在 __init__ 中
if matter_grid_enabled:
    self.matter_grid = torch.zeros(
        1, 1, self.grid_height, self.grid_width,
        device=device, dtype=torch.int8  # int8 for memory efficiency
    )
    # ⚠️ CRITICAL: 能量存储网格 (float32)
    self.matter_energy = torch.zeros(
        1, 1, self.grid_height, self.grid_width,
        device=device, dtype=torch.float32
    )
else:
    self.matter_grid = None
    self.matter_energy = None
```

**Step 1.3: 添加辅助方法（含能量存储）**

```python
# core/eoe/environment.py

def is_solid(self, x: float, y: float) -> bool:
    """检查坐标是否为固体"""
    if self.matter_grid is None:
        return False
    gx = int(x / self.matter_resolution) % self.matter_grid_width
    gy = int(y / self.matter_resolution) % self.matter_grid_height
    return self.matter_grid[gx, gy] == 1

def add_matter(self, x: float, y: float, stored_energy: float = 0.0) -> bool:
    """
    在指定坐标添加物质，返回是否成功
    
    Args:
        x, y: 目标坐标
        stored_energy: 物质中存储的能量（用于守恒）
    """
    if self.matter_grid is None:
        return False
    gx = int(x / self.matter_resolution) % self.matter_grid_width
    gy = int(y / self.matter_resolution) % self.matter_grid_height
    if self.matter_grid[gx, gy] == 0:
        self.matter_grid[gx, gy] = 1
        self.matter_energy[gx, gy] = stored_energy  # 存储能量
        return True
    return False

def remove_matter(self, x: float, y: float) -> bool:
    """移除指定坐标的物质"""
    if self.matter_grid is None:
        return False
    gx = int(x / self.matter_resolution) % self.matter_grid_width
    gy = int(y / self.matter_resolution) % self.matter_grid_height
    if self.matter_grid[gx, gy] == 1:
        self.matter_grid[gx, gy] = 0
        self.matter_energy[gx, gy] = 0.0  # 清空存储
        return True
    return False

def get_matter_energy(self, x: float, y: float) -> Optional[float]:
    """获取指定坐标物质存储的能量"""
    if self.matter_grid is None or self.matter_energy is None:
        return None
    gx = int(x / self.matter_resolution) % self.matter_grid_width
    gy = int(y / self.matter_resolution) % self.matter_grid_height
    if self.matter_grid[gx, gy] == 1:
        return self.matter_energy[gx, gy]
    return None
```

---

### T2: 建造/分解执行器节点

#### 目标
扩展 NodeType 和脑结构，使智能体能够建造和破坏物质

#### 修改文件
- `core/eoe/node.py`
- `core/eoe/brain_manager.py` (可能需要)
- `core/eoe/batched_agents.py`

#### 实现步骤

**Step 2.1: 在 NodeType 枚举中添加新节点**

```python
# core/eoe/node.py

class NodeType(Enum):
    # ... 现有节点 ...
    
    # v16.0: 构成性执行器
    ACTUATOR_CONSTRUCT = auto()   # 建造: 消耗能量生成物质块
    ACTUATOR_DECONSTRUCT = auto() # 分解: 破坏物质块回收少量能量
```

**Step 2.2: 添加建造/分解能量成本常量**

```python
# core/eoe/node.py 或新建 core/eoe/compositional_config.py

# 构成性动作能量参数
CONSTRUCT_ENERGY_COST = 15.0    # 建造消耗能量
CONSTRUCT_MIN_ENERGY = 25.0     # 建造所需最小能量
DECONSTRUCT_ENERGY_GAIN = 3.0   # 分解回收能量
DECONSTRUCT_COOLDOWN = 5        # 建造/分解冷却步数
```

**Step 2.3: 在 Node 类中添加执行逻辑**

```python
# core/eoe/node.py - Node 类新增方法

# v16.0: 物质能量存储 (用于全局能量守恒)
# MatterGrid 中的每个格子不仅存储 0/1，还存储其蕴含的能量

def execute_construct(self, env, agent) -> bool:
    """执行建造动作"""
    if self.node_type != NodeType.ACTUATOR_CONSTRUCT:
        return False
    if agent.energy < CONSTRUCT_MIN_ENERGY:
        return False
    
    # ⚠️ CRITICAL FIX (Review #2): 避免自我活埋
    # 建造距离必须严格大于智能体的物理半径
    forward_dist = agent.radius + env.matter_resolution
    target_x = agent.x + np.cos(agent.theta) * forward_dist
    target_y = agent.y + np.sin(agent.theta) * forward_dist
    
    # 检查是否已有物质
    if env.is_solid(target_x, target_y):
        return False
    
    # ⚠️ CRITICAL FIX (Review #3): 全局能量守恒
    # 智能体花费的能量存储在墙壁中，而非消失
    if env.add_matter(target_x, target_y, stored_energy=CONSTRUCT_ENERGY_COST):
        agent.energy -= CONSTRUCT_ENERGY_COST
        return True
    return False

def execute_deconstruct(self, env, agent) -> bool:
    """执行分解动作"""
    if self.node_type != NodeType.ACTUATOR_DECONSTRUCT:
        return False
    
    # 同理，避免自我活埋
    forward_dist = agent.radius + env.matter_resolution
    target_x = agent.x + np.cos(agent.theta) * forward_dist
    target_y = agent.y + np.sin(agent.theta) * forward_dist
    
    # 检查是否有物质可分解，并获取存储的能量
    stored_energy = env.get_matter_energy(target_x, target_y)
    if stored_energy is None:
        return False
    
    # 分解并回收存储的能量（完全守恒！）
    if env.remove_matter(target_x, target_y):
        agent.energy += stored_energy  # 而非 DECONSTRUCT_ENERGY_GAIN
        return True
    return False
```

**能量守恒补充说明**：
- 墙壁本身是一个"能量容器"
- 建造时：`agent.energy -= 15.0`，墙壁存储 `15.0`
- 分解时：墙壁返还 `15.0` 给智能体
- 可选拓展：墙壁随时间自然衰减（抵抗风沙），能量回归环境

**Step 2.4: 在脑激活循环中集成执行器**

需要在 `batched_agents.py` 的激活循环中添加对 `ACTUATOR_CONSTRUCT` 和 `ACTUATOR_DECONSTRUCT` 的处理：

```python
# core/eoe/batched_agents.py - 激活循环中新增

# 在处理执行器输出的部分
for node in actuator_nodes:
    if node.node_type == NodeType.ACTUATOR_CONSTRUCT:
        if node.activation > 0.5:  # 阈值触发
            node.execute_construct(env, agent)
    elif node.node_type == NodeType.ACTUATOR_DECONSTRUCT:
        if node.activation > 0.5:
            node.execute_deconstruct(env, agent)
```

---

### T3: 碰撞检测升级

#### 目标
让 matter_grid 拦截智能体运动

#### 修改文件
- `core/eoe/agent.py`
- `core/eoe/batched_agents.py`
- `core/eoe/environment.py`

#### 实现步骤

**Step 3.1: 在环境步进中集成碰撞检测**

```python
# core/eoe/environment.py - Environment.step 或 agent.move

def try_move_agent(self, agent, new_x: float, new_y: float) -> Tuple[float, float]:
    """尝试移动智能体，考虑matter_grid碰撞"""
    # 检查四个角是否碰撞
    corners = [
        (new_x - agent.radius, new_y - agent.radius),
        (new_x + agent.radius, new_y - agent.radius),
        (new_x - agent.radius, new_y + agent.radius),
        (new_x + agent.radius, new_y + agent.radius),
    ]
    
    for cx, cy in corners:
        if self.is_solid(cx, cy):
            # 发生碰撞，阻止移动
            return agent.x, agent.y
    
    # 无碰撞，正常移动
    return new_x, new_y
```

**Step 3.2: 在 GPU 环境中批量处理**

```python
# core/eoe/environment_gpu.py

def apply_matter_collision(self, positions: torch.Tensor) -> torch.Tensor:
    """
    批量应用物质碰撞
    positions: [N, 3] - x, y, theta
    返回: 掩码数组，1表示发生碰撞
    """
    if self.matter_grid is None:
        return torch.zeros(positions.shape[0], device=self.device)
    
    # 将位置转换为网格坐标
    gx = (positions[:, 0] / self.resolution).long() % self.matter_grid.shape[3]
    gy = (positions[:, 1] / self.resolution).long() % self.matter_grid.shape[2]
    
    # 采样 matter_grid
    collision = self.matter_grid[0, 0, gy, gx]
    return collision


# ⚠️ CRITICAL FIX (Review #4): GPU 并发竞争条件
def apply_construct_batch(
    self,
    construct_actions: torch.Tensor,  # [N, 3] - x, y, activation
    agent_energies: torch.Tensor      # [N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量处理建造动作，处理并发竞争
    
    Returns:
        (new_matter_grid, energy_changes)
    """
    if self.matter_grid is None:
        return self.matter_grid, torch.zeros_like(agent_energies)
    
    # 1. 过滤有效建造动作 (activation > 0.5 and energy >= min)
    valid_mask = (construct_actions[:, 2] > 0.5) & (agent_energies >= self.CONSTRUCT_MIN_ENERGY)
    
    if not valid_mask.any():
        return self.matter_grid, torch.zeros_like(agent_energies)
    
    # 2. 获取目标网格坐标
    target_x = (construct_actions[valid_mask, 0] / self.resolution).long() % self.matter_grid.shape[3]
    target_y = (construct_actions[valid_mask, 1] / self.resolution).long() % self.matter_grid.shape[2]
    
    # 3. ⚠️ 关键：去重处理 - 多个Agent对同一格子只执行一次
    # 创建唯一坐标索引
    grid_indices = target_y * self.matter_grid.shape[3] + target_x
    unique_indices, inverse_indices = torch.unique(grid_indices, return_inverse=True)
    
    # 4. 能量扣除（每个唯一格子只扣一次）
    energy_cost = self.CONSTRUCT_ENERGY_COST
    energy_changes = torch.zeros_like(agent_energies)
    energy_changes[valid_mask] = -energy_cost
    
    # 按 inverse_indices 分组求和，确保只扣一次
    unique_energy_changes = torch.zeros(unique_indices.shape[0], device=self.device) - energy_cost
    energy_changes[valid_mask] = unique_energy_changes[inverse_indices]
    
    # 5. 更新 matter_grid (使用 scatter)
    # 注意：这里需要创建副本，因为 torch.scatter 不支持 in-place boolean indexing
    new_grid = self.matter_grid.clone()
    for idx in unique_indices:
        gx = idx % self.matter_grid.shape[3]
        gy = idx // self.matter_grid.shape[3]
        if self.matter_grid[0, 0, gy, gx] == 0:  # 只在空格子建造
            new_grid[0, 0, gy, gx] = 1
    
    return new_grid, energy_changes
```

---

### T4: Ray-casting 感知

#### 目标
物质块遮挡智能体的视觉/感知输入

#### 实现方案

**Step 4.1: 添加光线投射函数**

```python
# core/eoe/perception.py 或 core/eoe/environment.py

def ray_cast(
    start: Tuple[float, float],
    direction: float,
    max_distance: float,
    env: 'Environment',
    resolution: float = 0.5
) -> Tuple[bool, float]:
    """
    从起点沿方向发射射线，返回是否击中固体及距离
    
    Returns:
        (hit_solid, distance)
    """
    steps = int(max_distance / resolution)
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    
    for i in range(steps):
        x = start[0] + cos_d * (i * resolution)
        y = start[1] + sin_d * (i * resolution)
        
        if env.is_solid(x, y):
            return True, i * resolution
    
    return False, max_distance
```

**Step 4.2: 修改感知输入以应用遮挡**

在 `perception.py` 或 `batched_agents.py` 中，当计算传感器输入时：

```python
# 伪代码示例
def compute_perception_with_occlusion(agent, env):
    # 对每个感知目标进行ray-cast
    for food in foods:
        hit, dist = ray_cast(
            (agent.x, agent.y), 
            angle_to(food),
            sensor_range,
            env
        )
        if hit:
            # 目标被遮挡，信号衰减为0
            sensor_input = 0.0
        else:
            # 正常计算
            sensor_input = compute_normal_sensor(dist, angle)
```

**Step 4.3: GPU 优化（使用 PyTorch）**

```python
# core/eoe/environment_gpu.py

def compute_occlusion_gpu(
    agent_positions: torch.Tensor,      # [N, 2]
    target_positions: torch.Tensor,     # [M, 2]
    matter_grid: torch.Tensor,          # [1, 1, H, W]
    max_distance: float = 20.0
) -> torch.Tensor:
    """
    GPU 批量计算遮挡 [N, M]
    """
    N = agent_positions.shape[0]
    M = target_positions.shape[0]
    
    # 计算方向向量
    dirs = target_positions.unsqueeze(0) - agent_positions.unsqueeze(1)  # [N, M, 2]
    dists = torch.norm(dirs, dim=-1)
    dirs = dirs / (dists.unsqueeze(-1) + 1e-6)
    
    # 简化的ray-march (步进采样)
    n_steps = int(max_distance / 1.0)  # 1.0单位步长
    occlusion = torch.zeros(N, M, device=matter_grid.device)
    
    for step in range(n_steps):
        sample_points = agent_positions.unsqueeze(1) + dirs * step  # [N, M, 2]
        
        # 转换为网格坐标
        gx = (sample_points[..., 0] / 1.0).long() % matter_grid.shape[3]
        gy = (sample_points[..., 1] / 1.0).long() % matter_grid.shape[2]
        
        # 检查是否有物质
        hit = matter_grid[0, 0, gy, gx]
        occlusion = occlusion | hit
    
    return occlusion
```

---

### T5: 压痕场遮挡

#### 目标
物质块阻断信息素的扩散

#### ⚠️ CRITICAL BUG FIX (Review #1): 穿墙/量子隧穿问题

**问题分析**：如果只在 `F.conv2d` **之后**将墙壁位置清零，信息素会通过3x3卷积核"穿透"1像素宽的墙。

**修正方案**：必须在卷积**之前和之后**都应用掩码。

```python
# core/eoe/stigmergy_field.py - step 方法中

def step(self, matter_grid=None):
    # v16.0: 应用物质遮挡 (防止量子隧穿)
    if matter_grid is not None:
        mask = (matter_grid == 0)  # 1 where no matter
        # Step 1: 先把墙壁内部的源清零（防止墙后接收墙前扩散）
        masked_field = self.field * mask
        # Step 2: 执行扩散
        new_field = self._diffuse(masked_field)
        # Step 3: 再把扩散到墙壁上的浓度清零（防止墙前受墙后回流）
        new_field = new_field * mask
    else:
        new_field = self._diffuse(self.field)
    
    self.field = new_field * self.decay_rate
```

#### 原始实现步骤

**Step 5.1: 修改扩散方程**

**Step 5.2: 在GPU版本中使用掩码**

```python
# core/eoe/environment_gpu.py - StigmergyFieldGPU

def step(self):
    # 原有扩散 (使用 conv2d)
    new_field = F.conv2d(
        self.field, 
        self.diffusion_kernel, 
        padding=1
    )
    
    # v16.0: 应用物质掩码
    if self.matter_grid is not None:
        mask = (self.matter_grid == 0).float()
        new_field = new_field * mask
    
    self.field = new_field * self.decay_rate
```

---

### T6: 风场环境（Phase 3 测试）

#### 目标
创建需要"挡风墙"才能生存的环境

#### 修改文件
- `core/eoe/environment.py` 或 `core/eoe/fields/wind.py` (新增)
- `core/eoe/environment_gpu.py`

#### 实现方案

**Step 6.1: 定义风场**

```python
# core/eoe/fields/wind.py (新增)

class WindField:
    """环境风场 - 对智能体造成持续伤害"""
    
    def __init__(
        self,
        direction: float = 0.0,        # 风向 (弧度)
        base_speed: float = 5.0,       # 基础风速
        damage_rate: float = 0.1,      # 每步伤害
        enabled: bool = True
    ):
        self.direction = direction
        self.base_speed = base_speed
        self.damage_rate = damage_rate
        self.enabled = enabled
    
    def get_damage(self, agent, env) -> float:
        """计算智能体受到的风伤害"""
        if not self.enabled:
            return 0.0
        
        # 检查智能体是否在"风中"（无遮挡）
        # 使用 ray-cast 检查风向是否有物质遮挡
        hit, dist = env.ray_cast(
            (agent.x, agent.y),
            self.direction,
            max_distance=50.0
        )
        
        if hit:
            # 有遮挡，无伤害
            return 0.0
        else:
            # 暴露在风中，有伤害
            return self.damage_rate
```

**Step 6.2: 在环境中集成风场**

```python
# core/eoe/environment.py

def __init__(self, ...):
    # ... 现有代码 ...
    
    # v16.0: 风场
    self.wind_field_enabled = wind_field_enabled
    if wind_field_enabled:
        self.wind_field = WindField(
            direction=wind_direction,
            base_speed=wind_speed,
            damage_rate=wind_damage_rate
        )
        print(f"  [WindField] Enabled: direction={wind_direction}, damage={wind_damage_rate}")

def step(self, ...):
    # ... 现有代码 ...
    
    # v16.0: 应用风场伤害
    if self.wind_field_enabled:
        for agent in self.agents:
            damage = self.wind_field.get_damage(agent, self)
            agent.energy -= damage
```

---

## 4. 分阶段实施计划

### Phase 1: 静态墙壁测试 (1-2周)

**目标**: 验证基础架构正确性

**任务**:
1. [ ] 实现 MatterGrid 基础设施 (T1)
2. [ ] 硬编码 U 型墙壁用于测试
3. [ ] 验证碰撞检测正确阻止移动
4. [ ] 验证压痕场扩散被墙壁阻断
5. [ ] 验证智能体在墙角饿死

**验收标准**:
- 智能体无法穿越墙壁
- 压痕场信号被墙壁阻挡

**测试命令**:
```bash
cd eoe_mvp
python -c "
from core.eoe.environment import Environment
env = Environment(width=50, height=50, matter_grid_enabled=True, n_walls=0)
# 手动放置U型墙
env.add_matter(20, 25)
env.add_matter(20, 26)
# ... 继续
print('MatterGrid test passed')
"
```

---

### Phase 2: 单节点建造测试 (2-3周)

**目标**: 验证智能体可以学习建造

**任务**:
1. [ ] 添加 ACTUATOR_CONSTRUCT 和 ACTUATOR_DECONSTRUCT 节点 (T2)
2. [ ] 扩展脑激活循环处理建造/分解 (T2)
3. [ ] 短周期实验（几百代）观察随机建造行为

**验收标准**:
- 智能体能够在前方建造物质块
- 能量不足时无法建造
- 智能体能够分解物质块回收能量

**测试命令**:
```bash
python scripts/test_construct_behavior.py --generations 500
```

---

### Phase 3: 挡风墙生态位演化 (3-4周)

**目标**: 观测到"人工生命使用工具"的现象

**任务**:
1. [ ] 实现 WindField (T6)
2. [ ] 集成风场伤害到环境步进
3. [ ] 长周期演化实验（数千代）
4. [ ] 监控智能体学会"挡风"的证据

**验收标准**:
- 智能体学会在迎风面建造墙壁
- 墙壁显著提高存活率
- 复杂结构（Supernode）涌现

**预期观测指标**:
- `analyze_champion.py` 捕获 "Construct" 节点激活模式
- 存活智能体的平均脑容量显著增加
- 出现"墙体包围"结构

---

## 5. 配置文件

### 5.1 新增配置项

```yaml
# config/compositional_v16.yaml

# v16.0 构成性环境
matter_grid:
  enabled: true
  resolution: 1.0
  initial_walls: 0  # 用于测试
  
# 建造/分解参数
construction:
  energy_cost: 15.0       # 建造消耗
  min_energy: 25.0        # 最低能量要求
  deconstruct_gain: 3.0   # 分解回收
  cooldown: 5             # 冷却步数
  
# 风场参数 (Phase 3)
wind_field:
  enabled: false           # 默认关闭
  direction: 0.0           # 弧度 (0 = 东)
  base_speed: 5.0
  damage_rate: 0.1         # 每步伤害
```

### 5.2 PoolConfig 扩展

```python
# core/eoe/batched_agents.py - PoolConfig

@dataclass
class PoolConfig:
    # ... 现有配置 ...
    
    # v16.0 构成性环境
    MATTER_GRID_ENABLED: bool = False
    MATTER_RESOLUTION: float = 1.0
    
    CONSTRUCT_ENERGY_COST: float = 15.0
    CONSTRUCT_MIN_ENERGY: float = 25.0
    DECONSTRUCT_ENERGY_GAIN: float = 3.0
    
    WIND_FIELD_ENABLED: bool = False
    WIND_DIRECTION: float = 0.0
    WIND_DAMAGE_RATE: float = 0.1
```

---

## 6. 风险与缓解

### 6.1 性能风险

**风险**: MatterGrid 碰撞检测引入 CPU 开销

**缓解**:
- 优先实现 GPU 版本 (environment_gpu.py)
- 使用张量掩码而非逐点检查
- 考虑空间哈希优化

### 6.2 演化停滞风险

**风险**: 建造消耗过高，无人演化

**缓解**:
- Phase 1 先用低成本测试
- 调整 CONSTRUCT_ENERGY_COST 至合理范围
- 考虑"免费初始脑容量"类似的免费建造次数

### 6.3 网格边界风险

**风险**: 环形世界边界处理

**缓解**:
- 使用取模运算处理坐标
- 测试角角落落的建造行为

---

### 🔬 前瞻性观测建议：搭便车效应 (Free Rider Problem)

**重要警告**：在 Phase 3（挡风墙实验）中，除了监控 Supernode 涌现外，务必警惕**寄生与搭便车**现象：

**预期涌现的社会学模式**：

1. **先驱者 (Pioneers)**：少数智能体演化出 `ACTUATOR_CONSTRUCT`，消耗能量造墙
2. **剥削者 (Exploiters)**：演化出极简大脑 + 高速移动，不造墙，专门寻找先驱者的墙躲避
3. **领地竞争**：造墙者需演化"驱赶"行为或"环形封闭墙"（只把自己关在里面）

**这不是 Bug，是_feature_**：
- 真实自然界就是如此（植物光合作用 vs 寄生藤蔓）
- 搭便车者逼迫造墙者演化更复杂的行为（领地防守、选择配偶等）
- 形成**无限上升的演化螺旋** → 推动脑容量爆发

**分析脚本扩展建议**：
```python
# scripts/analyze_champion.py - 新增监控
def analyze_social_parasitism(population):
    """监控搭便车效应"""
    builders = count_agents_with_node(ACTUATOR_CONSTRUCT)
    exploiters = count_fast_movers_without_construct()
    wall_users = count_agents_near_matter_grid()
    
    return {
        'builder_ratio': builders / total,
        'exploiter_ratio': exploiters / total,
        'wall_utilization': wall_users / total_walls
    }
```

---

## 7. 预期效果

| 指标 | v15 | v16 预期 |
|------|-----|----------|
| 环境复杂度 | 静态场 | 可编辑拓扑 |
| 智能体行为 | 寻路+觅食 | 筑墙+陷阱 |
| 脑容量上限 | ~20节点 | 无上限 |
| 涌现速度 | >300代 | >200代 (挡风墙) |
| 研究价值 | 认知溢价 | **工具使用** ⭐ |

---

## 8. 里程碑

- [ ] **M1** (Week 1): MatterGrid 基础设施完成
- [ ] **M2** (Week 2): 碰撞检测+压痕场遮挡完成
- [ ] **M3** (Week 3): 建造/分解节点完成
- [ ] **M4** (Week 4): Ray-casting 感知完成
- [ ] **M5** (Week 5-6): Phase 3 挡风墙实验
- [ ] **M6** (Week 7): 数据分析与论文材料

---

## 附录: 相关文件清单

### 新增文件
- `core/eoe/fields/matter.py` - 物质场类
- `core/eoe/fields/wind.py` - 风场类
- `scripts/test_matter_grid.py` - 物质网格测试
- `scripts/test_construct_behavior.py` - 建造行为测试
- `scripts/test_wind_barrier.py` - 挡风墙测试

### 修改文件
- `core/eoe/environment.py` - 添加 matter_grid, wind_field
- `core/eoe/environment_gpu.py` - GPU 版本支持
- `core/eoe/node.py` - 新增 ACTUATOR_CONSTRUCT/DECONSTRUCT
- `core/eoe/batched_agents.py` - 配置项扩展
- `core/eoe/perception.py` - 添加 ray-cast 感知
- `core/eoe/stigmergy_field.py` - 添加遮挡逻辑

---

*EOF - 等待审阅后实施*