# 具身运动学映射 (Embodied Kinematics Grounding) 提案

> 核心任务一：闭环物理学 | 日期: 2026-03-13

---

## 1. 背景

SENSOR 和 ACTUATOR 已完成数据结构升级：
- **SENSOR**: `receptor_key` [C] + `spatial_offset` [2]
- **ACTUATOR**: `emitter_key` [C] + `spatial_offset` [2]

但这些数据目前只是静态属性，还没有真正连接到物理世界。

---

## 2. Channel 物理语义定义

| Channel | 名称 | 物理含义 | 涌现行为 |
|---------|------|----------|----------|
| 0 | ENERGY | 推力/动量 | 前进/后退 |
| 1 | IMPEDANCE | 局部摩擦/硬度 | 防御/装甲 |
| 2 | STIGMERGY | 信息素排放 | 标记/追踪 |
| 3 | STRESS | 摄食/攻击 | 能量抽取 |

---

## 3. 数学模型

### 3.1 Actuator 输出分解

```
actuator_outputs: [N, A]     # A = actuator 数量
emitter_keys: [N, A, C]      # C = 4 通道
spatial_offsets: [N, A, 2]   # 相对 Agent 重心的位置
```

### 3.2 力提取 (Channel 0)

```python
# 只看 Channel 0 的推力
thrusts = actuator_outputs * emitter_keys[:, :, 0]  # [N, A]
```

### 3.3 线加速度 (非全向移动模型)

采用类似坦克/汽车的非全向移动模型：推力沿局部 x 轴（前方），速度为标量。

```python
# 局部坐标系定义：x轴=前方，y轴=左侧
# forward_thrusts: [N, A] 保留正负号（正=前进，负=后退）
forward_thrusts = thrusts  # 推力已在局部 x 方向

# 线加速度 = 净推力代数和 (允许负值，实现倒车/刹车)
# a = F / m (假设质量 m=1)
linear_accel = forward_thrusts.sum(dim=1)  # [N], 负值代表向后加速
```

### 3.4 角加速度 (力矩计算)

力矩 = 力 × 力臂（2D 叉乘的标量形式）

**关键**：推力沿 x 轴（纵向），产生旋转需要**横向力臂**（y 轴偏移）。

```python
# spatial_offset[:, :, 0] = 前后偏移 (x)
# spatial_offset[:, :, 1] = 左右偏移 (y)
#
# 物理意义：
# - 左侧 (y > 0) 向前推 -> 产生负力矩 (顺时针右转)
# - 右侧 (y < 0) 向前推 -> 产生正力矩 (逆时针左转)
# - 向前推 (x > 0) 不产生偏航力矩

torques = -forward_thrusts * spatial_offsets[:, :, 1]  # [N, A]
total_torque = torques.sum(dim=1)  # [N]

# 角加速度 alpha = torque / I (假设转动惯量 I=1)
angular_accel = total_torque  # [N]
```

### 3.5 物理阻尼 (Damping)

真实环境需要阻尼防止"永动滑行"：

```python
# manifest.py 中定义
LINEAR_DAMPING = 0.1    # 线速度阻尼系数
ANGULAR_DAMPING = 0.2   # 角速度阻尼系数
```

```python
# 物理积分时应用阻尼
def apply_actuator_physics(self, linear_accel, angular_accel, dt=0.1):
    # 1. 阻尼衰减当前速度
    self.velocity *= (1.0 - LINEAR_DAMPING * dt)
    self.angular_velocity *= (1.0 - ANGULAR_DAMPING * dt)

    # 2. 加入新的加速度
    self.velocity += linear_accel * dt
    self.angular_velocity += angular_accel * dt

    # 3. 位置更新 (非全向移动)
    self.positions[:, 0] += self.velocity * torch.cos(self.theta) * dt
    self.positions[:, 1] += self.velocity * torch.sin(self.theta) * dt
    self.theta += self.angular_velocity * dt
```

### 3.6 能量代谢 (Metabolic Cost)

Actuator 做功需要消耗能量，否则会演化出"永动机"：

```python
# manifest.py 中定义
ACTUATOR_COST_RATE = 0.01  # 单位推力做功耗能率

def compute_metabolic_cost(actuator_outputs, dt):
    """
    计算代谢能耗
    - 做功耗能 = |推力| × 肌肉耗能系数 (无论正推反推都要耗能)
    """
    # 绝对做功量
    work_done = torch.abs(actuator_outputs).sum(dim=1)  # [N]
    metabolic_cost = work_done * ACTUATOR_COST_RATE  # [N]
    return metabolic_cost * dt
```

### 3.7 速度限制 (Clamping)

```python
# 速度限制 (包含正负范围)
max_speed = 10.0
max_angular_speed = 5.0

velocity = torch.clamp(velocity, -max_speed, max_speed)
angular_velocity = torch.clamp(angular_velocity, -max_angular_speed, max_angular_speed)
```

> **注意**：由于使用了阻尼机制，速度限制主要用于安全防护，正常情况下阻尼会自动将速度衰减到零。

---

## 4. 涌现预期

| 形态 | 行为 | 解释 |
|------|------|------|
| 单侧 Actuator | 原地打转 | 力矩不平衡 |
| 双侧对称 Actuator | 直线冲刺 | 力矩平衡，净推力最大 |
| 前重后轻 | 倒车/倒退 | 质心偏移 |
| 交替脉冲 | 摆动/游动 | 时序控制 |

---

## 5. 实施计划

### Phase 1: 数据结构扩展

**文件**: `batched_agents.py`

```python
class BatchedAgents:
    def __init__(self, ...):
        # 新增状态
        self.velocity = torch.zeros(n_agents, device=device)      # 线速度
        self.angular_velocity = torch.zeros(n_agents, device=device)  # 角速度
        self.theta = torch.zeros(n_agents, device=device)         # 朝向角
```

### Phase 2: Actuator 解析

**文件**: `batched_agents.py`

```python
def decode_actuators(
    self,
    actuator_outputs: torch.Tensor,      # [N, A] 神经网络输出
    emitter_keys: torch.Tensor,          # [N, A, C]
    spatial_offsets: torch.Tensor        # [N, A, 2]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    解析 Actuator 输出为物理量
    
    Returns:
        linear_accel: [N] 线加速度
        angular_accel: [N] 角加速度
        channel_outputs: dict 各通道输出
    """
```

### Phase 3: 物理积分

**文件**: `batched_agents.py`

```python
def apply_actuator_physics(
    self,
    linear_accel: torch.Tensor,
    angular_accel: torch.Tensor,
    dt: float = 0.1
):
    """应用物理积分"""
    # 更新速度
    self.velocity += linear_accel * dt
    self.angular_velocity += angular_accel * dt
    
    # 位置更新
    self.positions += ...
    self.theta += ...
```

### Phase 4: 环境交互

**文件**: `batched_agents.py`

```python
def apply_environment_effects(
    self,
    channel_outputs: dict  # 各通道输出
):
    """
    Channel 1: 阻力场调制
    Channel 2: 信息素排放
    Channel 3: 攻击/摄食
    """
```

---

## 6. 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `core/eoe/batched_agents.py` | 新增动力学方法 |
| `core/eoe/manifest.py` | 添加物理常数 |
| `core/eoe/fields.py` | Channel 2/3 环境交互 |

---

## 7. 验证方式

```python
# 测试用例 1: 单侧 Actuator 应该产生旋转
actuator_outputs = torch.tensor([[1.0], [1.0]])  # 单侧发力
emitter_keys = torch.tensor([[[1.0, 0, 0, 0]]])  # Channel 0
offsets = torch.tensor([[[0.0, 1.0]]])           # 偏移在左侧 (y=1.0)

# forward_thrusts = [1.0]
# torques = -1.0 * 1.0 = -1.0 (负=顺时针右转)
# 预期: angular_accel < 0 (向左转，即顺时针)

# 测试用例 2: 双侧对称 Actuator 同向发力应该直线前进
actuator_outputs = torch.tensor([[1.0], [-1.0]])  # 两侧同时向前
emitter_keys = torch.tensor([[[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]]])
offsets = torch.tensor([[[0.0, -1.0]], [[0.0, 1.0]]])  # 左右对称

# forward_thrusts = [1.0, -1.0] (注意第二维是负，因为是 -1.0 * 1.0 = -1.0)
# 预期: linear_accel = 0 (净推力为零，无法前进)

# 测试用例 3: 双侧对称 Actuator 反向发力应该产生旋转
actuator_outputs = torch.tensor([[1.0], [1.0]])  # 两侧同时向前
emitter_keys = torch.tensor([[[1.0, 0, 0, 0]], [[1.0, 0, 0, 0]]])
offsets = torch.tensor([[[0.0, -1.0]], [[0.0, 1.0]]])  # 左右对称

# forward_thrusts = [1.0, 1.0]
# torques = -1.0 * (-1.0) + -1.0 * 1.0 = 1.0 - 1.0 = 0
# 预期: angular_accel = 0 (力矩平衡)
# 预期: linear_accel = 2.0 (净推力向前)
```

## 8. 修复记录

| 日期 | 修改 | 原因 |
|------|------|------|
| 2026-03-13 | 非全向移动模型 | 修复模长永远为正的倒车 Bug |
| 2026-03-13 | 力矩使用 y 轴偏移 | 修复力矩轴向错位 |
| 2026-03-13 | 添加阻尼机制 | 防止无限滑行 |
| 2026-03-13 | 添加能量代谢 | 防止永动机演化 |

---

## 9. 风险与注意事项

1. **数值稳定性**: 限制最大速度防止爆炸（现在包含正负范围）
2. **边界条件**: 处理碰到边界时的反弹
3. **涌现时间**: Agent 可能需要多代演化才能学会协调
4. **能量耗尽**: 添加代谢成本后，Agent 可能因能量耗尽而死亡，需要在环境反馈中补充能量来源（如 Channel 3 的摄食机制）

## 10. 预期涌现行为（修复后）

| 形态 | 行为 | 解释 |
|------|------|------|
| 单侧 Actuator | 原地打转 | 力矩不平衡 |
| 双侧对称 Actuator | 直线冲刺 | 力矩平衡，净推力最大 |
| 交替脉冲 | 摆动/游动 | 时序控制 |
| 能量管理 | 脉冲式移动 | 做功耗能迫使 Agent 节省能量 |