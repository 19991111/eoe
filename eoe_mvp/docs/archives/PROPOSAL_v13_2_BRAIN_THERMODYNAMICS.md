# EOE v13.2 提案：计算代谢率与环境场反向写入

> 版本: v13.2 | 日期: 2026-03-13 | 状态: 待审批

---

## 1. 概述

本文档描述 EOE 系统新增的两个核心机制：

1. **大脑热力学 (Brain Thermodynamics)** - 通过能量代谢压力自然淘汰臃肿大脑
2. **环境场反向写入 (Field Emission)** - Agent 主动塑造环境而非被动响应

这两个机制将 EOE 从"数据结构的空转"推进到"人工生命的自然演化"。

---

## 2. 动机

### 2.1 问题一：大脑膨胀

**现象**：没有代谢惩罚时，演化算法会迅速生成包含数百个节点的庞大网络。

**原因**：
- 更多的节点 = 更大的决策空间
- 但没有任何"代价"约束
- GPU 显存会被迅速撑爆

**解决思路**：不设硬性 `max_nodes` 限制，而是用热力学法则让臃肿大脑自然饿死。

### 2.2 问题二：被动 Agent

**现象**：Agent 只能读取环境场，被动移动，没有"主动觅食"行为。

**原因**：
- Channel 2 (信息素) 和 Channel 3 (摄食) 未实现
- Agent 无法留下痕迹或主动获取能量

**解决思路**：让 Agent 可以写入环境（留下信息素）和主动汲取能量（摄食）。

---

## 3. 方案设计

### 3.1 大脑热力学

#### 3.1.1 节点静态代谢成本

每个节点每步消耗能量（模拟生物体维持细胞存活的 ATP 消耗）：

| 节点类型 | 成本/步 | 理由 |
|---------|---------|------|
| CONSTANT | 0.001 | 极低 - 简单常数存储 |
| ADD | 0.001 | 极低 - 简单加法门 |
| MULTIPLY | 0.002 | 低 - 乘法略复杂 |
| THRESHOLD | 0.001 | 极低 - 简单比较 |
| DELAY | 0.005 | 中等 - 维持记忆状态 |
| SENSOR | 0.01 | 高 - 感官器官活跃 |
| ACTUATOR | 0.02 | 极高 - 肌肉细胞耗能 |

#### 3.1.2 连线成本

每条有效连接（权重非零）消耗 `edge_cost = 0.0005`/步。

#### 3.1.3 动态激活成本

Actuator 输出时额外消耗能量（模拟肌肉发力）：

```
activation_cost = |actuator_output| × activation_multiplier (0.1)
```

**关键约束**：如果 Agent 演化出"持续满功率抽搐"的 Actuator，会因能量耗尽而死。

#### 3.1.4 公式

```
total_metabolism = Σ(node_count[i] × node_cost[i]) + edge_count × edge_cost + Σ|actuator_output| × activation_multiplier
```

---

### 3.2 环境场反向写入

#### 3.2.1 Channel 2：信息素释放 (Stigmergy)

**机制**：
- Agent 的 Actuator Channel 2 输出 → 环境信息素场
- 写入位置：Agent 当前全局坐标
- 写入量：`stigmergy × 0.1`

**代码**：
```python
if hasattr(env, 'stigmergy_field'):
    env.stigmergy_field.deposit_batch(positions, stigmergy * 0.1)
```

**涌现预期**：
- 部分 Agent 演化出"边走边留痕迹"的行为
- 其他 Agent 的 SENSOR 演化出追踪这种痕迹
- 原始社会群集行为涌现

#### 3.2.2 Channel 3：能量场主动汲取 (Active Eating)

**机制**：
- Agent 的 Actuator Channel 3 (stress) 控制"摄食强度"
- 能量获取 = `stress × field_value × extract_rate (0.3)`
- 只在 `stress > 0` 时生效

**约束**：
- `stress < 0` 为防御模式（减少被攻击伤害）
- 持续激活 Channel 3 会因"动态激活成本"而亏本

**涌现预期**：
- Agent 必须演化出"感知高能量场 → 激活摄食 Actuator → 补充能量"的神经反射回路
- 不"张嘴"的 Agent 无法获取能量，过度"张嘴"的 Agent 因能耗过高而死

---

## 4. 实现细节

### 4.1 新增文件

**`core/eoe/brain_thermodynamics.py`**：
- `BrainThermodynamics` 类
- `compute_static_cost()` - 节点维护成本
- `compute_edge_cost()` - 连线消耗
- `compute_activation_cost()` - 激活惩罚
- `compute_total_metabolism()` - 总代谢
- `apply_to_energies()` - 扣除能量

### 4.2 manifest.py 新增参数

```python
# 大脑热力学
node_cost_constant: float = 0.001
node_cost_add: float = 0.001
node_cost_multiply: float = 0.002
node_cost_threshold: float = 0.001
node_cost_delay: float = 0.005
node_cost_sensor: float = 0.01
node_cost_actuator: float = 0.02
edge_cost: float = 0.0005
actuator_activation_cost_multiplier: float = 0.1
```

### 4.3 batched_agents.py 扩展

- `apply_channel_environment()` 新增能量场汲取逻辑

---

## 5. 测试验证

### 5.1 大脑热力学测试

```
简单大脑 (3节点):   0.10 能量/步
中等大脑 (12节点):  0.27 能量/步
臃肿大脑 (45节点):  0.45 能量/步 (4.5x 简单大脑)
```

### 5.2 多通道物理测试

```
Channel 0 ENERGY: 前进/倒车/转向 ✅
Channel 1 IMPEDANCE: 高阻尼减速 ✅
Channel 2 STIGMERGY: 信息素写入 ✅
Channel 3 STRESS: 能量转移 + 场汲取 ✅
```

---

## 6. 潜在问题与风险

### 6.1 代谢成本参数可能需要调优

**问题**：当前参数是估算值，可能导致：
- 成本过高 → 所有 Agent 迅速饿死
- 成本过低 → 无法抑制大脑膨胀

**缓解**：参数可通过 manifest 外部配置，建议在演化过程中动态调整。

---

## 7. 架构优化 (v13.2.1 更新)

> ⚠️ **审批意见：带条件批准**
> 需采纳以下架构优化以保证 GPU 并行极致性能

### 7.1 改进一：严禁 O(N²) —— 场媒介交互法则

**问题**：Agent 到 Agent 的直接距离计算是性能杀手。

**解决**：一切交互通过 `env_tensor` 中介：
- **攻击 = 写入危险场**：Channel 4 (攻击) 向 `env.danger_field` 写入高强度尖峰
- **受击 = 读取危险场**：每步从 `danger_field` 读取值并扣血
- **计算复杂度**：O(N) 的张量散点叠加 (`scatter_add_`)

```python
# 攻击: 向危险场写入伤害
danger_field[grid_y, grid_x] += attack_strength * damage

# 受击: 读取危险场并扣血
danger = danger_field[agent_grid_y, agent_grid_x]
energies -= danger * damage_rate
```

### 7.2 改进二：静态代谢预编译 (BMR)

**问题**：每步遍历节点计算能耗会把 GPU 拖垮。

**解决**：Agent 初始化时一次性计算 BMR：

```python
# 基因初始化时 (O(1) 查表)
agent_bmr = (
    node_count[CONSTANT] * 0.001 +
    node_count[ADD] * 0.001 +
    node_count[MULTIPLY] * 0.002 +
    ...
    edge_count * 0.0005
)

# 运行时 (O(N) 张量运算)
energies -= agent_bmr_tensor + (actuator_outputs.abs() * activation_multiplier)
```

### 7.3 改进三：Channel 3/4 语义解耦

**问题**：Channel 3 同时承担摄食和攻击，演化梯度不平滑。

**解决**：扩展为 5 通道：

| Channel | 名称 | 物理效果 |
|---------|------|----------|
| 0 | THRUST | 推力 → 位移 |
| 1 | ARMOR | 装甲 → 伤害减免 |
| 2 | PHEROMONE | 信息素 → 环境写入 |
| 3 | FEED | 摄食 → 从能量场汲取 |
| 4 | ATTACK | 攻击 → 向危险场写入伤害 |

**能量守恒**：Agent 吸取的能量必须从环境场等量扣除。

---

## 8. 待集成项

要将此提案集成到主循环，需要：

1. **主循环调用代谢计算**：在每个 step 结束后调用 `BrainThermodynamics.compute_total_metabolism()`
2. **能量扣除**：将代谢成本从 Agent 能量中扣除
3. **死亡判定**：能量 ≤ 0 的 Agent 标记为死亡
4. **演化筛选**：死亡 Agent 不参与下一代繁殖

---

## 8. 审批意见

**[待填写]**

---

## 9. 变更日志

| 日期 | 变更 |
|------|------|
| 2026-03-13 | 初始提案 |