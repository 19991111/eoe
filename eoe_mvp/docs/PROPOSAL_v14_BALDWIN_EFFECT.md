# PROPOSAL v14: 鲍德温效应 + 演化棘轮

> 审批人: 陆正旭  
> 日期: 2026-03-13  
> 状态: 待审批

---

## 背景痛点

### 方向二：鲍德温效应

**现状**: Agent大脑权重在出生时固定，纯靠基因突变适应环境  
**问题**: "迷宫"类动态环境需要终生学习能力  
**机会**: 塑性变异(plasticity)机制已存在但未真正激活

### 方向三：演化棘轮

**现状**: SUPERNODE是5%随机生成的"代金券"  
**问题**: 违背生物学模块化本质，超级节点应从涌现中涌现  
**机会**: 真实子图冻结 → 代谢优惠 → 加速演化

---

## 方案一：能量调制赫布学习 (Reward-modulated STDP)

### 核心机制

```python
# 伪代码: energy_modulated_hebbian_update
def energy_modulated_hebbian_update(pre_id, post_id, energy_delta, lr=0.01):
    """
    能量梯度驱动的赫布学习
    
    如果这步获得了能量(+ΔE) → 强化过去5步内活跃的连接
    如果这步损失了能量(-ΔE) → 弱化过去5步内活跃的连接
    
    这就是"学会"一件事的本质：
    - 传感器感知到食物气味
    - 执行器采取行动
    - 能量增加 → 连接被强化
    - 下次遇到同样情况，Agent更倾向于采取相同行动
    """
    # 多巴胺信号 = 能量变化率
    dopamine = sign(energy_delta) * min(abs(energy_delta) / 50.0, 1.0)
    
    # Hebbian update with dopamine modulation
    # "fire together, wire together" + reward signal
    delta = lr * pre_act * post_act * dopamine
    
    # 强化: 能量增加时，增强正向关联；减少时，减弱
    weight += delta
```

### 涌现预期

| 阶段 | 涌现行为 |
|------|----------|
| 早期 | Agent随机探索，碰巧吃到食物时强化连接 |
| 中期 | 记住"死胡同"不应走，记住"食物源"应该走 |
| 后期 | 无需基因突变，大脑学会适应新迷宫 |

这正是**鲍德温效应**：学习能力成为演化的催化剂！

### 技术实现

| 文件 | 改动 |
|------|------|
| `core/eoe/batched_agents.py` | 追踪 energy_delta，新方法 `reward_hebbian_step()` |
| `core/eoe/brain_thermodynamics.py` | 添加 reward_modulated_stdp() 函数 |
| `core/eoe/environment.py` | CPU版本: 集成能量调制赫布 |
| `tests/test_hebbian_learning.py` | 迷宫环境测试 |

### 参数配置

```yaml
hebbian_learning:
  enabled: true
  base_lr: 0.01              # 基础学习率
  reward_modulation: true    # 能量调制开关
  eligibility_trace: 5       # 5步 eligibility trace
  plasticity_probability: 0.3  # 突变时启用可塑性概率
```

---

## 方案二：运行时子图挖掘与 Supernode 冻结

### 核心机制

```python
# 伪代码: discover_and_freeze_supernode
def discover_and_freeze_supernode(agents: List[Agent], top_k: int = 50):
    """
    后台模式识别 + 基因封装
    
    1. 收集 Top 10% Elite Agent 的大脑拓扑
    2. 运行频繁子图挖掘 (gSpan 算法)
    3. 找出高度保守的子结构 (出现率 > 30%)
    4. 提取为新 SuperNode 注册到算子库
    5. 赋予代谢优惠 (成本 = 组件总和 × 0.7)
    """
    
    # 扫描拓扑
    subgraph_patterns = gspan_mine(elite_brains, min_support=0.3)
    
    for pattern in subgraph_patterns:
        # 检查是否已存在
        if not pattern_exists(pattern):
            # 注册新 Supernode
            register_supernode(
                name=f"SUPERNODE_{len(supernodes)}",
                topology=pattern,
                cost=calculate_component_cost(pattern) * 0.7  # 7折优惠!
            )
            
            # 替换种群中所有该模式为 SuperNode
            compress_agents(agents, pattern)
```

### 涌现预期

| 轮次 | 系统级涌现 |
|------|----------|
| 第1轮 | 3节点"雷达"(Sensor→Threshold→Actuator)被冻结 |
| 第2轮 | 4节点"导航模块"(方向估计)被冻结 |
| 第3轮 | 嵌套SuperNode形成层级推理 |

这才是真正的**演化棘轮**！

### 技术实现

| 文件 | 改动 |
|------|------|
| `core/eoe/subgraph_miner.py` | 新文件: gSpan子图挖掘 |
| `core/eoe/supernode_registry.py` | 新文件: SuperNode动态注册表 |
| `core/eoe/genome.py` | 添加 `can_compress_to_supernode()` |
| `core/eoe/batched_agents.py` | 后台定期触发挖掘 |
| `config/supernode_config.yaml` | 挖掘参数 |

### 参数配置

```yaml
subgraph_mining:
  enabled: true
  interval_steps: 1000       # 每1000步运行一次
  top_k_ratio: 0.1           # Top 10% Elite
  min_support: 0.3           # 出现率 > 30%
  min_size: 3                # 至少3节点
  max_size: 5                # 最多5节点
  cost_discount: 0.7         # 7折代谢优惠
```

---

## 改动清单

### Phase 1: 能量调制赫布 (优先级高)

```
A. core/eoe/batched_agents.py
   - 添加 self.energy_history: 每步能量变化
   - 添加 reward_hebbian_step() GPU批处理
   - 在 step() 中调用

B. core/eoe/brain_thermodynamics.py
   - 添加 reward_modulated_hebbian()
   - 添加 eligibility_trace_tensors

C. tests/test_hebbian_learning.py
   - 迷宫环境: 随机生成起点/终点
   - 测试: 有/无赫布学习的收敛速度对比
```

### Phase 2: 子图挖掘 (优先级中)

```
D. core/eoe/subgraph_miner.py
   - 实现 gSpan 算法 (简化版)
   - 拓扑编码 → 频繁模式挖掘

E. core/eoe/supernode_registry.py
   - SuperNode 动态注册
   - 代谢成本计算
   - 压缩/解压函数

F. core/eoe/batched_agents.py
   - 添加后台挖掘调度器
   - 每 N 步触发一次
```

---

## 测试计划

### Test 1: 赫布学习 Maze 收敛

```python
# 迷宫: 随机起点 → 随机终点
# 对比:
#   - Baseline: 无赫布，纯基因突变
#   - Ours:   有能量调制赫布

预期结果:
- Baseline: 500代后仍未收敛
- Ours:     50代学会走迷宫
```

### Test 2: SuperNode 涌现

```python
# 运行 5000 步
# 观察:
#   - 是否检测到重复拓扑
#   - SuperNode 数量
#   - 代谢成本节省

预期: 至少 1 个 SuperNode 从涌现中冻结
```

### Test 3: 端到端演化

```python
# 完整系统运行 100 代
# 指标:
#   - 存活率
#   - 平均节点数
#   - 平均能量
#   - SuperNode 数量
#   - 赫布边比例
```

---

## 风险与回退

| 风险 | 缓解 |
|------|------|
| 赫布学习不稳定 | 限制权重变化幅度 ±0.1/步 |
| 子图挖掘太慢 | 限制 Top 50 Agent，异步运行 |
| SuperNode 过度压缩 | 设置 max_supernodes=10 上限 |

**回退**: 设置 `hebbian_learning.enabled: false` 一键关闭

---

## 审批

- [ ] 方向二：能量调制赫布学习 ✓
- [ ] 方向三：子图挖掘冻结 ✓
- [ ] 改动清单确认
- [ ] 测试计划确认

**请审批后开始实现**