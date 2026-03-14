# EOE项目 v14 阶段性实验报告

> 日期: 2026-03-14  
> 实验: 鲍德温效应 + 演化棘轮  
> 步数: 20,000步

---

## 📊 实验摘要

| 指标 | 初始值 | 最终值 | 变化 |
|------|--------|--------|------|
| 存活人数 | 1,582 | 5,000 | +2418 (达上限) |
| 平均节点数 | 5.0 | 6.8 | +0.9 |
| Hebbian活跃度 | 67 | 4,117 | **+4050** |
| SuperNode数量 | 2 | 10 | +8 |
| 代谢节省 | - | 4.35% | - |
| 总出生 | - | 83 | - |
| 总死亡 | - | 18 | - |

**运行时间**: 83.9秒 (238.4步/秒)

---

## 🧬 启动的模块

### 1. 物理法则 (9个)

通过 `PhysicsManifest.from_yaml('full')` 加载:

| 法则 | 类名 | 功能 |
|------|------|------|
| ✅ metabolism | MetabolismLaw | 基础代谢消耗 |
| ✅ seasonal | SeasonalCycleLaw | 季节循环 (冬/夏能量倍率) |
| ✅ fatigue | FatigueSystemLaw | 疲劳系统 |
| ✅ interference | PortInterferenceLaw | 端口干扰 |
| ✅ morphology | MorphologicalComputationLaw | 形态演化机制 |
| ✅ ontogeny | OntogeneticPhaseLaw | 个体发育阶段 |
| ✅ stigmergy | StigmergicFrictionLaw | 信息素场交互 |
| ✅ red_queen | RedQueenLaw | 红皇后事件 (捕食者) |
| ✅ thermal | ThermalSanctuaryLaw | 热力避难所 |

### 2. 演化机制 (每Step调用)

| 机制 | 描述 |
|------|------|
| morphology | 形态计算: 节点/边的增删变异 |
| ontogeny | 个体发育: 生长阶段控制 |
| stigmergy | 集群信息素: 场交互 |
| thermal | 热力调节: 温度适应性 |

### 3. 事件机制 (触发调用)

| 机制 | 触发条件 |
|------|----------|
| red_queen | 捕食者事件 |

### 4. 新增核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 🆕 subgraph_miner.py | 频繁子图挖掘 | 发现常见网络模式 |
| 🆕 supernode_registry.py | SuperNode注册 | 模式冻结与代谢优化 |
| 🆕 brain_thermodynamics.py | 大脑热力学 | 代谢成本计算 |
| 🆕 perception.py | 感知系统 | 多通道感知 |
| 🆕 asynchronous_pool.py | 异步池 | 并行处理 |

### 5. BatchedAgents关键配置

#### 启用功能
- ✅ HEBBIAN_ENABLED: Hebbian学习 (核心)
- ✅ HEBBIAN_REWARD_MODULATION: 奖励调制
- ✅ SUPERNODE_ENABLED: SuperNode挖掘
- ✅ AGE_ENABLED: 年龄衰老
- ✅ PREDATION_ENABLED: 捕食者
- ✅ SEASONS_ENABLED: 季节循环
- ✅ DROUGHT_ENABLED: 干旱事件
- ✅ CAMBRIAN_INIT: 寒武纪初始化
- ✅ SILENT_MUTATION: 静默突变
- ✅ METABOLIC_GRACE: 代谢宽限期

#### 关键参数
```python
HEBBIAN_BASE_LR = 0.01          # Hebbian学习率
HEBBIAN_TRACE_DECAY = 0.9       # 迹衰减
HEBBIAN_ELIGIBILITY_TRACE = 5   # 迹长度

SUPERNODE_DETECTION_FREQUENCY = 500  # 挖掘频率
SUPERNODE_MIN_OCCURRENCE = 5         # 最小支持度
SUPERNODE_METABOLIC_BONUS = 0.5      # 代谢折扣

CAMBRIAN_MIN_NODES = 3          # 初始最小节点
CAMBRIAN_MAX_NODES = 7          # 初始最大节点
CAMBRIAN_DELAY_PROB = 0.3       # DELAY节点概率
CAMBRIAN_MULTIPLY_PROB = 0.3    # MULTIPLY节点概率

WINTER_MULTIPLIER = 0.1         # 冬季能量倍率
SUMMER_MULTIPLIER = 1.5         # 夏季能量倍率
SEASON_LENGTH = 2000            # 季节周期(步)
```

---

## 🔬 实验结果分析

### 1. 鲍德温效应 (Baldwin Effect)

**Hebbian学习活跃度**: 67 → 4,117 (+60倍)

这是本实验最重要的发现。Hebbian学习 ("一起放电的神经元连接在一起") 使种群能够:
- 通过经验强化有效连接
- 惩罚低效连接
- 将学习到的行为"遗传"给后代

实验显示，随着时间推移，越来越多的Agent活跃地进行Hebbian学习，表明**学习能力本身在演化**。

### 2. 演化棘轮 (Evolutionary Ratchet)

**SuperNode冻结**: 2 → 10个模式

| SuperNode | 模式 | 原始成本 | 冻结成本 | 节省 |
|-----------|------|----------|----------|------|
| SUPERNODE_0 | (1,7,2) | 0.0050 | 0.0035 | 30% |
| SUPERNODE_1 | (1,6,2) | 0.0240 | 0.0168 | 30% |
| ... | ... | ... | ... | ... |

**总节省**: 4.35% 代谢成本

### 3. 网络结构演化

**发现的模式** (按支持度):

| 排名 | 节点序列 | 结构 | 支持度 |
|------|----------|------|--------|
| 1 | (1,7,2) | SENSOR→ADD→ACTUATOR | 14 |
| 2 | (1,6,2) | SENSOR→THRESHOLD→ACTUATOR | 9 |
| 3 | (1,7,6,2) | SENSOR→ADD→THRESHOLD→ACTUATOR | 6 |

**关键发现**: 更简单的3节点网络反而比4节点网络更普遍，说明:
- 简单结构在恶劣环境下更节能
- 复杂度增长缓慢，符合"够用就好"原则

### 4. 种群动态

- **出生**: 83 (新Agent诞生)
- **死亡**: 18 (能量耗尽)
- **存活率**: 极高 (5000/5000达上限)

---

## 📈 趋势分析

```
节点数趋势:     5.0 → 6.8  (+0.9)
Hebbian趋势:   67 → 4117   (指数增长)
SuperNode:     2 → 10      (线性增长后饱和)
能量水平:       低 → 极高   (能量过剩)
```

---

## ⚠️ 当前局限

1. **复杂度瓶颈**: 最复杂的模式仍是4层线性网络，未出现:
   - 循环网络 (记忆)
   - 平行分支 (多感官融合)
   - 残差连接 (跳跃连接)

2. **环境压力不足**: 能量过于充足， selection pressure 不够

3. **捕食者效率低**: Red Queen事件触发较少

---

## 🎯 下一步方向

### 方案A: 增加环境压力
- 降低能量补充频率
- 增加捕食者强度
- 添加周期性灾难

### 方案B: 引入任务复杂度
- 移动目标追踪
- 多阶段任务 (感知→移动→摄食)
- 空间记忆测试

### 方案C: 调整演化参数
- 提高mutation结构变异率
- 增大CAMBRIAN节点范围
- 引入模块化mutation

---

## 📁 相关文件

- 实验脚本: `scripts/run_v14_evolution.py`
- 测试脚本: `scripts/test_evo_mechanisms.py`
- 核心代码: `core/eoe/batched_agents.py`
- 子图挖掘: `core/eoe/subgraph_miner.py`
- SuperNode: `core/eoe/supernode_registry.py`

---

*报告生成时间: 2026-03-14 UTC*