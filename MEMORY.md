

---

## 2026-03-17 上午: 配置拆分完成 ✅

### 配置管理系统 (configs/)
```
configs/
├── __init__.py           # 模块入口
├── base.py               # 基础配置 (PoolConfig, AgentState)
├── presets/
│   ├── __init__.py
│   ├── stable.py         # 稳定版 (StableConfig, MinimalConfig, HighPressureConfig)
│   └── experiment.py     # 实验版 (ExperimentConfig, V17Phase2Config等)
└── utils.py              # 工具函数 (validate_config, merge_config等)
```

### 使用方式
```python
# 方式1: 基础配置
from configs import PoolConfig
config = PoolConfig()

# 方式2: 稳定版预设
from configs.presets.stable import StableConfig
config = StableConfig()

# 方式3: 实验版预设
from configs.presets.experiment import V17Phase2Config
config = V17Phase2Config()

# 验证配置
from configs.utils import validate_config
warnings = validate_config(config)
```

### 向后兼容
- 旧导入路径 `from core.eoe.batched_agents import PoolConfig` 仍然可用
- 新代码建议使用 `from configs import PoolConfig`

---

## 2026-03-17 上午: 繁殖节点添加修复 + 能量系统调试

### 繁殖时节点添加修复 (batched_agents.py)
**问题**: 繁殖时只修改了 `node_counts` 计数器，没有真正复制父代基因组

**修复**:
1. 繁殖时复制父代基因组到子代 (`genome.copy()`)
2. 根据 `add_node_prob=0.30` 概率调用 `mutate_add_node()`
3. 繁殖后自动调用 `set_brains()` 更新脑结构
4. 修复 `age_multiplier` 类型错误 (当 AGE_ENABLED=False 时)

### 能量系统调试发现
**问题**: Agent 能量持续下降，无法繁殖

**根因**:
1. 能量源每10步才注入一次，需预热 `for _ in range(10): env.step()`
2. `NONLINEAR_METABOLISM=True` (默认) 会应用 sigmoid 成本曲线，导致额外消耗

**解决方案**:
```python
class StableConfig(PoolConfig):
    BASE_METABOLISM = 0.005
    NONLINEAR_METABOLISM = False  # 禁用
    AGE_ENABLED = False
    # 先预热环境10步
    for _ in range(10):
        env.step()
```

---

## 2026-03-17 凌晨-早上: v17.2 Phase 1 + Phase 2 实施 ✅

### Phase 1: Net-2-Net 零权重死亡之谷修复 (已验证)
见上一条记录

### Phase 2: 软承载力 + 拥挤度惩罚 (已实施)

**新增配置参数**:
```python
# 软承载力
SOFT_CARRYING_CAP = True
GLOBAL_ENERGY_BUDGET = 8000.0

# 拥挤度惩罚
CROWDING_PENALTY_ENABLED = True
CROWDING_RADIUS = 12.0
CROWDING_DECAY_EXPONENT = 0.6
CROWDING_MIN_FACTOR = 0.2
```

**新增方法**:
- `_compute_crowding_penalty()`: 基于局部密度的能量获取衰减
- `_apply_soft_carrying_capacity()`: 基于全局预算的人口调节

**验证结果**:
- 500步测试: 40/40 存活 (100%)
- 拥挤惩罚生效: 邻居多的Agent摄食量减少至60%

---

## 2026-03-17 凌晨: v17.1 阶段一修复 - 打破零权重死亡之谷 ✅

### 三个核心修复

**1. 代谢适应期 (Metabolic Grace Period)** - 已有实现
- 配置: `METABOLIC_GRACE = True`, `STEPS = 100`, `DISCOUNT = 0.5`
- 新拓扑变异的子代在前100步只付50%代谢成本

**2. 带噪恒等初始化 (Noisy Identity Init)** - 新增实现
- 配置: `NOISY_IDENTITY_INIT = True`, `NOISY_IDENTITY_SIGMA = 0.1`
- 在Net-2-Net零初始化基础上注入N(0, 0.1)噪声，打破对称性

**3. MODULATOR预偏置 (Pre-bias)** - 新增实现
- 配置: `MODULATOR_BIAS = 2.0`
- 新增Node属性: `modulator_bias = 2.0`
- sigmoid(2.0) ≈ 0.88，默认"开启"状态，平滑过渡旧网络

### 修改文件
- `core/eoe/batched_agents.py`: 新增3个配置参数
- `core/eoe/genome.py`: mutate_add_node添加带噪初始化，MODULATOR计算添加偏置
- `core/eoe/node.py`: Node.__init__添加modulator_bias属性

---

## 2026-03-16 下午 (续): v16.27 数值溢出修复 ✅

### 🔥 问题确认
从v16.26实验结果观察到:
- Step 4000: 总能量 188,422 (正常)
- Step 4200+: **1.84×10²²** (暴增10¹⁴倍!)

### 🔧 根因分析
`batched_agents.py` 能量分配逻辑:
```python
# 修复前 (溢出)
ratio = actual_consumed / safe_req  # 可能 >> 1.0
actual_feed = feed_amount * premium_invisible(20x) * ratio  # 爆炸!
```

### ✅ 修复方案 (v16.27)
```python
# 1. ratio上界约束
ratio = (actual_consumed / safe_req).clamp(max=1.0)

# 2. 硬性上界
ENERGY_UPPER_BOUND = 1e6
actual_feed = actual_feed.clamp(max=ENERGY_UPPER_BOUND)

# 3. NaN/Inf检测
if torch.isinf(actual_feed).any():
    actual_feed = torch.nan_to_num(actual_feed, nan=0.0, posinf=0.0)
```

### 📊 测试结果
| 测试场景 | 步数 | max_energy | 结果 |
|----------|------|------------|------|
| 基础测试 | 500 | 2,697 | ✅ |
| 中等压力 | 2000 | 4,495 | ✅ |
| 高压力(1500种群) | 2000 | 134,849 | ✅ |

### Git
- Commit: cb5d527 - v16.27: 修复数值溢出

---

## 2026-03-16 下午: 能量守恒Bug修复 + 生态震荡涌现!

### 🔥 能量泄漏Bug (核心问题)

**位置**: `batched_agents.py` 第1371行 `_apply_sensing`

**问题**: 
- 可见能量从环境正确扣除
- 隐身能量(20x倍数)没有从环境扣除
- 导致能量无限增殖 (1.06×10²³ 溢出!)

**修复**: 从能量场扣除invisible能量的等效消耗

### 🎯 生态震荡出现! (Lotka-Volterra)

v16.26 实验结果 (8000步):
| Step | 种群 | 总能量 |
|------|------|--------|
| 600 | 1500 | 103,426 (峰值) |
| 1000 | 486 | 15,980 (崩溃) |
| 1400 | 1500 | 135,530 (再达峰值) |
| 2200 | 412 | 12,405 (再次崩溃) |

→ 种群在 1500↔400↔1500↔400 间震荡!

### 📊 历史最佳结构

- struct_35936: **32.78分**, 9节点/17边 (v16.26)
  - 历史最高复杂度!

### 待优化
1. 移除 max_agents 硬编码上限 (当前1500)
2. 降低 source_capacity 让震荡更自然
3. 添加能量审计断言

---

## 2026-03-16 里程碑: SuperNode模块化进化成功!

### 核心发现: "展开节点数"揭示真相

| 版本 | 表层节点 | 含SuperNode | 适应度 |
|------|----------|-------------|--------|
| v16.21 (6000步) | 11.84 | 0% | ~50 |
| v16.22 (12000步) | 10.0 | 20% | **112.0** |

### 结构对比
- struct_24165: 4表层节点 + 1 SuperNode → 适应度112.0 (王者!)
- 表面变简洁，但深层逻辑封装成模块

### 关键洞察
1. SuperNode模块化 = "高效的微服务架构"
2. 表层节点降8%，但适应度升124%！
3. 捕食压力迫使系统优化为"小而精"而非"大而杂"

### 下一步
- 增加护盾/刺突执行器 (物理维度)
- 启用Stigmergy Field (信息素维度)

---

## 2026-03-21: 场域T-Maze Benchmark

### EoE Native 设计原则
用Agent的"母语"(物理场)设计考题:
- KIF (阻抗场) = 墙壁 - Agent碰到会减血，本能避开
- EPF (能量场) = 目标 - Agent本能追逐
- ISF (痕迹场) = 记忆

### 实现
- `_add_field_based_t_maze()` in benchmark_runner.py
- KIF墙: 在x=50设置高阻抗(1000)
- EPF目标: EnergySource(90,25), radius=25
- 启用ISF用于stigmergy

### 调试发现
1. **KIF墙有效**: Agent遇高阻抗真的停止
2. **EPF需预热**: env.step()后场max=199
3. **问题**: Agent不追逐能量 - brain在演化时没学过追逐EPF

### 根因
Brain在开放能量场演化，只学会"捡最近的能量"
T-Maze需要"记住目标位置并导航" - 不同技能

### 下一步选项
1. 重新演化能追逐EPF梯度的brain
2. 在T-Maze目标处放置食物(直接信号)
3. 使用更复杂brain基准测试

---

## 📊 当前状态 (2026-03-21 12:11)

### 已完成
- ✅ 场域T-Maze基础设施 (KIF墙+EPF目标+ISF)
- ✅ Benchmark框架 + Warmup
- ✅ CONSTANT偏置修复

### 待解决
- ❌ Brain不追逐EPF (0/3成功)
- 需要目标导向的brain重新测试
