# MEMORY - 长期记忆

> EOE项目核心记忆 | 最后更新: 2026-03-15

---

## 项目状态

- **项目目录**: `eoe_mvp/`
- **核心引擎**: v16.7 (欺骗性景观实验)
- **当前挑战**: 打破"伏地魔"策略 (Agent静止不动)

---

## v15.2 预加载脑结构机制 ✅

**注册配置 (PoolConfig):**
- `PRETRAINED_INIT`: 启用开关
- `PRETRAINED_STRUCTURES_FILE`: 结构文件路径
- `PRETRAINED_TOP_N`: 使用Top N最复杂结构

**实验结果 (10000步):**
- 种群稳定存活: 308人
- 复杂结构: 23种
- 最高复杂度: 16.89分 (5节点/6边+反馈+乘法)
- 涌现特征: 反馈连接(17%)、DELAY节点(记忆)、MULTIPLY节点(非线性)

**Bug修复:** set_brains()现填充genomes字典供复杂度追踪器使用

---

## 快速命令

```bash
cd eoe_mvp

# 运行实验
python scripts/run_v15_experiment.py

# 预加载实验
python -c "from core.eoe.batched_agents import PoolConfig; \
  config = PoolConfig(); \
  config.PRETRAINED_INIT = True; \
  config.PRETRAINED_STRUCTURES_FILE = '...'"
```

---

## v16.0 构成性环境升级 (2026-03-14)

### 方案文档
- `docs/upgrade_plans/COMPOSITIONALITY_ENVIRONMENT_PLAN.md` (922行)

### 实施进度

**Phase 1: MatterGrid 基础设施** ✅ 已完成
- [x] `environment.py` - 添加 matter_grid, matter_energy
- [x] `environment_gpu.py` - GPU 版本支持
- [x] 辅助方法: is_solid, add_matter, remove_matter, get_matter_energy
- [x] 能量存储机制 (Review #3) ✅
- [x] 碰撞检测集成 (`batched_agents.py` _apply_matter_collision) ✅
- [x] 压痕场双重掩码 (Review #1) ✅
- [x] 测试通过: `test_matter_grid.py`, `test_stigmergy_occlusion.py`

**Phase 1 剩余任务**:
- [x] 硬编码U墙测试智能体饿死 (隐含在碰撞检测中)
- [x] Phase 2: 建造/分解执行器节点 (T2)
- [x] Phase 3: 挡风墙生态位演化 (T6) - 核心实现完成

### Code Review 修复
| # | 问题 | 状态 |
|---|------|------|
| 1 | 压痕场穿墙 | ✅ 已实现 (双重掩码) |
| 2 | 自我活埋 | ✅ forward_dist > radius |
| 3 | 能量守恒 | ✅ 已实现 |
| 4 | GPU并发 | ✅ 已实现 (去重逻辑) |

---

## 2026-03-15 欺骗性景观实验 (v16.1-v16.7)

### 核心问题
如何在高运动惩罚下逼出"动态预测"智能体？

### 实验进展

| 版本 | 策略 | 结果 |
|------|------|------|
| v16.2 | MOVEMENT_PENALTY=0.05 | Agent完全不动 (Vel=0) |
| v16.3 | COGNITIVE_PREMIUM=10x | 伏地魔+Hebbian 20-35% |
| v16.4 | BMR=0.12 | 仍未打破伏地魔 |
| v16.5 | ACTIVE_SENSING | 复杂度首次下降10% |
| v16.6 | 可见1x vs 隐身10x | Hebbian爆发71-80% |
| v16.7 | 极端不对称0.1x vs 20x | 仍在静止 |

### 关键发现
- **Hebbian达到极限**: 80%同时学习，ΔE=30
- **伏地魔策略极其顽固**: 能量源自动移动，静止也能"捡漏"
- **演化算法极其理性**: 只要不动能活，就绝不动

### 待解决
- 限制能量源运动范围
- 或增加初始能量消耗迫使其移动

---

## 待办

- [ ] 形式化研究问题 (即将进行)

---

## 2026-03-15 v16.1 欺骗性景观实验模块

基于专家5大建议创建的实验模块：`experiments/v16_deceptive_landscape/`

### 文件结构
```
v16_deceptive_landscape/
├── test_invisible_energy.py    # 能量隐身基准测试
├── test_sparse_metabolism.py   # 稀疏代谢测试
├── run_experiment.py           # 主实验脚本
└── benchmark_results.png       # 测试结果图表
```

### 核心设计（已去目的论化）

**1. 欺骗性景观**
- 能量斑块周期性隐身（50步可见，150步隐身）
- 隐身时保持惯性运动（非布朗）
- 隐身距离=120，感知范围=15（8倍封堵乌龟策略）

**2. 稀疏代谢**
- 基础代谢: 0.005
- 激活阈值: 0.1
- 克莱伯指数: 0.75（N^0.75次线性缩放）
- 次线性优势: 50节点时达60%

**3. 可塑性死锁解决方案**
- 方案A: Hebbian学习使用pre-activation而非post-activation
- 方案B: 自发噪声（需调整参数）

### 基准测试结果
- 陷阱1验证: ✓ 隐身距离8倍于感知范围
- 克莱伯验证: ✓ 单调递增
- 记忆Agent对比: 需调整（当前环境斑块静止导致区分度不够）

### 待调整
- 基准测试需让斑块持续移动，才能体现记忆优势

---

## 2026-03-14 数学形式化文档

创建 `docs/MATHEMATICAL_FORMALIZATION.md`:

### 6个模块
1. **智能体与脑结构** - Agent状态、脑图G=(V,E,W)
2. **多通道物理场** - 能量场、痕迹场、阻抗场、压力场 (PDE方程)
3. **大脑热力学** - 代谢函数 M(A_i)、能量积分方程
4. **具身动力学** - 感知卷积、运动更新方程
5. **演化与鲍德温** - 繁殖、拓扑变异、Hebbian学习
6. **复杂性度量** - 拓扑熵、超节点检测

### Python架构映射
每个模块对应核心类的属性和更新方法

---

## 2026-03-14 终极目标形式化

创建 `docs/EOE Ultimate Theorem.md`:

### 核心定理
$$\exists \Theta^* \quad \text{s.t.} \quad \lim_{t \to \infty} \mathbb{E}[C(G_t)] \to \infty$$

### 三大约束
1. **微观约束**: 个体存活并繁衍需满足能量积分
2. **宏观约束**: 系统总能量守恒
3. **环境约束**: 环境被种群行为反向塑造

### 充要条件
$$\frac{\partial I_{gain}}{\partial C} > \frac{\partial M}{\partial C}$$

边缘认知收益 > 边缘热力学成本

---

## 2026-03-14 升级方案

创建 `docs/upgrade_plans/`:

- **DIFFERENTIABLE_EVOLUTION_PLAN.md**: 可微演化架构升级方案
  - Phase 1: PyTorch可微计算图
  - Phase 2: 生命周期梯度注入 (Truncated BPTT)
  - Phase 3: 深度鲍德温遗传同化

### 2026-03-14 可微演化实现 ✅

**新增文件:**
- `core/eoe/differentiable_brain.py` - 可微大脑 (PyG MessagePassing)
- `core/eoe/lifecycle_optimizer.py` - 生命周期优化器

**修改文件:**
- `core/eoe/genome.py` - 添加基因型/表型记录 + 拓扑保护同化
- `core/eoe/batched_agents.py` - 新增配置项

**配置项:**
- `DIFFERENTIABLE_BRAIN`: 启用可微大脑
- `BALDWIN_ASSIMILATION_KAPPA`: 同化率 0.5

---

## 2026-03-14 GitHub更新

已推送 v15.2:
- 56 files changed, 66,452 insertions
- 预加载机制 + 复杂结构追踪器 + T型迷宫 + 智能猎物
- 实验数据 (saved_structures)