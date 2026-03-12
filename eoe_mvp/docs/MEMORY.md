# MEMORY - 项目记忆管理

> EOE演化智能体项目核心记忆库 | 最后更新: 2026-03-12

---

## 一、项目概述

### 1.1 核心使命
**EOE (Evolving Organisms Explorer)** - 通过自然选择涌现真正智能的演化智能体系统。

### 1.2 核心哲学
> **"只设计环境压力，不设计大脑结构"**
> - 智能必须从持续的生存竞争中涌现
> - 预设节点类型是"先天结构"，加速自然选择
> - 精英选择机制本身也是一种环境压力

### 1.3 研究方向
- 大语言模型 (LLM)
- 通用人工智能 (AGI)
- 演化计算

---

## 二、版本体系

### 2.1 版本号规范
```
格式: MAJOR.MINOR.PATCH
- MAJOR: 架构重大变更 (如 v12 重构)
- MINOR: 新机制/新功能 (如 v11.1 压力熔炉)
- PATCH: Bug修复/参数调整
```

### 2.2 当前版本

| 组件 | 版本 | 说明 |
|------|------|------|
| **核心引擎** | v13.0 | EnergyField物理系统 + GPU批量系统 |
| **物理法则** | v13.0 | 能量场+渗透膜+热力学交换 |
| **GPU加速** | v13.0 | PyTorch张量 + VRAM常驻 |
| **大脑管理** | v1.0 | BrainManager |
| **实验脚本** | v13.0 | main_v13_gpu.py |

### 2.3 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v13.0 | 2026-03-12 | **GPU重构** - fields模块 + batch模块 + 统一API (210x加速) |
| v13.0 | 2026-03-12 | **能量场物理系统** - 连续标量场E(x,y)替代离散食物、扩散方程、能量源、渗透膜κ、热力学交换法则 |
| v12.6 | 2026-03-12 | 全部8个PhysicalLaw迁移完成 |
| v12.5 | 2026-03-12 | MechanismRegistry注册系统 |
| v12.0 | 2026-03-12 | PhysicsManifest基础架构 |
| v11.1 | 2026-03-12 | 压力梯度熔炉 |
| v11.0 | 2026-03-12 | 能量衰减+端口干涉+季节波动 |
| v10.4 | 2026-03-12 | 破解智力税黑洞 |
| v0.99 | 2026-03-11 | 阶段三重构 |
| v0.93 | 2026-03-10 | 终极奖励机制 |

---

## 三、架构体系

### 3.1 核心模块 (v13.0 GPU架构)

```
core/eoe/
├── __init__.py              # 统一API导出
├── version.py               # 版本信息
│
├── fields/                  # [NEW] 物理场模块
│   ├── __init__.py
│   ├── base.py             # Field抽象基类
│   ├── energy.py           # EPF能量场
│   ├── impedance.py        # KIF阻抗场
│   └── stigmergy.py        # ISF压痕场
│
├── batch/                   # [NEW] 批量GPU系统
│   ├── __init__.py
│   ├── state.py            # AgentState容器
│   ├── thermo.py           # 热力学定律
│   └── simulation.py       # 统一仿真入口
│
├── agent.py                 # 智能体定义 (CPU兼容)
├── genome.py                # 基因组与演化
├── node.py                  # 神经网络节点
├── population.py            # 种群管理
├── brain_manager.py         # 大脑管理系统
├── manifest.py              # 物理法则配置
├── environment.py           # CPU环境 (向后兼容)
├── environment_gpu.py       # GPU环境 (旧版)
└── _archive/                # 已归档代码
    └── environment_v11.py   # 旧版环境
```

### 3.2 新API使用

```python
# 方式1: 一行启动 (推荐)
from core.eoe import run
history = run(n_agents=1000, steps=1500)

# 方式2: 详细控制
from core.eoe import Simulation
sim = Simulation(n_agents=1000, device='cuda:0')
history = sim.run()

# 方式3: 命令行
python main_v13_gpu.py --agents 1000 --steps 1500
```

### 3.2 物理法则 (8个已注册)

| 法 则 | 类名 | 类型 | 状态 |
|-------|------|------|------|
| 代谢 | MetabolismLaw | 高频核心 | ✅ |
| 季节 | SeasonalCycleLaw | 周期性 | ✅ |
| 疲劳 | FatigueSystemLaw | 每步可选 | ✅ |
| 热力 | ThermalSanctuaryLaw | 周期性 | ✅ |
| 端口 | PortInterferenceLaw | 每步可选 | ✅ |
| 红皇后 | RedQueenLaw | 核心机制 | ✅ |
| 形态 | MorphologicalComputationLaw | P2 | ✅ |
| 压痕 | StigmergicFrictionLaw | P1 | ✅ |
| 发育 | OntogeneticPhaseLaw | P0 | ✅ |

### 3.3 配置管理 (SSOT)

**唯一真值**: `core/eoe/manifest.py`

所有参数必须从这里获取，包括:
- 代谢参数 (alpha, beta)
- 季节参数 (长度, 冬天乘数)
- 物理法则开关
- 实验参数

---

## 四、管理细则

### 4.1 文档管理

#### 分类体系

| 分类 | 前缀 | 用途 |
|------|------|------|
| 项目核心 | PROJECT_* | 项目定义/追踪 |
| 架构设计 | MIGRATION_*/ARCHITECTURE_ | 系统架构 |
| 方案文档 | *_MANAGEMENT | 具体方案 |
| 进度记录 | PROGRESS_* | 阶段进展 |
| 用户文档 | README | 面向用户 |

#### 核心文档清单

| 文档 | 状态 | 说明 |
|------|------|------|
| QUICKSTART.md | 活跃 | 快速入门 (v13.0) |
| API_REFERENCE.md | 活跃 | API参考 (v13.0) |
| ARCHITECTURE.md | 活跃 | 架构设计 (v13.0) |
| REFACTORING_PLAN.md | 活跃 | 重构方案 |
| VERSION_MANAGEMENT.md | 活跃 | 版本规范与历史 |
| MEMORY.md | 活跃 | 本文件 - 项目记忆 |

### 4.2 脚本管理

#### 活跃脚本 (v13.0)
```
scripts/
├── main_v13_gpu.py                   # 统一入口 (v13.0 GPU)
├── profile_v13_performance.py        # 性能探针
├── profile_v13_simple.py             # 简化探针
├── integrated_simulation.py          # 热力学集成
├── demiurge_render.py                # 可视化渲染
└── run_stage4_v111_crucible_test.py  # 旧版脚本 (保留)
```

#### 归档脚本
```
scripts/_archive/
├── stage1/    (1个)
├── stage2/    (4个)
├── stage3/    (1个)
├── debug/     (10个)
└── utility/   (2个)
```

### 4.3 大脑管理

#### 目录结构
```
champions/
├── _index.json           # 全局索引 (自动更新)
├── stage4/
│   ├── v110/             # v110实验
│   └── v111/             # v111实验
└── hall_of_fame/         # 英雄冢
```

#### BrainManager API
```python
from core.eoe.brain_manager import BrainManager

mgr = BrainManager()

# 保存
mgr.save_champion(brain, stage=4, experiment="v111", generation=20, fitness=694.7)

# 加载
champion = mgr.load_latest(stage=4)

# 查询
top10 = mgr.get_top_brains(n=10)

# 管理
mgr.prune_old_runs(keep_per_stage=3)
```

### 4.4 记忆管理

#### 文件结构
```
memory/
├── MEMORY.md              # 长期记忆 (核心原则)
└── YYYY-MM-DD.md          # 每日记录
```

#### 写入规则
- **PROJECT_TRACKER.md**: 每次版本迭代完成
- **EOE_PROJECT_STATE.md**: 每次实验结束/重大发现
- **MIGRATION_GUIDE_***: 每次架构重构
- **memory/YYYY-MM-DD.md**: 每次重要操作
- **MEMORY.md**: 每次重大里程碑/原则变更

#### 每日记忆内容
- 实验结果
- 代码变更
- 决策记录
- 待办事项

---

## 五、目录结构

```
eoe_mvp/
├── core/eoe/              # 核心代码 (8个文件)
│   └── _archive/          # 已归档
├── champions/             # 大脑库 (BrainManager)
├── scripts/               # 活跃脚本 (4个)
├── _archive/              # 归档
│   ├── config/            # 旧配置
│   ├── dryrun/            # 历史调试
│   ├── stage1-3/          # 旧阶段脚本
│   └── debug/             # 调试脚本
├── _experimental/         # 实验功能
│   └── gpu_accel/         # GPU加速 (未启用)
├── memory/                # 项目记忆
├── *.md                   # 文档 (7个)
└── physics_config.json    # deprecated
```

---

## 六、硬件与环境

### 6.1 硬件
- **GPU**: 4x NVIDIA A100 80GB
- **PyTorch**: 2.9.1+cu128
- **CuPy**: 13.6.0

### 6.2 GPU性能 (v13.0)

| 配置 | 每步耗时 | 吞吐量 | 加速比 |
|------|---------|--------|--------|
| CPU (原版) | ~350ms | 2.8K/s | 1x |
| 100 agents GPU | ~15ms | 6.7K/s | 23x |
| 500 agents GPU | ~16ms | 31K/s | 22x |
| 1000 agents GPU | ~17ms | 59K/s | 21x |
| 2000 agents GPU | ~18ms | 115K/s | 19x |

**1500步完整代**: ~26秒 (vs CPU 525秒 = **20x加速**)

### 6.3 关键技术

- **100% VRAM常驻**: 无CPU-GPU数据传输
- **批量张量操作**: 矩阵乘法替代循环
- **预计算梯度**: O(1)查表
- **F.conv2d**: ISF场扩散 (389x加速)

### 6.2 关键文件路径
- 最佳大脑: `champions/stage4_v111_r3.json` (125节点)
- 最高适应度: 40008.4 (v0.97 Stage 3)

---

## 七、待办事项

- [x] v13.0 GPU重构 (fields + batch模块)
- [x] 统一API (Simulation类)
- [x] 文档更新 (QUICKSTART, API, ARCHITECTURE)
- [ ] 清理旧environment.py代码
- [ ] 阶段五: LLM Demiurge

---

## 八、重要原则

1. **配置唯一真值**: manifest.py 是唯一配置源
2. **物理法则注册**: 所有机制通过 MechanismRegistry 管理
3. **大脑版本化**: 通过 BrainManager 统一管理
4. **文档版本化**: 迁移指南使用版本号
5. **记忆自动化**: 每次重要操作更新 memory/

---

## 记忆分层说明

| 层级 | 文件 | 用途 |
|------|------|------|
| **系统级** | /MEMORY.md | OpenClaw快速索引 |
| **项目级** | eoe_mvp/MEMORY.md | 核心项目记忆 |
| **每日级** | memory/YYYY-MM-DD.md | 工作日志 |

*本文件是项目的核心记忆库，任何重大变更都需要更新本文件。*