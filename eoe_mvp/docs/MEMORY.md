# MEMORY - 项目记忆管理

> EOE演化智能体项目核心记忆库 | 最后更新: 2026-03-13

---

## 一、项目概述

### 1.1 核心使命
**EOE (Evolving Organisms Explorer)** - 通过自然选择涌现真正智能的演化智能体系统。

### 1.2 核心哲学
> **"只设计环境压力，不设计大脑结构"**
> - 智能必须从持续的生存竞争中涌现
> - 预设节点类型是"先天结构"，加速自然选择
> - 精英选择机制本身也是一种环境压力

---

## 二、当前架构

### 2.1 核心模块

```
core/eoe/
├── __init__.py              # 统一API导出
├── manifest.py              # 物理法则配置 (唯一真值)
├── integrated_simulation.py # 集成仿真引擎
├── batched_agents.py        # GPU批量Agent系统
├── environment_gpu.py       # GPU环境
├── thermodynamic_law.py     # 热力学定律
└── brain_manager.py         # 大脑管理系统
```

### 2.2 配置系统

**唯一真值**: `core/eoe/manifest.py`

```python
from core.eoe.manifest import PhysicsManifest

# 加载预设配置
manifest = PhysicsManifest.from_yaml("full")

# 或使用默认配置
manifest = PhysicsManifest()
```

### 2.3 机制预设

| 预设 | 说明 |
|------|------|
| full | 全部启用 |
| simple | 基础功能 |
| no_signal | 禁用信号场 |
| infinite_energy | 无限能量 |
| no_evolution | 禁用进化 |
| wrap_world | 边界环绕 |

---

## 三、核心概念

### 3.1 物理场 (GPU)

| 场 | 作用 |
|---|------|
| EPF | 能量场 - 能量分布与流动 |
| KIF | 阻抗场 - 运动阻力 |
| ISF | 压痕场 - 空间记忆 |

### 3.2 Agent 系统

- **感知**: 6维传感器 (能量/阻抗/压痕/速度/位置/边界)
- **运动**: 推力/渗透/防御
- **信号**: 沉积/接收
- **能量**: 代谢/提取/死亡

### 3.3 大脑结构

- **拓扑**: 16x16 可演化神经网络
- **掩码**: 30% 连接率 (brain_masks)
- **前向**: torch.bmm 批量矩阵乘法

---

## 四、管理细则

### 4.1 文档管理

| 文档 | 用途 |
|------|------|
| VERSION_MANAGEMENT.md | 版本规范 |
| MEMORY.md | 本文件 - 项目记忆 |
| ARCHITECTURE.md | 架构设计 |
| QUICKSTART.md | 快速入门 |

### 4.2 脚本管理

```
scripts/
├── main_v13_gpu.py         # 统一入口
├── test_neural_evolution_v13.py  # 演化实验
└── integrated_simulation.py # 仿真引擎
```

### 4.3 记忆管理

```
memory/
├── MEMORY.md              # 长期记忆
└── YYYY-MM-DD.md          # 每日记录
```

---

## 五、重要原则

1. **配置唯一真值**: manifest.py 是唯一配置源
2. **物理法则注册**: 所有机制通过 MechanismRegistry 管理
3. **大脑版本化**: 通过 BrainManager 统一管理
4. **文档版本化**: 迁移指南使用版本号
5. **记忆自动化**: 每次重要操作更新 memory/

---

## 六、待办事项

- [ ] 版本更新时同步文档
- [ ] 保持文档与代码同步

---

*本文件是项目的核心记忆库，任何重大变更都需要更新本文件。*