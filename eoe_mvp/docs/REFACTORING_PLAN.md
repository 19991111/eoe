# v13.0 代码重构方案

## 当前状态分析

### 问题诊断

1. **文件职责混乱**
   - `environment.py` (4154行): 包含CPU版环境 + 旧传感器逻辑
   - `environment_gpu.py` (520行): GPU版环境
   - 两个文件功能重叠但接口不统一

2. **模块边界不清**
   - `thermodynamic_law.py`: 独立文件，但在GPU版本中内联
   - `stigmergy_field.py`, `kinetic_impedance.py`: 分散的场实现
   - CPU和GPU实现分裂

3. **缺少顶层入口**
   - 没有统一的 `main_v13_gpu.py`
   - 用户需要自己组装组件

4. **文档分散**
   - `docs/` 目录结构简单
   - 缺少API文档

---

## 重构方案

### 1. 目录结构重构

```
core/eoe/
├── __init__.py           # 公共API导出
├── version.py            # 版本信息
│
├── # === 核心实体 (Entities) ===
├── agent.py              # Agent类 (CPU版本，向后兼容)
├── genome.py             # 基因组/大脑
├── node.py               # 节点类型定义
│
├── # === 物理场 (Fields) - 统一接口 ===
├── fields/
│   ├── __init__.py
│   ├── base.py           # Field基类
│   ├── energy.py         # EPF (能量场)
│   ├── impedance.py      # KIF (阻抗场)
│   ├── stigmergy.py      # ISF (压痕场)
│   └── stress.py         # ESF (应力场)
│
├── # === 环境 (Environment) ===
├── env_cpu.py            # CPU版环境 (原environment.py精简)
└── env_gpu.py            # GPU版环境 (原environment_gpu.py)
│
├── # === 批量系统 (Batched - GPU专享) ===
├── batch/
│   ├── __init__.py
│   ├── state.py          # AgentState张量容器
│   ├── agents.py         # BatchedAgents
│   ├── thermo.py         # ThermodynamicLaw
│   └── simulation.py     # IntegratedSimulation
│
├── # === 演化引擎 (Evolution) ===
├── population.py         # 种群管理
├── brain_manager.py      # 大脑档案管理
└── manifest.py           # 物理法则配置
│
└── _archive/             # 归档 (保留但不导入)
    ├── environment_v11.py
    └── ...
```

### 2. 模块职责定义

| 模块 | 职责 | 公共API |
|------|------|---------|
| `fields/base.py` | 场抽象基类 | `class Field` |
| `fields/energy.py` | EPF实现 | `class EnergyField` |
| `env_cpu.py` | CPU仿真 | `class Environment` |
| `env_gpu.py` | GPU仿真 | `class EnvironmentGPU` |
| `batch/agents.py` | 批量Agent | `class BatchedAgents` |
| `batch/simulation.py` | 集成仿真 | `class Simulation` |

### 3. 统一API设计

```python
# 目标: 用户使用简单
from core.eoe import Simulation, FieldConfig

# CPU版本
sim = Simulation(n_agents=100, device='cpu')
sim.run(1500)

# GPU版本  
sim = Simulation(n_agents=1000, device='cuda:0')
sim.run(1500)
```

### 4. 迁移路径

```
Phase A: 创建目录结构和基础文件
  - 创建 fields/ 目录
  - 创建 batch/ 目录
  - 移动/重命名文件

Phase B: 统一接口
  - 定义 Field 抽象基类
  - 统一 Environment/EnvironmentGPU 接口
  - 更新 batch/ 模块使用新接口

Phase C: 清理归档
  - 移除 _archive 中不再需要的文件
  - 更新 __init__.py 导出

Phase D: 文档更新
  - 重写 docs/ 目录结构
  - 生成 API 文档
```

---

## 实施计划

### 优先级 P0 (必须)

1. 创建 `fields/` 子目录，移动场实现
2. 创建 `batch/` 子目录，移动批量系统
3. 创建统一入口 `main_v13_gpu.py`
4. 更新 `__init__.py`

### 优先级 P1 (重要)

5. 定义 `Field` 抽象基类
6. 统一环境接口
7. 文档全面更新

### 优先级 P2 (优化)

8. 清理 `_archive/`
9. 添加类型注解
10. 性能微调

---

## 预期收益

- 代码行数减少 30% (消除重复)
- 新用户学习成本降低 50%
- 维护性大幅提升
- GPU/CPU切换零成本

---

## 待用户确认后执行