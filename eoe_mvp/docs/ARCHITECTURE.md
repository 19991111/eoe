# 架构设计文档

## 概述

EOE v13.0 是一个基于统一场物理的GPU加速演化引擎。

```
┌─────────────────────────────────────────────────────────────┐
│                      Simulation (入口)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────────┐ ┌───────────┐ ┌─────────────────┐
│   Environment   │ │ Batched   │ │ Thermodynamic   │
│    (GPU)        │ │  Agents   │ │      Law        │
└────────┬────────┘ └─────┬─────┘ └────────┬────────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │         │      │         │      │         │
    ▼         ▼      ▼         ▼      ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│  EPF  │ │  KIF  │ │  ISF  │ │ State │ │ Fields│
│ Field │ │ Field │ │ Field │ │Tensor │ │  Base │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

## 模块设计

### 1. fields/ - 物理场

```
fields/
├── __init__.py       # 导出
├── base.py          # Field抽象基类
├── energy.py        # EPF能量场
├── impedance.py     # KIF阻抗场
└── stigmergy.py     # ISF压痕场
```

**设计原则:**
- 继承 `Field` 基类
- 实现 `step()`, `sample()`, `compute_gradient()`
- 支持GPU张量操作

### 2. batch/ - 批量GPU系统

```
batch/
├── __init__.py       # 导出
├── state.py          # AgentState容器
├── thermo.py         # 热力学定律
└── simulation.py     # 统一入口
```

**设计原则:**
- 100% VRAM常驻
- 批量操作替代循环
- 状态张量化

### 3. 环境层

**EnvironmentGPU:**
- 管理多个场
- 批量传感器采样
- 预计算梯度

**Environment (CPU):**
- 向后兼容
- NumPy实现

## 数据流

```
Step N:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   获取传感器   │ ──▶ │  神经网络前向  │ ──▶ │  物理更新    │
│  (批量采样)   │     │  (矩阵乘法)   │     │  (速度/位置)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                           ┌───────────────────────┘
                           ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  热力学交互   │ ──▶ │   场更新     │
                    │ (能量交换)   │     │ (扩散/衰减)  │
                    └──────────────┘     └──────────────┘
```

## 物理模型

### 能量提取
```
E_extract = EPF(x,y) × κ × rate
```
- EPF: 位置的能量场值
- κ: Agent渗透率 (0-1)
- rate: 提取率 (0.5)

### 代谢消耗
```
E_cost = base + |velocity| × 0.01 + signal² × 0.01
```

### 信号注入
```
ISF(x,y) += signal_strength
```

## 性能优化

### 1. VRAM常驻
- 所有张量创建时指定device
- 避免CPU-GPU数据传输

### 2. 批量操作
- `get_value_tensor()` 批量查表
- 矩阵乘法替代循环

### 3. 预计算梯度
- 每步计算一次梯度矩阵
- Agent查表O(1)

## 扩展性

### 添加新场

```python
from core.eoe.fields import Field

class NewField(Field):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化
        
    def step(self):
        # 更新逻辑
        pass
    
    def sample(self, x, y):
        # 采样逻辑
        return value
    
    def compute_gradient(self):
        return grad_x, grad_y
```

### 添加新物理法则

```python
from core.eoe.batch import ThermodynamicLaw

class CustomLaw(ThermodynamicLaw):
    def apply(self, env, agents, alive_mask):
        # 自定义能量交互
        pass
```