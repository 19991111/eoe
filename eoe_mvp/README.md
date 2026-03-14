# EOE (Evolving Organisms Engine) 🧬

> 大脑演化模拟引擎 - 探索数字生命的认知涌现与智能演化

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

EOE是一个GPU加速的演化模拟引擎，旨在研究数字生物的大脑如何演化出学习能力、记忆、推理和复杂智能结构。

## ✨ 核心特性 (v15)

### 🧠 神经架构演化
- **DAG大脑**: 有向无环图神经网络，节点类型包括 SENSOR, ADD, MULTIPLY, THRESHOLD, DELAY, ACTUATOR
- **寒武纪初始化**: 随机生成3-7节点的初始大脑结构
- **预加载脑结构**: 从保存的复杂结构初始化种群 ⭐ NEW

### 🧬 演化机制 (v15)

| 机制 | 描述 | 状态 |
|------|------|------|
| **鲍德温效应** | Hebbian学习使种群"学会学习" | ✅ |
| **演化棘轮** | SuperNode模式冻结，节省代谢成本 | ✅ |
| **预加载脑结构** | 从复杂结构初始化，跳过随机探索 | ✅ NEW |
| **非线性代谢** | 对数规模经济，前5节点免费 | ✅ |
| **T型迷宫** | POMDP任务，强制记忆与推理 | ✅ |
| **红皇后** | 智能猎物，提供持续选择压力 | ✅ |
| **能量循环** | 60%代谢能量回归环境 | ✅ |

### ⚡ GPU加速
- PyTorch张量批量处理
- 虚拟内存管理 (10,000+ Agent支持)
- 能量场、环境场GPU运算
- ~280步/秒

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/19991111/eoe.git
cd eoe

# 运行预加载实验 (推荐)
PYTHONPATH=. python scripts/run_v15_pretrained.py --structures experiments/v15_cognitive_premium/saved_structures/complexity_step30000.json --steps 10000

# 运行标准实验
PYTHONPATH=. python scripts/run_v15_experiment.py --steps 10000
```

## 📊 实验结果 (v15.2 预加载)

```
运行: 10,000步 | 速度: ~280步/秒

最终人口:     308 (稳定存活)
复杂结构:     23种
总快照数:     5,944
最高复杂度:   16.89分
```

### 涌现的高级结构

| 结构 | 复杂度 | 节点 | 特征 |
|------|--------|------|------|
| struct_66957 | 16.89 | 5节点/6边 | 反馈+乘法 |
| struct_58092 | 12.08 | 4节点/5边 | 反馈连接 |
| struct_37797 | 12.08 | 4节点/5边 | 负权重 |

**关键发现:**
- 17%结构有反馈连接 → 记忆能力涌现
- DELAY节点普遍存在 → 内部状态保持
- MULTIPLY节点提升复杂度 → 非线性计算

## 🏗️ 项目结构

```
eoe_mvp/
├── core/
│   └── eoe/
│       ├── batched_agents.py     # 核心Agent池 (GPU加速)
│       ├── environment_gpu.py    # GPU环境模拟
│       ├── genome.py             # 基因组 (DAG大脑)
│       ├── node.py               # 节点类型定义
│       ├── manifest.py           # 物理法则注册
│       ├── complexity_tracker.py # 复杂结构追踪 ⭐
│       ├── subgraph_miner.py     # 频繁子图挖掘
│       └── t_maze.py             # T型迷宫任务
├── scripts/
│   ├── run_v15_experiment.py     # v15主实验
│   ├── run_v15_pretrained.py     # 预加载实验
│   └── load_brain_structures.py  # 结构加载器
├── experiments/
│   ├── v15_cognitive_premium/    # v15实验数据
│   └── v15_pretrained/           # 预加载实验数据
└── docs/
    └── REPORT_v15.md             # 实验报告
```

## 🔧 配置参数

### 预加载脑结构 (v15.2)
```python
PRETRAINED_INIT = True
PRETRAINED_STRUCTURES_FILE = "path/to/structures.json"
PRETRAINED_TOP_N = 20
```

### 非线性代谢
```python
NONLINEAR_METABOLISM = True
LOG_BASE = 2.0
FREE_NODES = 5  # 前5节点免费
SPARSE_ACTIVATION = True
```

### T型迷宫 (POMDP)
```python
T_MAZE_ENABLED = True
T_MAZE_SIGNAL_DURATION = 5
T_MAZE_BLIND_ZONE = 20
```

## 📈 演化进程

| 版本 | 关键突破 |
|------|----------|
| v14 | 鲍德温效应 + 演化棘轮 |
| v15 | 非线性代谢 + T迷宫 + 智能猎物 |
| v15.2 | **预加载脑结构** - 复杂智能涌现 |

## 📄 许可证

MIT License

---

*Built with 🔥 by EOE Team*