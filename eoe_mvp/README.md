# EOE (Evolving Organisms Engine) 🧬

> 大脑演化模拟引擎 - 探索数字生命的鲍德温效应与演化棘轮

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

EOE是一个GPU加速的演化模拟引擎，旨在研究数字生物的大脑如何演化出学习能力、模块化结构和代谢效率。

## ✨ 核心特性

### 🧠 神经架构演化
- **DAG大脑**: 有向无环图神经网络，节点类型包括 SENSOR, ADD, MULTIPLY, THRESHOLD, DELAY, ACTUATOR
- **寒武纪初始化**: 随机生成3-7节点的初始大脑结构
- **结构变异**: 节点/边的增删改 mutation

### 🔬 演化机制 (v14)

| 机制 | 描述 |
|------|------|
| **鲍德温效应** | Hebbian学习使种群"学会学习"，学习能力本身在演化 |
| **演化棘轮** | SuperNode模式冻结，节省代谢成本30% |
| **形态计算** | 节点的增删变异 |
| **个体发育** | 生长阶段控制 |
| **信息素场** | 集群交互与信息共享 |
| **红皇后** | 捕食者-猎物军备竞赛 |

### ⚡ GPU加速
- PyTorch张量批量处理
- 虚拟内存管理 (10,000+ Agent支持)
- 能量场、环境场GPU运算

### 🌡️ 环境系统
- **季节循环**: 冬季(0.1x) / 夏季(1.5x) 能量倍率
- **干旱事件**: 周期性资源枯竭
- **能量场**: 空间分布的食物源
- **危险场**: 捕食者区域

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/19991111/eoe.git
cd eoe

# 运行演化实验 (20,000步)
PYTHONPATH=. python scripts/run_v14_evolution.py 20000

# 运行测试
PYTHONPATH=. python scripts/test_evo_mechanisms.py
```

## 📊 实验结果 (v14)

```
运行: 20,000步 | 时间: 83.9秒 | 速度: 238步/秒

存活人数:     1,582 → 5,000 (达上限)
平均节点:     5.0 → 6.8
Hebbian活跃:  67 → 4,117 (+60倍) ⭐
SuperNode:    2 → 10 (模式冻结)
代谢节省:     4.35%
```

### 关键发现

**鲍德温效应**: Hebbian学习活跃度增长60倍，证明种群在演化"学习如何学习"的能力，而非仅仅演化固定行为。

**演化棘轮**: 10个常见网络模式被识别并冻结，每个模式节省30%代谢成本。

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
│       ├── subgraph_miner.py     # 频繁子图挖掘 ⭐
│       ├── supernode_registry.py # SuperNode管理 ⭐
│       └── brain_thermodynamics.py # 代谢成本计算 ⭐
├── scripts/
│   ├── run_v14_evolution.py      # 主实验脚本
│   └── test_*.py                 # 各类测试
├── config/
│   └── mechanisms.yaml           # 演化机制配置
└── docs/
    └── REPORT_v14_STAGE.md       # 详细实验报告
```

## 🔧 配置参数

### Hebbian学习
```python
HEBBIAN_ENABLED = True
HEBBIAN_BASE_LR = 0.01
HEBBIAN_TRACE_DECAY = 0.9
HEBBIAN_ELIGIBILITY_TRACE = 5
```

### SuperNode挖掘
```python
SUPERNODE_ENABLED = True
SUPERNODE_DETECTION_FREQUENCY = 500
SUPERNODE_MIN_OCCURRENCE = 5
SUPERNODE_METABOLIC_BONUS = 0.5  # 30%折扣
```

### 寒武纪初始化
```python
CAMBRIAN_INIT = True
CAMBRIAN_MIN_NODES = 3
CAMBRIAN_MAX_NODES = 7
CAMBRIAN_DELAY_PROB = 0.3
```

## 📈 发现的模式

实验发现的常见网络结构 (SuperNode):

| 模式 | 结构 | 支持度 |
|------|------|--------|
| Linear | SENSOR→ADD→ACTUATOR | 14 |
| Gated | SENSOR→THRESHOLD→ACTUATOR | 9 |
| Chain | SENSOR→ADD→THRESHOLD→ACTUATOR | 6 |

## 🔬 下一步方向

- [ ] 引入循环网络 (RNN) 支持记忆
- [ ] 增加环境压力促进复杂行为涌现
- [ ] 多智能体协作任务
- [ ] 空间记忆与导航演化

## 📄 许可证

MIT License

---

*Built with 🔥 by EOE Team*