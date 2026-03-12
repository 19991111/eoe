# EOE v13.0 快速入门

## 安装

```bash
cd eoe_mvp
pip install -r requirements.txt
```

确保有 NVIDIA GPU 和 PyTorch CUDA 版本:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 快速开始

### 方式1: 命令行

```bash
# 运行GPU仿真
python main_v13_gpu.py --agents 1000 --steps 1500

# 保存结果
python main_v13_gpu.py --agents 500 --steps 500 --save results.json
```

### 方式2: Python API

```python
from core.eoe import Simulation, run

# 方式A: 直接运行
history = run(n_agents=100, steps=500)

# 方式B: 详细控制
sim = Simulation(
    n_agents=1000,
    lifespan=1500,
    env_size=100.0,
    device='cuda:0'
)
history = sim.run()

# 获取状态用于可视化
state = sim.get_state()
```

## 核心概念

### 统一场物理 (v13.0)

- **EPF** (Energy Field): 能量场 - Agent的能量来源
- **KIF** (Kinetic Impedance Field): 阻抗场 - 地形障碍
- **ISF** (Stigmergy Field): 压痕场 - Agent间通信

### 热力学

- 能量提取: EPF × 渗透率(κ)
- 代谢消耗: 基础 + 运动 + 信号²
- 信号注入: 向ISF场写入

### GPU加速

- 100% VRAM常驻
- 批量矩阵运算
- 毫秒级步进

## 性能

| 配置 | 每步耗时 | 吞吐量 |
|------|---------|--------|
| 100 agents | ~15ms | 6.7K/s |
| 500 agents | ~16ms | 31K/s |
| 1000 agents | ~17ms | 59K/s |

1500步 × 1000 agents ≈ 26秒 (vs CPU 525秒)

## 下一步

- [API参考](./API_REFERENCE.md)
- [架构设计](./ARCHITECTURE.md)
- [GPU优化](./gpu/OVERVIEW.md)