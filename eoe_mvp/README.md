# EOE - Evolutionary Optimalism Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

## 项目简介

EOE (Evolutionary Optimalism Engine) 是一个演化智能体研究项目，目标是**仅通过设计环境压力**（不预设大脑结构），让智能体在生存竞争中自然涌现出"贮粮过冬"的高级行为。

### 核心原则

- **只设计环境压力，不设计大脑结构**
- 智能必须从持续的生存竞争中涌现
- 避免"适应度悬崖"，创造"适应度缓坡"

## 成果展示

| 版本 | 适应度 | 贮粮数 | 里程碑 |
|------|--------|--------|--------|
| v0.86 | 6,999 | 3 | 贮粮首次稳定涌现 |
| v0.93 | 13,029 | - | 终极奖励机制 |
| v0.97 Stage 2 | **40,008** | **198** | 🎉 贮粮行为彻底固化！ |

### 冠军大脑

- **节点数**: 17
- **边数**: 16
- **关键结构**: DELAY=4帧 (短期记忆) + 2个META节点 (涌现)

## 快速开始

### 环境要求

- Python 3.8+
- NumPy
- 建议: 4GB+ RAM, 支持GPU加速

### 安装

```bash
git clone https://github.com/19991111/eoe_mvp.git
cd eoe_mvp
pip install numpy
```

### 运行训练

```bash
cd eoe_mvp
python scripts/run_v097_stage2.py
```

### 查看冠军大脑

```bash
# 直接查看JSON
cat champions/best_v097_brain.json

# 或用浏览器打开HTML可视化
open brain_v097_viewer.html
```

## 项目结构

```
eoe_mvp/
├── core/
│   └── eoe/
│       ├── agent.py          # Agent类
│       ├── environment.py    # 环境类 (物理掉落机制)
│       ├── genome.py         # 脑基因组
│       ├── node.py           # 节点类型
│       └── population.py     # 演化种群
├── champions/                # 冠军大脑
│   └── best_v097_brain.json # v0.97最佳大脑
├── scripts/
│   ├── run_v097_stage2.py   # Stage 2训练
│   └── run_v097_stage3_winter.py  # 凛冬测试
├── brain_v097_viewer.html   # 大脑可视化
├── PROJECT_REVIEW.md        # 项目复盘文档
└── README.md
```

## 核心机制

### 1. 物理掉落 (Physical Drop)
Agent睡觉时携带的食物自动掉在脚边，计入贮粮。

### 2. 起床饥饿 (Wake-up Hunger)
醒来时一次性扣除能量，逼迫Agent"带着食物去睡觉"。

### 3. 疲劳系统 (Fatigue System)
疲劳影响移动速度（不强制死亡），静止时恢复。

## 论文/引用

如需引用本项目：

```bibtex
@software{eoe_mvp,
  title = {EOE - Evolutionary Optimalism Engine},
  author = {陆正旭},
  year = {2026},
  url = {https://github.com/19991111/eoe_mvp}
}
```

## 许可证

MIT License - See [LICENSE](LICENSE) for details.

## 联系方式

- 作者: 陆正旭 (南京大学人工智能学院)
- GitHub: [@19991111](https://github.com/19991111)