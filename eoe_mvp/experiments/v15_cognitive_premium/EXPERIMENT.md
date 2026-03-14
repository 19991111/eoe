# v15 认知溢价实验 (Cognitive Premium Experiment)

> 实验启动: 2026-03-14 | 核心问题: 打破"所见即所得"的马尔可夫陷阱

## 背景

v14成功实现了鲍德温效应+演化棘轮，但环境仍是MDP，Agent不需要记忆和推理。
本实验引入**认知溢价**机制，强制演化出真正的心智能力。

---

## Q1: 收益-成本不对称 & 能量获取机制

### 问题
当前环境是"所见即所得"——看到能量走过去就能吃到，不需要复杂推理。

### 方案

#### 1.1 异或问题 (XOR Problem)
- **红色食物**: 高能量 (+50)
- **红色毒素**: 扣能量 (-30)
- **区分方式**: 视觉(红色) + 嗅觉(特定气味)
- **实现**: 感知节点必须同时接收VISUAL和OLFACTORY输入，通过MULTIPLY或XOR节点判断
- **效果**: 打破2节点直连，强制3节点以上网络

#### 1.2 能量壳 (Energy Shells)
- **密码机制**: 能量站需要"解锁"
  1. 位置A触发机关（+1信号）
  2. 10步内到达位置B获取能量
  3. 超时或错误顺序则失败
- **效果**: 强制制造"认知溢价"——需要记忆+推理才能获取能量

---

## Q2: 代谢模型 (Metabolic Model)

### 问题
线性代谢 `Cost = N × Base` 对复杂基因惩罚过重，未成型功能就被饿死。

### 方案

#### 2.1 非线性代谢 (对数规模经济)
```
Cost = log(N + 1) × BaseCost
```
- 1节点: 0×Base = 0
- 5节点: ~1.79×Base
- 10节点: ~2.39×Base
- 100节点: ~4.62×Base

#### 2.2 免费基础脑容量
- 前5个PROCESSOR节点免代谢费
- 鼓励低压力下的结构探索

#### 2.3 稀疏激活 (Sparse Activation)
- 休眠节点不耗能
- 仅当信号真实流经节点并产生动作时计算代谢
- 基于发放率(spike rate)计算: `Cost = base × firing_rate × dt`

---

## Q3: 演化驱动力 - POMDP

### 问题
动态高斯斑块仍是MDP，当前状态包含所有信息，不需要记忆。

### 方案

#### 3.1 T型迷宫任务 (T-Maze)
- **出生时**: 短暂光信号(左/右)，持续5步
- **盲区**: 20步无任何信号
- **决策点**: 岔路口，根据记忆选择
- **奖励**: 正确方向+100，错误方向+0
- **效果**: 强制演化DELAY/循环连接

#### 3.2 资源周期性消失
- **冬季**: 能量场完全隐形
- **线索残留**: 温度/Stigmergy信号逐渐衰减
- **要求**: Agent演化内部状态机追踪轨迹
- **周期**: 500步一循环

---

## Q4: 涌现触发条件 - Red Queen Dynamics

### 问题
当前只有单向非生物压力，需要真正的捕食者-猎物协同演化。

#### 4.1 红皇后动力学
- **智能猎物**: 能量斑块会"逃跑"
  - 检测直线靠近的Agent
  - Z字形逃脱模式
  - 需要计算一阶/二阶导数(加速度)
- **效果**: 简单2节点预测失效，强制多节点网络

#### 4.2 形态学锁定 (Morphological Lock)
- **SuperNode成本**: 调用成本 = 1个单节点
- **打包条件**: 稳定工作超过100代的3节点子图
- **效果**: 极大刺激层级嵌套

---

## 实现清单

### 核心模块
- [ ] `core/eoe/environment/xor_environment.py` - XOR感知环境
- [ ] `core/eoe/energy/energy_shell.py` - 能量壳机制
- [x] `core/eoe/batched_agents.py` - 非线性代谢 ✅
- [x] `core/eoe/t_maze.py` - T型迷宫任务 ✅
- [x] `core/eoe/intelligent_prey.py` - 智能猎物 ✅

### 配置
- [x] `experiments/v15_cognitive_premium/configs/mechanisms_v15.yaml` ✅
- [ ] `experiments/v15_cognitive_premium/configs/environment_v15.yaml`

### 测试
- [x] `scripts/test_nonlinear_metabolism.py` ✅
- [x] `scripts/test_red_queen.py` ✅
- [ ] `scripts/test_xor_perception.py`
- [ ] `scripts/test_energy_shell.py`
- [ ] `scripts/run_v15_experiment.py`

---

## 预期效果

| 指标 | v14 | v15预期 |
|------|-----|---------|
| 平均脑容量 | ~5节点 | ~15-20节点 |
| 记忆能力 | 无 | T迷宫>80%正确 |
| 代谢效率 | 线性 | 对数(5x免费) |
| 涌现时间 | >500代 | >300代(Red Queen) |

---

## 实验分组

### 对照组: v14机制
- 线性代谢
- MDP能量获取
- 无XOR问题

### 实验组: v15机制
- 非线性代谢 + 免费脑容量
- POMDP + T迷宫
- XOR感知 + 能量壳
- 智能猎物