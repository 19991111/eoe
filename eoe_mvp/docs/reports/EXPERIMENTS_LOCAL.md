# EOE 实验项目文档

> 本文档详细记录 EOE (Evolving Organisms Explorer) 项目的所有实验流程和结果
> 
> **注意**: 本文档仅保存在本地，不上传到 GitHub
> 
> 最后更新: 2026-03-13

---

## 一、项目概述

### 1.1 核心使命

**EOE (Evolving Organisms Explorer)** - 通过自然选择涌现真正智能的演化智能体系统。

核心哲学: **"只设计环境压力，不设计大脑结构"**
- 智能必须从持续的生存竞争中涌现
- 预设节点类型是"先天结构"，加速自然选择
- 精英选择机制本身也是一种环境压力

### 1.2 当前版本

- **核心引擎**: v13.0 (GPU重构版)
- **物理法则**: 8个已注册
- **GPU加速**: 210x (PyTorch + VRAM常驻)
- **大脑结构**: 16x16 可演化神经网络

---

## 二、实验环境

### 2.1 硬件配置

```
设备: CUDA GPU (cuda:0)
环境大小: 100.0 x 100.0
分辨率: 1.0
```

### 2.2 软件依赖

```python
torch >= 2.0
numpy
matplotlib
```

### 2.3 项目结构

```
eoe_mvp/
├── core/
│   └── eoe/
│       ├── __init__.py          # 统一API导出
│       ├── manifest.py          # 物理法则配置
│       ├── integrated_simulation.py  # 集成仿真引擎
│       ├── batched_agents.py    # GPU批量Agent系统
│       ├── environment_gpu.py   # GPU环境
│       ├── thermodynamic_law.py # 热力学定律
│       └── brain_manager.py     # 大脑管理系统
├── scripts/
│   ├── main_v13_gpu.py          # 统一入口
│   ├── test_first_light_v13.py  # 物理基准测试
│   ├── test_neural_evolution_v13.py  # 神经演化实验
│   └── ...
├── champions/                    # 大脑库
│   └── hall_of_fame/            # 精英大脑
└── docs/
```

---

## 三、实验流程详解

### 3.1 实验流程概览

EOE 项目的实验遵循以下三步流程:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EOE 神经进化流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 神经接管                                                     │
│     ↓                                                            │
│     1000个异构大脑的GPU并行前向传播                               │
│     ↓                                                            │
│  2. 热力学淘汰                                                   │
│     ↓                                                            │
│     1500-2000步纯能量筛选                                        │
│     ↓                                                            │
│  3. 跨代遗传                                                     │
│     ↓                                                            │
│     精英交叉变异 + ISF生态印记                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 第一步: 神经接管 (Neural Takeover)

**目的**: 用神经网络控制Agent的行为，而非随机或预设规则

**实现方式**:
- 创建异构大脑 (HeterogeneousBrain)
- 输入: 7维传感器数据
- 输出: 5维致动器控制
- 网络结构: Input(7) -> Hidden(16) -> Output(5)

**传感器维度** (7维):
| 索引 | 传感器 | 描述 |
|------|--------|------|
| 0 | 能量感知 | 当前位置的能量密度 |
| 1 | 阻抗感知 | 当前位置的阻抗 |
| 2 | 压痕感知 | 当前位置的压痕强度 |
| 3-4 | 速度 | vx, vy |
| 5-6 | 位置 | x, y |

**致动器维度** (5维):
| 索引 | 致动器 | 描述 |
|------|--------|------|
| 0 | 推力方向 | 运动推力的角度 |
| 1 | 推力强度 | 推力大小 |
| 2 | 渗透调整 | 身体渗透率 |
| 3 | 防御调整 | 防御姿态 |
| 4 | 信号沉积 | 向ISF场释放信息素 |

### 3.3 第二步: 热力学淘汰 (Thermodynamic Culling)

**目的**: 通过能量守恒法则淘汰不适应的Agent

**热力学定律**:
- 能量守恒: EPF场能量 + Agent能量 = 常数
- 代谢消耗: 每个步进消耗能量
- 能量获取: 从EPF场提取能量
- 死亡条件: 能量 <= 0

**物理场**:
| 场 | 名称 | 作用 |
|----|------|------|
| EPF | 能量场 | 能量分布与流动 |
| KIF | 阻抗场 | 运动阻力 |
| ISF | 压痕场 | 空间记忆/信息素 |

### 3.4 第三步: 跨代遗传 (Cross-Generation Inheritance)

**目的**: 将优秀基因传递给下一代

**精英选择**:
- 保留比例: 10% (elite_ratio)
- 选择依据: 存活能量值
- 300个Agent中保留30个精英

**变异操作**:
- 变异率: 10% (mutation_rate)
- 变异强度: 0.5
- 高斯噪声注入

**交叉操作**:
- 精英间随机配对
- 权重矩阵随机交叉

**生态印记**:
- ISF场跨代衰减: 50% (isf_decay)
- 上一代的"记忆"部分传递给下一代

---

## 四、实验记录

### 4.1 实验一: First Light (物理基准测试)

**日期**: 2026-03-12
**脚本**: `scripts/test_first_light_v13.py`
**目的**: 验证宇宙物理法则的坚不可摧

**测试内容**:
1. 随机灵魂注入: 1000个Agent，随机致动器输入
2. 热力学守恒断言: 宇宙总能量 = 初始注入总能量
3. NaN/Inf检测: 所有张量完整性检查
4. 上帝之眼可视化: 每50步渲染宇宙快照

**运行命令**:
```bash
python scripts/test_first_light_v13.py --steps 500 --agents 1000
```

**结果**:
- ✅ 热力学守恒验证通过
- ✅ 无NaN/Inf异常
- ✅ 可视化快照保存成功 (first_light_snapshots/)

**可视化输出**:
- `snapshot_00050.png`: 第50步宇宙状态
- `snapshot_00100.png`: 第100步宇宙状态

---

### 4.2 实验二: 神经演化 20代 (Neural Evolution 20Gen)

**日期**: 2026-03-12
**脚本**: `scripts/test_neural_evolution_v13.py`
**目的**: 验证神经网络能否从随机权重演化出生存策略

**运行参数**:
```python
n_agents = 300          # Agent数量
n_steps = 300           # 每代步数
n_generations = 20      # 进化代数
device = 'cuda:0'       # 计算设备
elite_ratio = 0.1       # 精英保留比例
mutation_rate = 0.1     # 变异率
isf_decay = 0.5         # ISF场衰减率
```

**运行命令**:
```bash
python scripts/test_neural_evolution_v13.py \
    --agents 300 \
    --steps 300 \
    --generations 20
```

**结果数据**:
```
Generation History:
--------------------------------------------------
 Gen |  Alive |     Mean E |      Max E |      Top E
--------------------------------------------------
   1 |    231 |     130.80 |     199.76 |     199.76
   2 |    264 |     136.19 |     199.98 |     199.98
   3 |    280 |     140.57 |     199.82 |     199.82
   4 |    273 |     140.04 |     199.95 |     199.95
   5 |    275 |     141.22 |     199.84 |     199.84
   6 |    287 |     145.39 |     149.70 |     149.70
   7 |    295 |     148.26 |     149.70 |     149.70
   8 |    295 |     149.01 |     149.70 |     149.70
   9 |    300 |     149.49 |     149.70 |     149.70
  10 |    300 |     149.58 |     149.70 |     149.70
  11 |     0 |       0.00 |       0.00 |       0.00
```

**观察分析**:
- 第1代: 231/300 存活 (23%死亡率)
- 第5代: 275/300 存活
- 第9代: 300/300 存活 (100%存活!)
- 第6代后: 最大能量收敛到 ~149.70 (能量天花板)

**结论**:
- ✅ 神经网络成功演化出生存策略
- ✅ 存活率从77%提升到100%
- ⚠️ 第11代出现异常 (0 存活) - 待调查

**输出文件**:
- `evolution_20gen.log`: 完整日志
- `evolution_history.png`: 演化曲线图

---

### 4.3 实验三: 机制配置测试

**日期**: 2026-03-13
**脚本**: 内置测试
**目的**: 验证 manifest.py 机制配置系统

**测试预设**:
| 预设 | 用途 | 测试结果 |
|------|------|----------|
| no_signal | 禁用ISF场 | ✅ 通过 |
| no_evolution | 禁用进化 | ✅ 通过 |

**验证方式**:
```python
from core.eoe.manifest import PhysicsManifest

# 加载预设
manifest = PhysicsManifest.from_yaml("no_signal")
# ISF 场正确禁用

manifest = PhysicsManifest.from_yaml("no_evolution")
# 进化机制正确禁用
```

---

### 4.4 实验四: Bug修复验证

**日期**: 2026-03-13
**问题**: 神经网络前向传播未正确应用 brain_masks

**Bug描述**:
- `torch.matmul` 不支持批量掩码矩阵乘法
- `brain_masks` 从未真正应用到权重

**修复方案**:
```python
# 修复前
hidden = torch.matmul(sensors, W1)  # 忽略掩码

# 修复后
W1_masked = W1 * M1  # 应用连接掩码
hidden = torch.bmm(sensors.unsqueeze(1), W1_masked).squeeze(1)
```

**验证**: 修复后神经网络正确应用连接掩码

---

## 五、关键代码模块

### 5.1 HeterogeneousBrain (异构大脑)

```python
class HeterogeneousBrain:
    """异构大脑前向传播器"""
    
    def forward(
        self, 
        sensors: torch.Tensor, 
        brain_weights: torch.Tensor,
        brain_masks: torch.Tensor,
        node_types: torch.Tensor
    ) -> torch.Tensor:
        """
        批量前向传播
        
        网络结构: Input(7) -> Hidden(16) -> Output(5)
        
        Args:
            sensors: [N, 7] 传感器输入
            brain_weights: [N, 16, 16] 大脑权重矩阵
            brain_masks: [N, 16, 16] 连接掩码 (30%连接率)
            node_types: [N, 16] 节点类型
            
        Returns:
            Tensor [N, 5] 致动器输出
        """
        # 提取权重并应用掩码
        W1 = brain_weights[:, :input_dim, :hidden_dim]
        M1 = brain_masks[:, :input_dim, :hidden_dim]
        W1_masked = W1 * M1
        
        # 隐藏层
        hidden = torch.bmm(sensors.unsqueeze(1), W1_masked).squeeze(1)
        hidden = F.relu(hidden)
        
        # 输出层
        output = torch.bmm(hidden.unsqueeze(1), W2_masked).squeeze(1)
        
        return output
```

### 5.2 进化操作

```python
def mutate(brain_weights, mutation_rate=0.1, mutation_strength=0.5):
    """变异操作"""
    mutation_mask = torch.rand_like(brain_weights) < mutation_rate
    noise = torch.randn_like(brain_weights) * mutation_strength
    brain_weights = brain_weights + mutation_mask.float() * noise
    return torch.clamp(brain_weights, -3, 3)

def crossover(parent1_weights, parent2_weights):
    """精英交叉"""
    crossover_mask = torch.rand_like(parent1_weights) > 0.5
    return torch.where(crossover_mask, parent1_weights, parent2_weights)
```

### 5.3 统一入口

```python
from core.eoe import quick_run

# 快速运行
history = quick_run(n_agents=100, steps=500, device='cuda:0')

# 或使用主入口
python main_v13_gpu.py --agents 500 --steps 1500
```

---

## 六、待办实验

### 6.1 计划中的实验

- [ ] 神经演化 30代 (更长时间尺度)
- [ ] 神经演化 50代+ (极限演化)
- [ ] 异构网络拓扑实验 (可变隐藏层)
- [ ] 多环境压力测试 (能量分布不均匀)
- [ ] 信号场 (ISF) 演化实验
- [ ] 大脑可视化与分析

### 6.2 已发现问题

- [ ] 第11代异常 (0存活) - 需要调试
- [ ] 能量天花板问题 (~149.70) - 需要增加能量注入机制
- [ ] 文档中的旧版本注释需要清理

---

## 七、实验数据

### 7.1 日志文件

| 文件 | 描述 |
|------|------|
| `evolution_20gen.log` | 20代演化完整日志 |
| `evolution_30gen.log` | 30代演化日志 (空) |
| `evolution_opt.log` | 优化实验日志 |

### 7.2 可视化输出

| 文件 | 描述 |
|------|------|
| `evolution_history.png` | 20代演化曲线图 |
| `first_light_snapshots/snapshot_00050.png` | 第50步快照 |
| `first_light_snapshots/snapshot_00100.png` | 第100步快照 |

### 7.3 大脑库

| 目录 | 描述 |
|------|------|
| `champions/hall_of_fame/by_fitness/` | 按适应度保存 |
| `champions/hall_of_fame/by_stage/` | 按阶段保存 |

---

## 八、快速启动

### 8.1 运行神经演化实验

```bash
cd eoe_mvp

# 20代演化
python scripts/test_neural_evolution_v13.py \
    --agents 300 \
    --steps 300 \
    --generations 20

# 10代快速测试
python scripts/test_neural_evolution_v13.py \
    --agents 100 \
    --steps 200 \
    --generations 10
```

### 8.2 运行基准测试

```bash
# First Light 物理验证
python scripts/test_first_light_v13.py \
    --steps 500 \
    --agents 1000

# 性能分析
python scripts/profile_v13_performance.py
```

### 8.3 快速仿真

```bash
# 命令行
python main_v13_gpu.py --agents 500 --steps 1500

# Python代码
from core.eoe import quick_run
history = quick_run(n_agents=100, steps=500)
```

---

## 九、附录

### 9.1 配置预设

| 预设 | 说明 |
|------|------|
| full | 全部启用 |
| simple | 基础功能 |
| no_signal | 禁用信号场 |
| infinite_energy | 无限能量 |
| no_evolution | 禁用进化 |
| wrap_world | 边界环绕 |

### 9.2 物理常数

```python
# 热力学参数
METABOLISM_BASE = 0.5       # 基础代谢消耗
ENERGY_EXTRACT_RATE = 0.8   # 能量提取率
MAX_VELOCITY = 5.0          # 最大速度
INIT_ENERGY = 150.0         # 初始能量

# 神经网络参数
HIDDEN_DIM = 16             # 隐藏层维度
CONNECTION_RATE = 0.3       # 连接率 (30%)
MUTATION_RATE = 0.1         # 变异率
ELITE_RATIO = 0.1           # 精英比例
```

### 9.3 Git 提交记录

| Commit | 描述 |
|--------|------|
| d33ad84 | fix: 修复神经网络前向传播bug |
| d052cbb | feat: 统一机制配置到 manifest.py |

---

*本文档为本地实验记录，包含项目全部实验流程和细节。*