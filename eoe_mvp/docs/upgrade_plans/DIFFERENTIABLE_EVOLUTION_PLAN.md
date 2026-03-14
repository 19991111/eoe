# 可微演化 (Differentiable Evolution) 架构升级方案

> 基于 EOE Project Lead 指令 | 版本: 1.0 | 日期: 2026-03-14

---

## 一、当前状态分析

### 现有机制
| 模块 | 当前实现 | 问题 |
|------|----------|------|
| Brain前向传播 | 图遍历手动计算 | 不可微，无法反向传播 |
| 权重学习 | 简化Hebbian (标量更新) | 非梯度方法，收敛慢 |
| 遗传同化 | 无 | 子代继承随机突变，无经验传递 |
| 拓扑变异 | NEAT标准实现 | 纯随机，无梯度引导 |

### 差距分析
- ❌ 无自动微分能力
- ❌ 无生命周期内梯度优化
- ❌ 无深度鲍德温效应
- ❌ 收敛效率低 (需大量代际)

---

## 二、修改方案

### Phase 1: 基础张量化 (PyTorch Autograd)

#### 任务 1.1: 创建 `DifferentiableBrain` 类

```python
# core/eoe/differentiable_brain.py (新文件)

class DifferentiableBrain(torch.nn.Module):
    """
    可微计算图大脑
    - 将NEAT图拓扑编译为 torch.nn.Module
    - 保留计算图，支持 .backward()
    """
    
    def __init__(self, genome: OperatorGenome):
        super().__init__()
        self._build_from_genome(genome)
    
    def _build_from_genome(self, genome: OperatorGenome):
        """
        将图拓扑转为稀疏权重张量
        """
        # 为每个节点创建可学习参数
        # 使用 sparse tensor 优化
        pass
    
    def forward(self, sensor_input: torch.Tensor) -> torch.Tensor:
        """
        前向传播，保留计算图
        """
        pass
    
    def compute_loss(self, energy_delta: torch.Tensor, 
                     predicted_next_state: torch.Tensor = None) -> torch.Tensor:
        """
        计算局部损失:
        L = -energy_gain + lambda * prediction_error
        """
        pass
```

#### 任务 1.2: 修改 PoolConfig

```python
# 在 batched_agents.py 的 PoolConfig 中添加:

# 可微演化配置
DIFFERENTIABLE_BRAIN = False        # 启用可微大脑
DIFFERENTIABLE_LR = 0.001           # 生命周期学习率
DIFFERENTIABLE_UPDATE_INTERVAL = 10 # 每N步更新一次权重
PREDICTION_LOSS_WEIGHT = 0.1        # 预测损失权重

# 深度鲍德温配置
BALDWIN_ASSIMILATION_KAPPA = 0.5    # 同化率 (0-1)
BALDWIN_EXPLORATION_SIGMA = 0.01    # 变异噪声
```

---

### Phase 2: 生命周期梯度注入

#### 任务 2.1: 实现 LifecycleOptimizer

```python
# 在 batched_agents.py 中添加

class LifecycleOptimizer:
    """
    生命周期优化器
    - 为每个Agent维护独立的optimizer状态
    - 使用截断反向传播 (Truncated BPTT)
    """
    
    def __init__(self, agents: 'BatchedAgents', config: PoolConfig):
        self.agents = agents
        self.config = config
        self.step_buffer = []  # 存储 (state, action, reward) 元组
        
        # 为每个可能存活的Agent创建optimizer
        self.optimizers = {}  # {agent_id: torch.optim.Optimizer}
    
    def step(self, batch: ActiveBatch, energy_deltas: torch.Tensor):
        """
        在每个Agent的生命周期内执行梯度更新
        """
        # 1. 收集经验到buffer
        # 2. 每DIFFERENTIABLE_UPDATE_INTERVAL步执行一次反向传播
        # 3. 使用 vmap 批量计算梯度
        pass
    
    def _compute_local_loss(self, agent_idx: int, energy_delta: float) -> torch.Tensor:
        """
        L_local = -energy_gain + lambda * prediction_error
        """
        pass
```

#### 任务 2.2: 修改 BatchedAgents.step() 集成可微学习

```python
# 在 batched_agents.py 的 step() 方法中添加:

def step(self, env, dt, brain_fn):
    # ... 现有逻辑 ...
    
    # [新增] 生命周期梯度更新
    if self.config.DIFFERENTIABLE_BRAIN:
        self.lifecycle_optimizer.step(batch, energy_deltas)
    
    # ... 现有逻辑 ...
```

---

### Phase 3: 深度鲍德温遗传同化

#### 任务 3.1: 扩展 Genome 类记录基因型/表型

```python
# 在 genome.py 中修改 OperatorGenome 类

class OperatorGenome:
    def __init__(self, ...):
        # ... 现有 ...
        
        # [新增] 深度鲍德温支持
        self.genotype_weights = {}  # 出生时的初始权重
        self.phenotype_weights = {} # 死亡/繁殖时的最终权重
    
    def finalize_phenotype(self):
        """
        在死亡/繁殖前保存表型权重
        """
        for edge in self.edges:
            edge_id = self.get_edge_id(edge['source_id'], edge['target_id'])
            self.phenotype_weights[edge_id] = edge['weight']
    
    def apply_baldwin_assimilation(self, parent_genome, kappa: float, sigma: float):
        """
        深度鲍德温遗传同化:
        W_child = W_parent + kappa * (W_phenotype - W_genotype) + N(0, sigma)
        """
        pass
```

#### 任务 3.2: 修改繁殖逻辑

```python
# 在 batched_agents.py 的 _reproduce() 中修改

def _reproduce(self, parent_idx: int, child_idx: int):
    parent_genome = self.genomes[parent_idx]
    
    if self.config.DIFFERENTIABLE_BRAIN and self.config.BALDWIN_ASSIMILATION_KAPPA > 0:
        # 应用深度鲍德温遗传同化
        child_genome = parent_genome.copy()
        child_genome.apply_baldwin_assimilation(
            parent_genome,
            kappa=self.config.BALDWIN_ASSIMILATION_KAPPA,
            sigma=self.config.BALDWIN_EXPLORATION_SIGMA
        )
    else:
        # 传统NEAT变异
        child_genome = self._neat_mutate(parent_genome)
```

---

## 三、实施顺序

| 阶段 | 任务 | 预期时间 | 验证指标 |
|------|------|----------|----------|
| Phase 1.1 | DifferentiableBrain基础结构 | 2小时 | 可执行forward+backward |
| Phase 1.2 | PoolConfig新增配置 | 30分钟 | 配置可读 |
| Phase 2.1 | LifecycleOptimizer | 3小时 | 内存不爆 |
| Phase 2.2 | 集成到step() | 1小时 | 权重在生命周期内变化 |
| Phase 3.1 | 基因型/表型记录 | 1小时 | 可保存最终权重 |
| Phase 3.2 | 遗传同化逻辑 | 2小时 | 子代继承父代表型 |

---

## 四、技术难点与解决方案

### 难点1: 批量梯度计算显存爆炸
**方案**: 使用 `torch.func.vmap` 或自定义 batched backward
```python
# 错误示范 (显存爆炸)
for agent in agents:
    loss.backward()  # N次显存分配

# 正确示范
losses = torch.stack([a.loss for a in agents])
torch.autograd.backward(losses, gradient=...)  # 一次反向传播
```

### 难点2: 变长图结构的张量编译
**方案**: Padding + Mask 或 动态计算图
```python
# 方案A: 固定最大节点数，填充0
# 方案B: 使用 torch Geometric 的 MessagePassing
```

### 难点3: 预测编码损失函数
**方案**: 简化实现 - 仅使用能量损失
```python
L_local = -energy_delta  # 最大化净能量获取
# 预测损失作为可选扩展
```

---

## 五、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `core/eoe/differentiable_brain.py` | 新增 | 可微大脑类 |
| `core/eoe/genome.py` | 修改 | 添加genotype/phenotype |
| `core/eoe/batched_agents.py` | 修改 | 添加LifecycleOptimizer |
| `core/eoe/lifecycle_optimizer.py` | 新增 | 生命周期优化器 |
| `scripts/test_differentiable.py` | 新增 | 单元测试 |
| `scripts/benchmark_differentiable.py` | 新增 | 性能测试 |

---

## 六、预期效果

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| 达到相同性能的代数 | 1000代 | 50代 | **20x** |
| 收敛速度 | 慢 | 快 | 待测 |
| 复杂结构涌现时间 | 5000步 | 1000步 | **5x** |

---

*方案待审阅后实施*
*EOE Project - Differentiable Evolution*