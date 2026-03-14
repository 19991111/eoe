# 可微演化 (Differentiable Evolution) 架构升级方案

> 基于 EOE Project Lead 指令 | 版本: 1.1 | 日期: 2026-03-14
> 更新: 根据Architect AI批阅意见修正 v1.0 漏洞

---

## 一、当前状态分析

### 现有机制
| 模块 | 当前实现 | 问题 |
|------|----------|------|
| Brain前向传播 | 图遍历手动计算 | 不可微，无法反向传播 |
| 权重学习 | 简化Hebbian (标量更新) | 非梯度方法，收敛慢 |
| 遗传同化 | 无 | 子代继承随机突变，无经验传递 |
| 拓扑变异 | NEAT标准实现 | 纯随机，无梯度引导 |

---

## 二、架构修正 (v1.1)

### 🚨 修正一：打破"环境不可微"幻觉

**问题**: `energy_delta` 来自物理引擎，是标量无梯度，无法执行 `.backward()`

**修正方案**: 使用 **预测编码 (Self-Supervised)** + **能量调节学习率**

```python
class DifferentiableBrain(torch.nn.Module):
    def __init__(self, genome: OperatorGenome):
        super().__init__()
        # 主网络 - 动作输出
        self.action_head = ...
        # 预测头 - 预测下一帧感知
        self.prediction_head = ...
    
    def forward(self, sensor_input):
        # 主输出: 动作
        action = self.action_head(sensor_input)
        # 预测输出: 下一帧感知
        prediction = self.prediction_head(sensor_input)
        return action, prediction
    
    def compute_loss(self, sensor_t, sensor_tplus1, energy_delta):
        """
        L_local = MSE(prediction, actual_next_sensor) 
        
        能量奖励用于调节学习率:
        - 吃到能量 -> 学习率放大 (加深记忆)
        - 损失能量 -> 学习率缩小
        """
        prediction_loss = F.mse_loss(self.prediction, sensor_tplus1)
        
        # 能量调制学习率
        lr_modulator = 1.0 + torch.sigmoid(energy_delta) * 0.5
        
        return prediction_loss * lr_modulator
```

**为什么这种方法可行**:
- 整个过程在张量空间内，全程可微
- 能量不参与梯度计算，只调节学习率
- 预测编码让网络学会环境建模（隐式学习）

---

### 💡 修正二：使用 torch_geometric (PyG)

**问题**: Padding方案会导致海量冗余计算

**修正方案**: 全面拥抱 `torch_geometric`

```python
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

class DifferentiableBrain(MessagePassing):
    """
    继承 PyG MessagePassing
    支持变长图结构的批量前向+反向传播
    """
    
    def __init__(self, genome: OperatorGenome):
        # 将NEAT图转为PyG Data格式
        self.edge_index, self.edge_weight = self._compile_to_pyg(genome)
    
    def forward(self, x):
        """
        使用PyG的message passing进行前向传播
        保留完整计算图
        """
        return self.propagate(self.edge_index, x=x, edge_weight=self.edge_weight)
    
    def message(self, x_j, edge_weight):
        return x_j * edge_weight
    
    def update(self, aggr_out):
        return torch.relu(aggr_out)
```

**批量处理**:
```python
def batched_forward(brains: List[DifferentiableBrain], sensor_inputs: torch.Tensor):
    """
    使用 PyG Batch.from_data_list 批量处理
    一次性对整个种群进行前向和反向传播
    """
    graphs = [brain.to_data() for brain in brains]
    batch = pyg.Batch.from_data_list(graphs)
    
    # 一次性前向传播 (稀疏块对角矩阵)
    out = model(batch.x, batch.edge_index, batch.edge_attr)
    return out
```

---

### 🛡️ 修正三：鲍德温同化的拓扑保护

**问题**: 新增边的权重可能不在父代表型字典中，导致KeyError

**修正方案**: 严格掩码验证 + 极小值初始化

```python
def apply_baldwin_assimilation(self, parent_genome, kappa: float, sigma: float):
    """
    深度鲍德温遗传同化 with 拓扑保护
    
    W_child = W_parent + kappa * (W_phenotype - W_genotype) + N(0, sigma)
    
    规则:
    1. 边存在于父代表型中 -> 应用鲍德温公式
    2. 边是全新变异 -> 初始化为 0 (或 1e-4)
    """
    child = self.copy()
    
    # 获取父代的innovation编号集合
    parent_innovations = set(parent_genome.phenotype_weights.keys())
    
    for edge in child.edges:
        edge_id = self.get_edge_id(edge.source_id, edge.target_id)
        
        if edge_id in parent_innovations:
            # 旧边: 应用鲍德温同化
            w_genotype = edge.weight
            w_phenotype = parent_genome.phenotype_weights[edge_id]
            
            # 遗传同化
            delta = kappa * (w_phenotype - w_genotype)
            noise = torch.randn(1).item() * sigma
            
            edge.weight = w_genotype + delta + noise
        else:
            # 新边: 极小值初始化，绕过同化
            edge.weight = 1e-4  # 极小值，保护计算图
    
    return child
```

---

## 三、实施方案 (修正后)

### Phase 1: PyG可微计算图

#### 任务 1.1: 创建 DifferentiableBrain (基于PyG)

```python
# core/eoe/differentiable_brain.py (新文件)

import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing

class DifferentiableBrain(MessagePassing):
    """
    可微大脑 - 基于 PyG MessagePassing
    - 全程保留计算图
    - 支持批量处理
    """
    
    def __init__(self, genome: 'OperatorGenome'):
        super().__init__(aggr='add')  # 聚合方式
        self.genome = genome
        self._build_parameters()
    
    def _build_parameters(self):
        """将边权重转为可学习参数"""
        # 为每条边创建权重参数
        for i, edge in enumerate(self.genome.edges):
            self.register_parameter(
                f'weight_{i}',
                torch.nn.Parameter(torch.tensor(edge.weight))
            )
    
    def forward(self, x):
        """前向传播，保留计算图"""
        edge_attr = self._get_weights()
        return self.propagate(self.edge_index, x=x, edge_attr=edge_attr)
    
    def compute_local_loss(self, next_sensor_pred, next_sensor_actual, energy_delta):
        """
        预测编码损失 + 能量调制
        """
        pred_loss = F.mse_loss(next_sensor_pred, next_sensor_actual)
        lr_mod = 1.0 + torch.sigmoid(energy_delta) * 0.5
        return pred_loss * lr_mod
```

#### 任务 1.2: PoolConfig配置

```python
# batched_agents.py PoolConfig 新增:

DIFFERENTIABLE_BRAIN = False        # 启用可微大脑
DIFFERENTIABLE_USE_PYG = True       # 使用torch_geometric
PREDICTION_LOSS_WEIGHT = 0.1        # 预测损失权重
ENERGY_LR_MODULATOR = True          # 能量调节学习率

DIFFERENTIABLE_LR = 0.001
DIFFERENTIABLE_UPDATE_INTERVAL = 10

BALDWIN_ASSIMILATION_KAPPA = 0.5
BALDWIN_EXPLORATION_SIGMA = 0.01
```

---

### Phase 2: 生命周期梯度注入

#### 任务 2.1: LifecycleOptimizer (基于PyG vmap)

```python
class LifecycleOptimizer:
    """
    生命周期优化器
    - 使用 torch.vmap 批量梯度计算
    - 截断反向传播 (Truncated BPTT)
    """
    
    def __init__(self, agents: 'BatchedAgents'):
        self.agents = agents
        self.buffer = []
    
    @torch.vmap
    def batched_backward(self, losses: torch.Tensor):
        """
        使用 vmap 批量反向传播
        避免 for 循环显存爆炸
        """
        # 批量梯度计算
        pass
    
    def step(self, batch, sensor_t, sensor_tplus1, energy_deltas):
        # 收集经验
        self.buffer.append((sensor_t, sensor_tplus1, energy_deltas))
        
        # 每 N 步执行一次梯度更新
        if len(self.buffer) >= self.config.DIFFERENTIABLE_UPDATE_INTERVAL:
            losses = self._compute_batched_losses()
            self.batched_backward(losses)
            self.buffer.clear()
```

---

### Phase 3: 深度鲍德温遗传同化

#### 任务 3.1: 基因型/表型记录

```python
# genome.py 修改

class OperatorGenome:
    def __init__(self, ...):
        # 现有...
        
        # 深度鲍德温支持
        self.genotype_weights = {}  # 出生时权重
        self.phenotype_weights = {} # 死亡/繁殖时权重
    
    def save_phenotype(self):
        """保存当前权重作为表型"""
        for edge in self.edges:
            eid = self.get_edge_id(edge.source_id, edge.target_id)
            self.phenotype_weights[eid] = edge.weight
    
    def init_from_genotype(self):
        """从基因型初始化"""
        for edge in self.edges:
            eid = self.get_edge_id(edge.source_id, edge.target_id)
            if eid in self.genotype_weights:
                edge.weight = self.genotype_weights[eid]
            else:
                edge.weight = 1e-4  # 新边极小值
```

#### 任务 3.2: 拓扑保护同化

```python
def apply_baldwin_assimilation(self, parent, kappa, sigma):
    """拓扑保护的鲍德温同化"""
    child = self.copy()
    
    parent_innovations = set(parent.phenotype_weights.keys())
    
    for edge in child.edges:
        eid = self.get_edge_id(edge.source_id, edge.target_id)
        
        if eid in parent_innovations:
            # 旧边: 同化
            w_g = parent.genotype_weights.get(eid, edge.weight)
            w_p = parent.phenotype_weights[eid]
            edge.weight = w_g + kappa * (w_p - w_g) + random_normal() * sigma
        else:
            # 新边: 极小值保护
            edge.weight = 1e-4
    
    return child
```

---

## 四、技术难点与解决方案

### 难点1: 批量梯度显存爆炸
**方案**: `torch.vmap` + PyG Batch
```python
# 正确示范
out = model(batch.x, batch.edge_index, batch.edge_attr)
loss = out.sum()
loss.backward()  # 一次反向传播
```

### 难点2: 变长图结构
**方案**: 彻底使用 torch_geometric
```python
batch = pyg.Batch.from_data_list(graphs)
# 稀疏块对角矩阵，无Padding冗余
```

### 难点3: 环境不可微
**方案**: 预测编码 + 能量调制学习率
```python
# 完全张量空间
L = MSE(prediction, actual_next_sensor)
L = L * (1 + sigmoid(energy_delta))
```

---

## 五、修改文件清单

| 文件 | 修改 | 说明 |
|------|------|------|
| `core/eoe/differentiable_brain.py` | 新增 | PyG可微大脑 |
| `core/eoe/lifecycle_optimizer.py` | 新增 | 批量梯度优化器 |
| `core/eoe/genome.py` | 修改 | 基因型/表型+拓扑保护 |
| `core/eoe/batched_agents.py` | 修改 | 集成可微学习 |

---

## 六、预期效果

| 指标 | 当前 | 预期 | 提升 |
|------|------|------|------|
| 达到相同性能代数 | 1000代 | 50代 | **20x** |
| 复杂结构涌现 | 5000步 | 1000步 | **5x** |
| 显存效率 | O(N) | O(1) 批处理 | **显著改善** |

---

*方案 v1.1 - 根据 Architect AI 批阅意见修正*
*EOE Project - Differentiable Evolution*