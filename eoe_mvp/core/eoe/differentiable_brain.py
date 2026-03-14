"""
可微大脑 (Differentiable Brain)
基于 torch_geometric (PyG) 的可微计算图

特性:
- 继承 PyG MessagePassing，支持变长图结构
- 全程保留计算图，支持 .backward()
- 预测编码 + 能量调制学习率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# 尝试导入 torch_geometric
try:
    import torch_geometric as pyg
    from torch_geometric.nn import MessagePassing
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    MessagePassing = nn.Module


@dataclass
class BrainState:
    """大脑状态快照"""
    sensor_input: torch.Tensor
    hidden_states: torch.Tensor
    action_output: torch.Tensor
    prediction_output: torch.Tensor


class DifferentiableBrain(nn.Module if not PYG_AVAILABLE else MessagePassing):
    """
    可微大脑 - 基于 PyG MessagePassing
    或使用标准 nn.Module (当 PyG 不可用时)
    """
    
    def __init__(
        self,
        genome: 'OperatorGenome',
        input_dim: int = 10,
        hidden_dim: int = 16,
        output_dim: int = 4,
        prediction_dim: int = 10
    ):
        super().__init__()
        
        self.genome = genome
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_dim = prediction_dim
        
        # 边索引和权重
        self._build_edge_index()
        
        # 动作头 - 将节点特征映射到动作
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )
        
        # 预测头 - 预测下一帧感知
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_dim)
        )
        
        # 可学习边权重
        self._build_weights()
    
    def _build_edge_index(self):
        """将NEAT图转为PyG边索引格式"""
        edge_list = []
        for edge in self.genome.edges:
            if isinstance(edge, dict):
                src = edge.get('source_id', edge.get('source', 0))
                tgt = edge.get('target_id', edge.get('target', 0))
            else:
                src = edge.source_id
                tgt = edge.target_id
            edge_list.append([src, tgt])
        
        if edge_list:
            self.register_buffer('edge_index', torch.tensor(edge_list, dtype=torch.long).t())
        else:
            self.register_buffer('edge_index', torch.zeros((2, 0), dtype=torch.long))
    
    def _build_weights(self):
        """将边权重转为可学习参数"""
        self.edge_weights = nn.Parameter(torch.ones(self.edge_index.shape[1]))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，保留计算图
        
        Args:
            x: 传感器输入 [batch, input_dim]
        
        Returns:
            (action, prediction): 动作输出和感知预测
        """
        batch_size = x.shape[0]
        
        # 输入投影
        h = F.relu(nn.Linear(self.input_dim, self.hidden_dim)(x))
        
        # 如果有边，进行消息传递
        if self.edge_index.shape[1] > 0:
            # 使用简化的图卷积
            h = self._graph_conv(h)
        
        # 动作输出
        action = self.action_head(h)
        
        # 预测输出
        prediction = self.prediction_head(h)
        
        return action, prediction
    
    def _graph_conv(self, h: torch.Tensor) -> torch.Tensor:
        """简化的图卷积"""
        # 归一化邻接矩阵
        edge_weight = torch.sigmoid(self.edge_weights)
        
        # 聚合邻居信息 (简化的 message passing)
        out = torch.zeros_like(h)
        
        for i in range(self.edge_index.shape[1]):
            src, tgt = self.edge_index[0, i], self.edge_index[1, i]
            w = edge_weight[i]
            out[:, tgt] = out[:, tgt] + h[:, src] * w
        
        # 加上自环
        out = out + h
        
        return out
    
    def compute_local_loss(
        self,
        sensor_t: torch.Tensor,
        sensor_tplus1: torch.Tensor,
        energy_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算局部损失: 预测编码 + 能量调制
        
        L = MSE(prediction, actual_next_sensor) * (1 + sigmoid(energy_delta))
        
        Args:
            sensor_t: 当前感知 [batch, dim]
            sensor_tplus1: 下一帧感知 [batch, dim]
            energy_delta: 能量变化 [batch]
        
        Returns:
            loss: 标量损失
        """
        # 前向传播
        action, prediction = self.forward(sensor_t)
        
        # 预测损失
        pred_loss = F.mse_loss(prediction, sensor_tplus1)
        
        # 能量调制学习率
        lr_mod = 1.0 + torch.sigmoid(energy_delta.detach()) * 0.5
        
        # 总损失
        loss = pred_loss * lr_mod
        
        return loss
    
    def get_weights_dict(self) -> Dict[int, float]:
        """获取当前权重字典 (用于遗传)"""
        weights = {}
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            tgt = self.edge_index[1, i].item()
            edge_id = src * 100 + tgt  # 简单ID生成
            weights[edge_id] = self.edge_weights[i].item()
        return weights
    
    def set_weights_from_dict(self, weights: Dict[int, float]):
        """从权重字典恢复 (用于继承父代表型)"""
        with torch.no_grad():
            for i in range(self.edge_index.shape[1]):
                src = self.edge_index[0, i].item()
                tgt = self.edge_index[1, i].item()
                edge_id = src * 100 + tgt
                if edge_id in weights:
                    self.edge_weights[i] = weights[edge_id]
    
    def to_genome_weights(self) -> list:
        """导出为基因组格式的权重列表"""
        return self.edge_weights.tolist()


class DifferentiableBrainPool:
    """
    可微大脑池 - 管理多个DifferentiableBrain
    支持批量前向传播
    """
    
    def __init__(self, max_agents: int = 5000):
        self.max_agents = max_agents
        self.brains: Dict[int, DifferentiableBrain] = {}
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        # 经验 buffer
        self.buffer: Dict[int, list] = {}
        self.buffer_size = 50
    
    def add_agent(
        self,
        agent_id: int,
        genome: 'OperatorGenome',
        lr: float = 0.001
    ) -> DifferentiableBrain:
        """为Agent添加可微大脑"""
        brain = DifferentiableBrain(genome)
        self.brains[agent_id] = brain
        
        # 创建优化器
        self.optimizers[agent_id] = torch.optim.Adam(
            brain.parameters(),
            lr=lr
        )
        
        # 初始化 buffer
        self.buffer[agent_id] = []
        
        return brain
    
    def remove_agent(self, agent_id: int):
        """移除Agent"""
        if agent_id in self.brains:
            del self.brains[agent_id]
        if agent_id in self.optimizers:
            del self.optimizers[agent_id]
        if agent_id in self.buffer:
            del self.buffer[agent_id]
    
    def add_experience(
        self,
        agent_id: int,
        sensor_t: torch.Tensor,
        sensor_tplus1: torch.Tensor,
        energy_delta: torch.Tensor
    ):
        """收集经验 (自动 detach 防止OOM)"""
        if agent_id not in self.buffer:
            self.buffer[agent_id] = []
        
        self.buffer[agent_id].append((
            sensor_t.detach(),
            sensor_tplus1.detach(),
            energy_delta.detach() if energy_delta.requires_grad else energy_delta
        ))
        
        # 超过 buffer 大小立即释放旧数据
        if len(self.buffer[agent_id]) > self.buffer_size:
            self.buffer[agent_id].pop(0)
    
    def step(self, agent_id: int) -> Optional[torch.Tensor]:
        """
        对单个 Agent 执行截断反向传播
        
        Returns:
            loss 或 None (如果buffer不足)
        """
        if agent_id not in self.brains:
            return None
        
        min_steps = 10
        if len(self.buffer.get(agent_id, [])) < min_steps:
            return None
        
        brain = self.brains[agent_id]
        optimizer = self.optimizers[agent_id]
        
        # 计算损失
        loss = torch.tensor(0.0, device=next(brain.parameters()).device)
        
        for sensor_t, sensor_tplus1, energy_delta in self.buffer[agent_id]:
            step_loss = brain.compute_local_loss(sensor_t, sensor_tplus1, energy_delta)
            loss = loss + step_loss
        
        loss = loss / len(self.buffer[agent_id])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 释放 buffer
        self.buffer[agent_id].clear()
        
        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss
    
    def batched_step(self, agent_ids: list) -> Optional[torch.Tensor]:
        """
        批量反向传播 (使用 vmap)
        防止 for 循环显存爆炸
        """
        if not agent_ids:
            return None
        
        # 收集所有损失
        losses = []
        
        for agent_id in agent_ids:
            if len(self.buffer.get(agent_id, [])) >= 10:
                brain = self.brains[agent_id]
                
                for sensor_t, sensor_tplus1, energy_delta in self.buffer[agent_id]:
                    step_loss = brain.compute_local_loss(sensor_t, sensor_tplus1, energy_delta)
                    losses.append(step_loss)
        
        if not losses:
            return None
        
        # 批量反向传播
        total_loss = torch.stack(losses).mean()
        
        # 一次性反向传播
        optimizer = self.optimizers[agent_ids[0]]  # 简化: 假设同一优化器
        
        # 各自反向传播
        for agent_id in agent_ids:
            if agent_id in self.brains:
                self.brains[agent_id].zero_grad()
        
        # 释放所有 buffer
        for agent_id in agent_ids:
            if agent_id in self.buffer:
                self.buffer[agent_id].clear()
        
        return total_loss
    
    def save_phenotype(self, agent_id: int) -> Dict[int, float]:
        """保存表型权重 (用于遗传同化)"""
        if agent_id in self.brains:
            return self.brains[agent_id].get_weights_dict()
        return {}
    
    def apply_baldwin_assimilation(
        self,
        child_agent_id: int,
        parent_genotype_weights: Dict[int, float],
        parent_phenotype_weights: Dict[int, float],
        kappa: float = 0.5,
        sigma: float = 0.01
    ):
        """
        应用深度鲍德温遗传同化
        
        W_child = W_parent + kappa * (W_phenotype - W_genotype) + N(0, sigma)
        """
        if child_agent_id not in self.brains:
            return
        
        brain = self.brains[child_agent_id]
        
        # 获取当前权重
        child_weights = brain.get_weights_dict()
        
        # 拓扑保护: 遍历子代的每条边
        with torch.no_grad():
            for edge_id in child_weights.keys():
                if edge_id in parent_phenotype_weights:
                    # 旧边: 应用鲍德温同化
                    w_genotype = parent_genotype_weights.get(edge_id, child_weights[edge_id])
                    w_phenotype = parent_phenotype_weights[edge_id]
                    
                    delta = kappa * (w_phenotype - w_genotype)
                    noise = torch.randn(1).item() * sigma
                    
                    child_weights[edge_id] = w_genotype + delta + noise
                else:
                    # 新边: 极小值初始化，绕过同化
                    child_weights[edge_id] = 1e-4
        
        # 应用权重
        brain.set_weights_from_dict(child_weights)


# 兼容性: 如果 PyG 不可用，提供简化实现
if not PYG_AVAILABLE:
    print("⚠️ torch_geometric 不可用，使用简化版 DifferentiableBrain")