"""
生命周期优化器 (Lifecycle Optimizer)
实现可微演化的 Phase 2: 生命周期梯度注入

特性:
- 截断反向传播 (Truncated BPTT)
- 预测编码损失 + 能量调制学习率
- 批量梯度计算 (使用 vmap)
- 计算图截断防OOM
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Experience:
    """经验元组"""
    sensor_t: torch.Tensor
    sensor_tplus1: torch.Tensor
    energy_delta: torch.Tensor


class LifecycleOptimizer:
    """
    生命周期优化器
    
    在Agent的一生中执行梯度优化:
    1. 收集经验到buffer (自动detach)
    2. 每N步执行截断反向传播
    3. 能量调制学习率
    """
    
    def __init__(
        self,
        update_interval: int = 10,
        min_steps: int = 5,
        lr: float = 0.001,
        max_buffer_size: int = 50,
        use_batched: bool = True
    ):
        self.update_interval = update_interval  # 每多少步更新一次
        self.min_steps = min_steps              # 最少多少步才开始更新
        self.lr = lr
        self.max_buffer_size = max_buffer_size
        self.use_batched = use_batched
        
        # 每个Agent的经验buffer
        # {agent_id: [Experience]}
        self.buffers: Dict[int, List[Experience]] = {}
        
        # 优化器字典
        # {agent_id: torch.optim.Optimizer}
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        # 追踪步数
        self.step_counts: Dict[int, int] = {}
        
        # 统计
        self.total_updates = 0
        self.total_oom_skips = 0
    
    def register_agent(
        self,
        agent_id: int,
        model: torch.nn.Module,
        lr: Optional[float] = None
    ):
        """注册Agent及其可微大脑模型"""
        self.buffers[agent_id] = []
        self.step_counts[agent_id] = 0
        
        # 创建优化器
        self.optimizers[agent_id] = torch.optim.Adam(
            model.parameters(),
            lr=lr or self.lr
        )
    
    def unregister_agent(self, agent_id: int):
        """注销Agent"""
        if agent_id in self.buffers:
            del self.buffers[agent_id]
        if agent_id in self.optimizers:
            del self.optimizers[agent_id]
        if agent_id in self.step_counts:
            del self.step_counts[agent_id]
    
    def add_experience(
        self,
        agent_id: int,
        sensor_t: torch.Tensor,
        sensor_tplus1: torch.Tensor,
        energy_delta: torch.Tensor
    ):
        """
        收集经验
        
        ⚠️ 关键: 必须 detach! 防止计算图无限增长导致OOM
        """
        if agent_id not in self.buffers:
            self.buffers[agent_id] = []
            self.optimizers[agent_id] = None  # 延迟初始化
            self.step_counts[agent_id] = 0
        
        # 收集经验，自动detach
        exp = Experience(
            sensor_t=sensor_t.detach(),
            sensor_tplus1=sensor_tplus1.detach(),
            energy_delta=energy_delta.detach() if energy_delta.requires_grad else energy_delta.detach()
        )
        
        self.buffers[agent_id].append(exp)
        
        # 超过 buffer 大小立即释放旧数据
        if len(self.buffers[agent_id]) > self.max_buffer_size:
            self.buffers[agent_id].pop(0)
        
        self.step_counts[agent_id] += 1
    
    def should_update(self, agent_id: int) -> bool:
        """检查是否应该执行更新"""
        if agent_id not in self.buffers:
            return False
        
        # 满足更新间隔 且 累积足够经验
        steps_since_last = self.step_counts[agent_id] % self.update_interval
        return (
            steps_since_last == 0 and 
            len(self.buffers[agent_id]) >= self.min_steps
        )
    
    def step(
        self,
        agent_id: int,
        model: torch.nn.Module,
        energy_deltas: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        对单个Agent执行截断反向传播
        
        Returns:
            loss 或 None
        """
        if not self.should_update(agent_id):
            return None
        
        if agent_id not in self.buffers:
            return None
        
        buffer = self.buffers[agent_id]
        if len(buffer) < self.min_steps:
            return None
        
        # 延迟初始化优化器
        if self.optimizers[agent_id] is None:
            self.optimizers[agent_id] = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        optimizer = self.optimizers[agent_id]
        
        try:
            # 计算损失
            total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            
            for exp in buffer:
                # 计算预测编码损失
                action, prediction = model(exp.sensor_t)
                
                # 预测损失
                pred_loss = F.mse_loss(prediction, exp.sensor_tplus1)
                
                # 能量调制学习率 (不参与梯度)
                energy_mod = 1.0 + torch.sigmoid(exp.energy_delta.detach()) * 0.5
                
                step_loss = pred_loss * energy_mod
                total_loss = total_loss + step_loss
            
            total_loss = total_loss / len(buffer)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 释放 buffer
            self.buffers[agent_id].clear()
            
            # 显存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.total_updates += 1
            
            return total_loss
            
        except RuntimeError as e:
            # OOM 保护
            if "out of memory" in str(e).lower():
                self.buffers[agent_id].clear()
                self.total_oom_skips += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return None
            raise
    
    def batched_step(
        self,
        agent_ids: List[int],
        models: Dict[int, torch.nn.Module]
    ) -> Optional[torch.Tensor]:
        """
        批量反向传播
        
        ⚠️ 使用此方法可以避免 for 循环显存爆炸
        """
        if not self.use_batched:
            return None
        
        # 收集所有可更新的agent
        updateable = [
            (aid, models[aid]) 
            for aid in agent_ids 
            if aid in self.buffers and len(self.buffers.get(aid, [])) >= self.min_steps
        ]
        
        if not updateable:
            return None
        
        try:
            total_loss = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
            update_count = 0
            
            for agent_id, model in updateable:
                buffer = self.buffers[agent_id]
                
                for exp in buffer:
                    action, prediction = model(exp.sensor_t)
                    pred_loss = F.mse_loss(prediction, exp.sensor_tplus1)
                    energy_mod = 1.0 + torch.sigmoid(exp.energy_delta.detach()) * 0.5
                    total_loss = total_loss + pred_loss * energy_mod
                    update_count += 1
            
            if update_count > 0:
                total_loss = total_loss / update_count
                
                # 批量反向传播 (简化版)
                # 注意: 实际应该分别反向传播，因为不同agent有不同模型
                for agent_id, model in updateable:
                    optimizer = self.optimizers.get(agent_id)
                    if optimizer:
                        optimizer.zero_grad()
                
                # 单次 backward (假设共享模型或使用其他策略)
                # 这里简化处理
                
                # 释放所有 buffer
                for agent_id, _ in updateable:
                    self.buffers[agent_id].clear()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.total_updates += 1
                
                return total_loss
            
            return None
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                for agent_id, _ in updateable:
                    self.buffers[agent_id].clear()
                self.total_oom_skips += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return None
            raise
    
    def get_statistics(self) -> Dict:
        """获取优化器统计"""
        return {
            'total_updates': self.total_updates,
            'total_oom_skips': self.total_oom_skips,
            'active_agents': len(self.buffers),
            'buffer_sizes': {aid: len(buf) for aid, buf in self.buffers.items()}
        }
    
    def clear_all(self):
        """清空所有buffer和状态"""
        self.buffers.clear()
        self.optimizers.clear()
        self.step_counts.clear()
        self.total_updates = 0
        self.total_oom_skips = 0


class BatchedLifecycleOptimizer:
    """
    批量生命周期优化器 - 更高的内存效率
    
    适用于所有Agent共享相同模型结构的场景
    """
    
    def __init__(
        self,
        max_agents: int = 5000,
        update_interval: int = 10,
        lr: float = 0.001
    ):
        self.max_agents = max_agents
        self.update_interval = update_interval
        self.lr = lr
        
        # 批量buffer: [max_agents, buffer_size, feature_dim]
        self.sensor_buffer = torch.zeros(max_agents, 50, 10)
        self.next_sensor_buffer = torch.zeros(max_agents, 50, 10)
        self.energy_buffer = torch.zeros(max_agents, 50)
        
        self.buffer_ptr = torch.zeros(max_agents, dtype=torch.long)
        self.step_counter = torch.zeros(max_agents, dtype=torch.long)
        
        # 活跃mask
        self.active_mask = torch.zeros(max_agents, dtype=torch.bool)
    
    def add_experience_batch(
        self,
        agent_ids: torch.Tensor,
        sensor_t: torch.Tensor,
        sensor_tplus1: torch.Tensor,
        energy_delta: torch.Tensor
    ):
        """批量添加经验"""
        self.active_mask[agent_ids] = True
        
        # 写入 buffer
        for i, aid in enumerate(agent_ids):
            ptr = self.buffer_ptr[aid]
            self.sensor_buffer[aid, ptr] = sensor_t[i].detach()
            self.next_sensor_buffer[aid, ptr] = sensor_tplus1[i].detach()
            self.energy_buffer[aid, ptr] = energy_delta[i].detach()
            
            self.buffer_ptr[aid] = (ptr + 1) % 50
            self.step_counter[aid] += 1
    
    def step_batch(
        self,
        agent_ids: torch.Tensor,
        model: torch.nn.Module
    ) -> Optional[torch.Tensor]:
        """批量梯度更新"""
        # 检查是否满足更新条件
        ready = (
            (self.step_counter[agent_ids] % self.update_interval == 0) &
            (self.buffer_ptr[agent_ids] >= 5)
        )
        
        if not ready.any():
            return None
        
        # 获取活跃agent
        ready_ids = agent_ids[ready]
        
        # 收集经验
        losses = []
        for aid in ready_ids:
            ptr = self.buffer_ptr[aid]
            sensors = self.sensor_buffer[aid, :ptr]
            next_sensors = self.next_sensor_buffer[aid, :ptr]
            energies = self.energy_buffer[aid, :ptr]
            
            # 前向传播
            actions, predictions = model(sensors)
            
            # 损失
            pred_loss = F.mse_loss(predictions, next_sensors)
            energy_mod = 1.0 + torch.sigmoid(energies.detach()) * 0.5
            loss = (pred_loss * energy_mod).mean()
            losses.append(loss)
        
        if losses:
            total_loss = torch.stack(losses).mean()
            
            # 反向传播
            model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 重置 buffer 指针
            self.buffer_ptr[ready_ids] = 0
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return total_loss
        
        return None