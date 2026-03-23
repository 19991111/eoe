"""
批量Agent状态容器
=================
GPU优化的Agent状态张量

所有状态存储在GPU显存中，支持批量操作
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentState:
    """
    Agent状态容器 (GPU张量)
    
    Attributes:
        positions: [N, 2] 位置 (x, y)
        velocities: [N, 2] 速度 (vx, vy)
        energies: [N] 内部能量
        thetas: [N] 朝向角
        permeabilities: [N] 渗透率 (0-1)
        defenses: [N] 防御力 (0-1)
        signals: [N] 信号强度 (0-1)
        is_alive: [N] 存活标志
    """
    positions: torch.Tensor
    velocities: torch.Tensor
    energies: torch.Tensor
    thetas: torch.Tensor
    permeabilities: torch.Tensor
    defenses: torch.Tensor
    signals: torch.Tensor
    is_alive: torch.Tensor
    
    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]
    
    @property
    def alive_mask(self) -> torch.Tensor:
        """存活掩码 (bool tensor)"""
        return self.is_alive
    
    @property
    def alive_indices(self) -> torch.Tensor:
        """存活Agent索引"""
        return torch.where(self.is_alive)[0]


def create_agent_state(
    n_agents: int,
    env_width: float,
    env_height: float,
    device: str = 'cuda:0',
    init_energy: float = 150.0
) -> AgentState:
    """
    创建Agent状态
    
    Args:
        n_agents: Agent数量
        env_width: 环境宽度
        env_height: 环境高度
        device: 设备
        init_energy: 初始能量
    """
    # 位置: 随机分布
    positions = torch.rand(n_agents, 2, device=device) * \
                torch.tensor([env_width, env_height], device=device)
    
    # 速度: 零初始化
    velocities = torch.zeros(n_agents, 2, device=device)
    
    # 能量: 固定值
    energies = torch.ones(n_agents, device=device) * init_energy
    
    # 朝向: 随机
    thetas = torch.rand(n_agents, device=device) * 2 * np.pi
    
    # 物理参数: 默认值
    permeabilities = torch.ones(n_agents, device=device) * 0.5
    defenses = torch.ones(n_agents, device=device) * 0.5
    signals = torch.zeros(n_agents, device=device)
    
    # 存活
    is_alive = torch.ones(n_agents, device=device, dtype=torch.bool)
    
    return AgentState(
        positions=positions,
        velocities=velocities,
        energies=energies,
        thetas=thetas,
        permeabilities=permeabilities,
        defenses=defenses,
        signals=signals,
        is_alive=is_alive
    )


def to_cpu(state: AgentState) -> dict:
    """
    转换到CPU (用于可视化)
    
    Returns:
        dict: numpy数组
    """
    return {
        'positions': state.positions.cpu().numpy(),
        'velocities': state.velocities.cpu().numpy(),
        'energies': state.energies.cpu().numpy(),
        'thetas': state.thetas.cpu().numpy(),
        'permeabilities': state.permeabilities.cpu().numpy(),
        'defenses': state.defenses.cpu().numpy(),
        'signals': state.signals.cpu().numpy(),
        'is_alive': state.is_alive.cpu().numpy(),
    }


__all__ = ['AgentState', 'create_agent_state', 'to_cpu']