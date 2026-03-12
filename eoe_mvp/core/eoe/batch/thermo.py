"""
热力学定律 (批量GPU版本)
========================
Agent与场之间的能量交互

物理模型:
- 能量提取: E_extract = EPF(x,y) * κ * rate
- 代谢消耗: E_cost = base + movement + signal²
- 信号注入: ISF += λ
"""

import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThermoStats:
    """热力学统计"""
    extracted: float
    metabolic: float
    signal_deposited: float
    mean_energy: float
    max_energy: float
    min_energy: float
    alive_count: int


class ThermodynamicLaw:
    """
    热力学定律 (GPU批量版本)
    """
    
    def __init__(
        self,
        device: str = 'cuda:0',
        extraction_rate: float = 0.5,
        signal_cost_coef: float = 0.01,
        metabolic_base: float = 0.001,
        max_energy: float = 200.0,
        min_energy: float = 0.0
    ):
        self.device = device
        
        # 物理参数
        self.extraction_rate = extraction_rate
        self.signal_cost_coef = signal_cost_coef
        self.metabolic_base = metabolic_base
        self.max_energy = max_energy
        self.min_energy = min_energy
        
        # 统计
        self.initial_universe_energy: Optional[float] = None
        self.energy_history = []
    
    def initialize(self, env: 'EnvironmentGPU', agents: 'BatchedAgents'):
        """初始化宇宙能量基准"""
        epf_energy = torch.sum(env.energy_field.field).item()
        agent_energy = torch.sum(agents.state.energies).item()
        
        self.initial_universe_energy = epf_energy + agent_energy
        self.energy_history.append(self.initial_universe_energy)
        
        return self.initial_universe_energy
    
    def apply(
        self,
        env: 'EnvironmentGPU',
        agents: 'BatchedAgents',
        alive_mask: torch.Tensor
    ) -> Tuple[ThermoStats, torch.Tensor]:
        """
        应用热力学定律
        
        Args:
            env: GPU环境
            agents: 批量Agent
            alive_mask: [N] 存活掩码
            
        Returns:
            (stats, new_alive_mask)
        """
        state = agents.state
        
        # === 1. EPF能量提取 ===
        field_values = env.get_field_values(state.positions)  # [N, 9]
        
        epf_values = field_values[:, 0]  # [N] EPF中心值
        extraction = epf_values * state.permeabilities * self.extraction_rate
        extraction = extraction * alive_mask.float()
        
        # === 2. 信号注入ISF ===
        signal_deposit = state.signals * alive_mask.float()
        
        # === 3. 代谢消耗 ===
        metabolic = (
            self.metabolic_base +
            torch.sum(torch.abs(state.velocities), dim=1) * 0.01 +
            state.signals ** 2 * self.signal_cost_coef
        )
        metabolic = metabolic * alive_mask.float()
        
        # === 4. 能量更新 ===
        new_energies = state.energies + extraction - metabolic
        new_energies = torch.clamp(new_energies, self.min_energy, self.max_energy)
        
        # === 5. 死亡判定 ===
        new_alive_mask = (new_energies > self.min_energy)
        
        # === 6. ISF批量注入 ===
        if hasattr(env, 'stigmergy_field') and env.stigmergy_field_enabled:
            alive_indices = torch.where(alive_mask & new_alive_mask)[0]
            if len(alive_indices) > 0:
                pos = state.positions[alive_indices]
                sig = signal_deposit[alive_indices]
                env.stigmergy_field.deposit_batch(pos, sig)
        
        # 更新能量
        state.energies = new_energies
        
        # 统计
        stats = ThermoStats(
            extracted=torch.sum(extraction).item(),
            metabolic=torch.sum(metabolic).item(),
            signal_deposited=torch.sum(signal_deposit).item(),
            mean_energy=torch.mean(new_energies).item(),
            max_energy=torch.max(new_energies).item(),
            min_energy=torch.min(new_energies).item(),
            alive_count=torch.sum(new_alive_mask).item()
        )
        
        return stats, new_alive_mask
    
    def check_stability(
        self,
        env: 'EnvironmentGPU',
        agents: 'BatchedAgents'
    ) -> Tuple[bool, float]:
        """
        检查能量稳定性
        
        Returns:
            (is_stable, total_energy)
        """
        epf_energy = torch.sum(env.energy_field.field).item()
        isf_energy = torch.sum(env.stigmergy_field.field).item()
        
        alive_mask = agents.state.energies > self.min_energy
        agent_energy = torch.sum(agents.state.energies * alive_mask.float()).item()
        
        total = epf_energy + isf_energy + agent_energy
        
        # 检查是否在合理范围
        initial = self.initial_universe_energy or total
        is_stable = (initial * 0.1) < total < (initial * 2.0)
        
        self.energy_history.append(total)
        
        return is_stable, total


# 兼容旧版本
ThermodynamicLawGPU = ThermodynamicLaw