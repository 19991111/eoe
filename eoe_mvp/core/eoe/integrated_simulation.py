"""
v13.0 Phase 3: 热力学闭环与集成引擎
====================================
统一场物理 + 批量Agent的热力学集成

特性:
- 能量守恒探针 (Energy Conservation Probe)
- 生死张量掩码 (Life/Death Masking)
- 完整的GPU仿真循环
- 观测可视化接口
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SimState:
    """仿真状态容器"""
    generation: int
    step: int
    total_energy: float
    alive_count: int
    mean_energy: float
    max_energy: float
    min_energy: float


class ThermodynamicLaw:
    """
    热力学定律 (GPU 加速版本)
    ==========================
    负责 Agent 与 场之间的能量交互
    """
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        
        # 物理参数
        self.base_extraction_rate = 0.5  # 基础能量提取率
        self.signal_cost_coef = 0.01     # 信号代价系数
        self.metabolic_base = 0.001      # 基础代谢率
        self.max_energy = 200.0          # 最大能量上限
        self.min_energy = 0.0            # 死亡阈值
        
        # 能量统计
        self.initial_universe_energy = None
        self.energy_history = []
    
    def initialize_universe(self, env: 'EnvironmentGPU', agents: 'BatchedAgents'):
        """初始化宇宙总能量"""
        # EPF 场能量
        epf_energy = torch.sum(env.energy_field.field).item()
        
        # Agent 能量
        agent_energy = torch.sum(agents.state.energies).item()
        
        self.initial_universe_energy = epf_energy + agent_energy
        self.energy_history.append(self.initial_universe_energy)
        
        print(f"[ThermodynamicLaw] 初始宇宙能量: {self.initial_universe_energy:.2f}")
        return self.initial_universe_energy
    
    def apply(
        self,
        env: 'EnvironmentGPU',
        agents: 'BatchedAgents',
        alive_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        应用热力学定律 (GPU 批量计算)
        
        Args:
            env: GPU 环境
            agents: 批量 Agent
            alive_mask: [N] 存活掩码
            
        Returns:
            dict: 能量变化统计
        """
        N = agents.n_agents
        device = agents.device
        
        # === 1. 从 EPF 场提取能量 ===
        # 获取当前位置的场值
        field_values = env.get_field_values(agents.state.positions)  # [N, 9]
        
        # EPF 能量提取 (仅存活Agent)
        epf_extraction = field_values[:, 0] * agents.state.permeabilities * self.base_extraction_rate
        epf_extraction = epf_extraction * alive_mask.float()
        
        # === 2. 信号注入 ISF 场 ===
        signal_deposit = agents.state.signals * alive_mask.float()
        
        # === 3. 代谢消耗 ===
        # 基础代谢 + 运动代谢 + 信号代谢
        metabolic_cost = (
            self.metabolic_base + 
            torch.sum(torch.abs(agents.state.velocities), dim=1) * 0.01 +
            agents.state.signals ** 2 * self.signal_cost_coef
        )
        metabolic_cost = metabolic_cost * alive_mask.float()
        
        # === 4. 能量更新 ===
        new_energies = agents.state.energies + epf_extraction - metabolic_cost
        
        # 钳制到有效范围
        new_energies = torch.clamp(new_energies, self.min_energy, self.max_energy)
        
        # === 5. 死亡判定 ===
        new_alive_mask = (new_energies > self.min_energy)
        
        # === 6. ISF 信号注入 (向场写入) ===
        if env.stigmergy_field_enabled:
            # 批量注入信号 (需要遍历每个存活Agent的位置)
            alive_indices = torch.where(alive_mask)[0]
            for idx in alive_indices:
                if new_alive_mask[idx] and signal_deposit[idx] > 0:
                    pos = agents.state.positions[idx]
                    env.stigmergy_field.deposit(pos[0].item(), pos[1].item(), signal_deposit[idx].item())
        
        # 更新 Agent 能量
        agents.state.energies = new_energies
        
        # 统计
        stats = {
            'extracted': torch.sum(epf_extraction).item(),
            'metabolic': torch.sum(metabolic_cost).item(),
            'signal_deposited': torch.sum(signal_deposit).item(),
            'mean_energy': torch.mean(new_energies).item(),
            'max_energy': torch.max(new_energies).item(),
            'min_energy': torch.min(new_energies).item(),
            'alive_count': torch.sum(new_alive_mask).item(),
        }
        
        return stats, new_alive_mask
    
    def check_energy_conservation(
        self,
        env: 'EnvironmentGPU',
        agents: 'BatchedAgents',
        step: int
    ) -> Tuple[bool, float]:
        """
        检查能量守恒 (考虑开放的源注入)
        
        注意: EPF源持续向系统注入能量，因此检查的是:
        - Agent能量变化是否合理
        - 场能量是否在预期范围内
        
        Returns:
            (is_stable, current_total_energy)
        """
        # EPF 场能量
        epf_energy = torch.sum(env.energy_field.field).item()
        
        # ISF 场能量 (压痕场也有能量)
        isf_energy = torch.sum(env.stigmergy_field.field).item()
        
        # Agent 能量 (仅存活)
        alive_mask = agents.state.energies > self.min_energy
        agent_energy = torch.sum(agents.state.energies * alive_mask.float()).item()
        
        current_total = epf_energy + isf_energy + agent_energy
        
        # 检查能量是否在合理范围 (不爆炸)
        # 预期: EPF场有源注入会增长, ISF场有衰减, Agent有代谢
        # 允许较大范围变化 (50%)
        initial = self.initial_universe_energy
        max_allowed = initial * 2.0  # 允许最多翻倍
        min_allowed = initial * 0.1  # 允许最少降到10%
        
        is_stable = min_allowed < current_total < max_allowed
        
        self.energy_history.append(current_total)
        
        return is_stable, current_total


class IntegratedSimulation:
    """
    集成仿真引擎
    =============
    统一管理环境、Agent、热力学定律
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        env_width: float = 100.0,
        env_height: float = 100.0,
        lifespan: int = 1500,
        device: str = 'cuda:0'
    ):
        self.n_agents = n_agents
        self.lifespan = lifespan
        self.device = device
        
        print("="*60)
        print("🚀 初始化集成仿真引擎 (GPU)")
        print("="*60)
        
        # 1. 环境
        from core.eoe.environment_gpu import EnvironmentGPU
        self.env = EnvironmentGPU(
            width=env_width,
            height=env_height,
            device=device,
            energy_field_enabled=True,
            impedance_field_enabled=True,
            stigmergy_field_enabled=True
        )
        
        # 2. 批量 Agent
        from core.eoe.batched_agents import BatchedAgents
        self.agents = BatchedAgents(
            n_agents=n_agents,
            env_width=env_width,
            env_height=env_height,
            device=device,
            init_energy=150.0
        )
        
        # 3. 热力学定律
        self.thermo = ThermodynamicLaw(device=device)
        self.thermo.initialize_universe(self.env, self.agents)
        
        # 状态跟踪
        self.alive_mask = torch.ones(n_agents, device=device, dtype=torch.bool)
        self.generation = 0
        self.step_count = 0
        
        # 历史记录
        self.history: List[SimState] = []
        
        print(f"\n✅ 集成引擎就绪")
        print(f"   Agent数量: {n_agents}")
        print(f"   生命周期: {lifespan}")
        print(f"   设备: {device}")
    
    def step(self, verbose: bool = False) -> SimState:
        """执行单步"""
        # === Step 1: 获取传感器 ===
        sensors = self.agents.get_sensors(self.env)  # [N, 10]
        
        # === Step 2: 神经网络前向 ===
        brain_outputs = self.agents.forward_brains(sensors)  # [N, 5]
        
        # === Step 3: Agent 物理更新 (应用脑输出) ===
        self.agents.step(brain_outputs)
        
        # === Step 4: 热力学交互 (Agent ↔ 场) ===
        thermo_stats, self.alive_mask = self.thermo.apply(
            self.env, self.agents, self.alive_mask
        )
        
        # === Step 5: 场物理更新 ===
        self.env.step()
        
        # === Step 6: 能量守恒检查 ===
        is_conserved, total_energy = self.thermo.check_energy_conservation(
            self.env, self.agents, self.step_count
        )
        
        if not is_conserved and self.step_count > 10:
            print(f"\n⚠️ 能量守恒警告 @ step {self.step_count}")
            print(f"   初始能量: {self.thermo.initial_universe_energy:.4f}")
            print(f"   当前能量: {total_energy:.4f}")
            print(f"   差异: {abs(total_energy - self.thermo.initial_universe_energy):.6f}")
        
        # === 记录状态 ===
        state = SimState(
            generation=self.generation,
            step=self.step_count,
            total_energy=total_energy,
            alive_count=int(thermo_stats['alive_count']),
            mean_energy=thermo_stats['mean_energy'],
            max_energy=thermo_stats['max_energy'],
            min_energy=thermo_stats['min_energy']
        )
        self.history.append(state)
        
        if verbose and self.step_count % 100 == 0:
            print(f"  Step {self.step_count}: {thermo_stats['alive_count']} alive, "
                  f"energy={total_energy:.2f}, conserved={is_conserved}")
        
        self.step_count += 1
        return state
    
    def run(
        self,
        max_steps: Optional[int] = None,
        verbose: bool = True
    ) -> List[SimState]:
        """运行完整生命周期"""
        max_steps = max_steps or self.lifespan
        
        if verbose:
            print(f"\n🔄 运行 {max_steps} 步仿真...")
        
        import time
        start = time.perf_counter()
        
        for step in range(max_steps):
            self.step(verbose=(verbose and step % 100 == 0))
            
            # 早停: 如果全部死亡
            if torch.sum(self.alive_mask) == 0:
                if verbose:
                    print(f"  ⚠️ 全部Agent死亡于 step {step}")
                break
        
        torch.cuda.synchronize() if self.device.startswith('cuda') else None
        elapsed = time.perf_counter() - start
        
        if verbose:
            print(f"\n✅ 仿真完成!")
            print(f"   耗时: {elapsed:.2f}秒")
            print(f"   存活: {self.history[-1].alive_count}/{self.n_agents}")
            print(f"   平均能量: {self.history[-1].mean_energy:.2f}")
        
        return self.history
    
    def get_visualization_data(self) -> Dict[str, np.ndarray]:
        """获取可视化数据 (CPU 数组)"""
        return {
            # Agent 状态
            'positions': self.agents.state.positions.cpu().numpy(),
            'energies': self.agents.state.energies.cpu().numpy(),
            'permeabilities': self.agents.state.permeabilities.cpu().numpy(),
            'alive_mask': self.alive_mask.cpu().numpy(),
            
            # 场数据
            'epf_field': self.env.energy_field.field[0, 0].cpu().numpy(),
            'isf_field': self.env.stigmergy_field.field[0, 0].cpu().numpy(),
            'kif_field': self.env.impedance_field.field[0, 0].cpu().numpy(),
        }


def benchmark_integrated_simulation():
    """集成仿真基准测试"""
    import time
    
    print("\n" + "="*60)
    print("🎯 集成仿真基准测试")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 小规模测试
    sim = IntegratedSimulation(
        n_agents=500,
        lifespan=1500,
        device=device
    )
    
    # 预热
    print("\n预热 (20步)...")
    for _ in range(20):
        sim.step()
    
    torch.cuda.synchronize() if device.startswith('cuda') else None
    
    # 基准测试
    print("\n运行 500 步...")
    start = time.perf_counter()
    
    for _ in range(500):
        sim.step()
    
    torch.cuda.synchronize() if device.startswith('cuda') else None
    elapsed = time.perf_counter() - start
    
    print(f"\n📊 结果:")
    print(f"  500步 x 500 agents: {elapsed:.2f}秒")
    print(f"  平均每步: {elapsed/500*1000:.2f}ms")
    print(f"  吞吐量: {500*500/elapsed:.0f} agent-steps/sec")
    
    # 1500步预估
    estimated_1500 = elapsed / 500 * 1500
    print(f"\n  1500步预估: {estimated_1500:.2f}秒")
    
    return sim


if __name__ == "__main__":
    benchmark_integrated_simulation()