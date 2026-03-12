"""
统一仿真入口
============
GPU加速的演化仿真主入口

Usage:
    from core.eoe.batch import Simulation
    
    # 创建仿真
    sim = Simulation(n_agents=1000, device='cuda:0')
    
    # 运行
    history = sim.run(1500)
"""

import torch
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .state import AgentState, create_agent_state, to_cpu
from .thermo import ThermodynamicLaw, ThermoStats


@dataclass
class SimState:
    """仿真状态"""
    generation: int
    step: int
    total_energy: float
    alive_count: int
    mean_energy: float
    max_energy: float
    min_energy: float


class BatchedAgents:
    """批量Agent管理器"""
    
    def __init__(
        self,
        n_agents: int,
        env_width: float = 100.0,
        env_height: float = 100.0,
        device: str = 'cuda:0',
        init_energy: float = 150.0
    ):
        self.n_agents = n_agents
        self.env_width = env_width
        self.env_height = env_height
        self.device = device
        
        # 创建状态
        self.state = create_agent_state(
            n_agents, env_width, env_height, device, init_energy
        )
        
        # 初始化alive_mask
        self.alive_mask = torch.ones(n_agents, device=device, dtype=torch.bool)
    
    def get_sensors(self, env: 'EnvironmentGPU') -> torch.Tensor:
        """获取传感器输入"""
        field_values = env.get_field_values(self.state.positions)
        energy_norm = torch.clamp(self.state.energies / 200.0, 0, 1)
        
        return torch.cat([
            field_values,
            energy_norm.unsqueeze(1)
        ], dim=1)
    
    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        """神经网络前向 (简化版)"""
        N = sensors.shape[0]
        
        # 简化: 随机输出 (实际应连接brain matrix)
        outputs = torch.zeros(N, 5, device=self.device)
        
        # 填充随机值 (临时)
        outputs[:, 0] = torch.randn(N, device=self.device) * 0.5  # permeability
        outputs[:, 1] = torch.randn(N, device=self.device)         # thrust_x
        outputs[:, 2] = torch.randn(N, device=self.device)         # thrust_y
        outputs[:, 3] = torch.randn(N, device=self.device).abs()   # signal
        outputs[:, 4] = torch.randn(N, device=self.device) * 0.5   # defense
        
        return outputs
    
    def step(self, outputs: torch.Tensor):
        """单步更新"""
        state = self.state
        
        # 激活函数
        permeabilities = torch.sigmoid(outputs[:, 0])
        thrust_x = torch.tanh(outputs[:, 1])
        thrust_y = torch.tanh(outputs[:, 2])
        signals = torch.relu(outputs[:, 3])
        defenses = torch.sigmoid(outputs[:, 4])
        
        # 速度 = 推力 * 渗透率
        velocities = torch.stack([thrust_x, thrust_y], dim=1) * \
                     permeabilities.unsqueeze(1) * 5.0
        
        # 位置更新
        new_positions = state.positions + velocities * 1.0
        new_positions[:, 0] = new_positions[:, 0] % self.env_width
        new_positions[:, 1] = new_positions[:, 1] % self.env_height
        
        # 朝向
        new_thetas = torch.atan2(velocities[:, 1], velocities[:, 0])
        
        # 更新状态
        state.positions = new_positions
        state.velocities = velocities
        state.thetas = new_thetas
        state.permeabilities = permeabilities
        state.defenses = defenses
        state.signals = signals


class EnvironmentGPU:
    """GPU环境 (使用fields模块)"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        device: str = 'cuda:0',
        energy_field_enabled: bool = True,
        impedance_field_enabled: bool = True,
        stigmergy_field_enabled: bool = True
    ):
        self.width = width
        self.height = height
        self.device = device
        
        self.energy_field_enabled = energy_field_enabled
        self.impedance_field_enabled = impedance_field_enabled
        self.stigmergy_field_enabled = stigmergy_field_enabled
        
        # 导入fields
        from ..fields import EnergyField, ImpedanceField, StigmergyField
        
        if energy_field_enabled:
            self.energy_field = EnergyField(width, height, device=device)
            self.epf_grad_x = None
            self.epf_grad_y = None
        
        if impedance_field_enabled:
            self.impedance_field = ImpedanceField(width, height, device=device)
            self.kif_grad_x = None
            self.kif_grad_y = None
        
        if stigmergy_field_enabled:
            self.stigmergy_field = StigmergyField(width, height, device=device)
            self.isf_grad_x = None
            self.isf_grad_y = None
    
    def step(self):
        """场更新"""
        if self.energy_field_enabled:
            self.energy_field.step()
            self.epf_grad_x, self.epf_grad_y = self.energy_field.compute_gradient()
        
        if self.impedance_field_enabled:
            self.impedance_field.step()  # 静态场
            self.kif_grad_x, self.kif_grad_y = self.impedance_field.compute_gradient()
        
        if self.stigmergy_field_enabled:
            self.stigmergy_field.step()
            self.isf_grad_x, self.isf_grad_y = self.stigmergy_field.compute_gradient()
    
    def get_field_values(self, positions: torch.Tensor) -> torch.Tensor:
        """批量获取场值"""
        N = positions.shape[0]
        results = []
        
        gx = (positions[:, 0]).long()
        gy = (positions[:, 1]).long()
        
        max_x = self.energy_field.grid_width - 1 if self.energy_field_enabled else 99
        max_y = self.energy_field.grid_height - 1 if self.energy_field_enabled else 99
        
        gx = torch.clamp(gx, 0, max_x)
        gy = torch.clamp(gy, 0, max_y)
        
        if self.energy_field_enabled:
            epf = self.energy_field.field
            results.extend([
                epf[gy, gx],
                self.epf_grad_x[gy, gx] if self.epf_grad_x is not None else torch.zeros(N, device=self.device),
                self.epf_grad_y[gy, gx] if self.epf_grad_y is not None else torch.zeros(N, device=self.device)
            ])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        if self.impedance_field_enabled:
            kif = self.impedance_field.field
            results.extend([
                kif[gy, gx],
                self.kif_grad_x[gy, gx] if self.kif_grad_x is not None else torch.zeros(N, device=self.device),
                self.kif_grad_y[gy, gx] if self.kif_grad_y is not None else torch.zeros(N, device=self.device)
            ])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        if self.stigmergy_field_enabled:
            isf = self.stigmergy_field.field
            results.extend([
                isf[gy, gx],
                self.isf_grad_x[gy, gx] if self.isf_grad_x is not None else torch.zeros(N, device=self.device),
                self.isf_grad_y[gy, gx] if self.isf_grad_y is not None else torch.zeros(N, device=self.device)
            ])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        return torch.stack(results, dim=1)


class Simulation:
    """
    统一仿真入口
    ============
    简化用户接口，一行代码启动仿真
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        env_width: float = 100.0,
        env_height: float = 100.0,
        lifespan: int = 1500,
        device: str = 'cuda:0',
        energy_field: bool = True,
        impedance_field: bool = True,
        stigmergy_field: bool = True
    ):
        self.n_agents = n_agents
        self.lifespan = lifespan
        self.device = device
        
        print(f"🚀 初始化仿真: {n_agents} agents, {lifespan} steps on {device}")
        
        # 环境
        self.env = EnvironmentGPU(
            width=env_width,
            height=env_height,
            device=device,
            energy_field_enabled=energy_field,
            impedance_field_enabled=impedance_field,
            stigmergy_field_enabled=stigmergy_field
        )
        
        # Agent
        self.agents = BatchedAgents(
            n_agents=n_agents,
            env_width=env_width,
            env_height=env_height,
            device=device
        )
        
        # 热力学
        self.thermo = ThermodynamicLaw(device=device)
        self.thermo.initialize(self.env, self.agents)
        
        # 状态
        self.generation = 0
        self.step_count = 0
        self.history: List[SimState] = []
        
        print(f"✅ 仿真就绪")
    
    def step(self) -> SimState:
        """执行单步"""
        # 1. 传感器
        sensors = self.agents.get_sensors(self.env)
        
        # 2. 前向传播
        outputs = self.agents.forward(sensors)
        
        # 3. Agent更新
        self.agents.step(outputs)
        
        # 4. 热力学
        stats, self.agents.alive_mask = self.thermo.apply(
            self.env, self.agents, self.agents.alive_mask
        )
        
        # 5. 场更新
        self.env.step()
        
        # 6. 状态记录
        is_stable, total_energy = self.thermo.check_stability(self.env, self.agents)
        
        state = SimState(
            generation=self.generation,
            step=self.step_count,
            total_energy=total_energy,
            alive_count=stats.alive_count,
            mean_energy=stats.mean_energy,
            max_energy=stats.max_energy,
            min_energy=stats.min_energy
        )
        self.history.append(state)
        
        self.step_count += 1
        return state
    
    def run(self, max_steps: Optional[int] = None, verbose: bool = True) -> List[SimState]:
        """运行仿真"""
        max_steps = max_steps or self.lifespan
        
        if verbose:
            print(f"\n🔄 运行 {max_steps} 步仿真...")
        
        start = time.perf_counter()
        
        for step in range(max_steps):
            self.step()
            
            if verbose and step % 100 == 0:
                state = self.history[-1]
                print(f"  Step {step}: {state.alive_count} alive, energy={state.total_energy:.0f}")
            
            # 早停
            if torch.sum(self.agents.alive_mask) == 0:
                if verbose:
                    print(f"  ⚠️ 全部死亡于 step {step}")
                break
        
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        if verbose:
            print(f"\n✅ 完成! 耗时: {elapsed:.2f}秒")
            print(f"   存活: {self.history[-1].alive_count}/{self.n_agents}")
        
        return self.history
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态 (用于可视化)"""
        return {
            'agent': to_cpu(self.agents.state),
            'epf': self.env.energy_field.field.cpu().numpy() if self.env.energy_field_enabled else None,
            'isf': self.env.stigmergy_field.field.cpu().numpy() if self.env.stigmergy_field_enabled else None,
            'kif': self.env.impedance_field.field.cpu().numpy() if self.env.impedance_field_enabled else None,
        }


# 统一导出
__all__ = ['Simulation', 'BatchedAgents', 'EnvironmentGPU', 'SimState']