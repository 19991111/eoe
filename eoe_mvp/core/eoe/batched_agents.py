"""
v13.0 Phase 2: 批量 Agent 引擎
==============================
GPU 加速的 Agent 批处理系统 - 100% VRAM 常驻

特性:
- 所有 Agent 状态存储在 GPU 张量中
- 批量传感器采样 (F.grid_sample)
- 批量神经网络前向传播 (矩阵乘法)
- 异构大脑掩码对齐

核心概念:
- StateTensor: 所有 Agent 的状态 (位置、能量、信号等)
- BrainMatrix: 掩码对齐的大脑权重矩阵
- BatchSampler: 批量传感器采样器
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# ============================================================================
# Agent 状态张量
# ============================================================================

@dataclass
class AgentState:
    """Agent 状态容器 (GPU 张量)"""
    positions: torch.Tensor      # [N, 2] (x, y)
    velocities: torch.Tensor     # [N, 2] (vx, vy)
    energies: torch.Tensor       # [N] 内部能量
    thetas: torch.Tensor         # [N] 朝向角
    permeabilities: torch.Tensor # [N] 渗透率 (0-1)
    defenses: torch.Tensor       # [N] 防御力 (0-1)
    signals: torch.Tensor        # [N] 信号强度 (0-1)
    is_alive: torch.Tensor       # [N] 存活标志
    
    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]


class BatchedAgents:
    """
    批量 Agent 管理器
    =================
    所有状态存储在 GPU 显存中，支持批量操作
    """
    
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
        
        print(f"[BatchedAgents] 初始化 {n_agents} 个 Agent on {device}")
        
        # 初始化状态张量 (全部 VRAM 常驻)
        self._init_states(init_energy)
        
        # 大脑矩阵 (延迟初始化)
        self.brain_matrix = None
        self.brain_masks = None
        self.node_counts = None
        
        # 性能统计
        self.step_times = []
    
    def _init_states(self, init_energy: float):
        """初始化所有 Agent 状态"""
        # 随机位置
        self.state = AgentState(
            positions = torch.rand(self.n_agents, 2, device=self.device) * \
                        torch.tensor([self.env_width, self.env_height], device=self.device),
            velocities = torch.zeros(self.n_agents, 2, device=self.device),
            energies = torch.ones(self.n_agents, device=self.device) * init_energy,
            thetas = torch.rand(self.n_agents, device=self.device) * 2 * np.pi,
            permeabilities = torch.ones(self.n_agents, device=self.device) * 0.5,
            defenses = torch.ones(self.n_agents, device=self.device) * 0.5,
            signals = torch.zeros(self.n_agents, device=self.device),
            is_alive = torch.ones(self.n_agents, device=self.device, dtype=torch.bool)
        )
        
        print(f"  ✅ 状态张量: {self.state.positions.shape}")
    
    def set_brains(self, genomes: List['OperatorGenome']):
        """
        设置大脑矩阵 (异构大脑掩码对齐)
        
        Args:
            genomes: Agent 的基因组列表
        """
        max_nodes = max(len(g.nodes) for g in genomes)
        max_edges = max(len(g.edges) for g in genomes)
        
        print(f"  构建大脑矩阵: max_nodes={max_nodes}, max_edges={max_edges}")
        
        # 大脑权重矩阵 [N, max_nodes, max_nodes]
        # 使用掩码处理异构拓扑
        self.brain_matrix = torch.zeros(
            self.n_agents, max_nodes, max_nodes,
            device=self.device, dtype=torch.float32
        )
        
        # 节点类型矩阵 [N, max_nodes]
        self.node_types = torch.zeros(
            self.n_agents, max_nodes,
            device=self.device, dtype=torch.long
        )
        
        # 掩码 [N, max_nodes, max_nodes] (1=有效连接)
        self.brain_masks = torch.zeros(
            self.n_agents, max_nodes, max_nodes,
            device=self.device, dtype=torch.bool
        )
        
        self.node_counts = []
        
        # 填充大脑矩阵
        for i, genome in enumerate(genomes):
            nodes = list(genome.nodes.values())
            node_ids = {n.node_id: idx for idx, n in enumerate(nodes)}
            
            self.node_counts.append(len(nodes))
            
            # 填充节点类型
            for j, node in enumerate(nodes):
                self.node_types[i, j] = node.node_type.value
            
            # 填充边权重
            for edge in genome.edges.values():
                if edge.source in node_ids and edge.target in node_ids:
                    src_idx = node_ids[edge.source]
                    tgt_idx = node_ids[edge.target]
                    self.brain_matrix[i, src_idx, tgt_idx] = edge.weight
                    self.brain_masks[i, src_idx, tgt_idx] = True
        
        print(f"  ✅ 大脑矩阵: {self.brain_matrix.shape}")
        print(f"  ✅ 掩码矩阵: {self.brain_masks.shape}")
    
    def step(
        self,
        brain_outputs: torch.Tensor,
        dt: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        单步更新所有 Agent
        
        Args:
            brain_outputs: [N, 5] (permeability, thrust_x, thrust_y, signal, defense)
            dt: 时间步长
            
        Returns:
            dict: 状态变化量
        """
        # 解码脑输出
        permeabilities = torch.sigmoid(brain_outputs[:, 0])  # [N]
        thrust_x = torch.tanh(brain_outputs[:, 1])           # [N]
        thrust_y = torch.tanh(brain_outputs[:, 2])           # [N]
        signals = torch.relu(brain_outputs[:, 3])            # [N]
        defenses = torch.sigmoid(brain_outputs[:, 4])        # [N]
        
        # 更新状态
        # 速度 = 推力 * 渗透率
        new_velocities = torch.stack([thrust_x, thrust_y], dim=1) * \
                         permeabilities.unsqueeze(1) * 5.0
        
        # 位置 = 旧位置 + 速度 * dt
        new_positions = self.state.positions + new_velocities * dt
        
        # 环形世界边界
        new_positions[:, 0] = new_positions[:, 0] % self.env_width
        new_positions[:, 1] = new_positions[:, 1] % self.env_height
        
        # 朝向 = 速度方向
        new_thetas = torch.atan2(new_velocities[:, 1], new_velocities[:, 0])
        
        # 能量代谢 (简化)
        energy_cost = (permeabilities * 0.01 + 
                      torch.sum(torch.abs(new_velocities), dim=1) * 0.01 +
                      signals * 0.1)
        new_energies = self.state.energies - energy_cost * dt
        
        # 死亡检查
        new_is_alive = new_energies > 0
        
        # 更新状态
        self.state.positions = new_positions
        self.state.velocities = new_velocities
        self.state.energies = new_energies
        self.state.thetas = new_thetas
        self.state.permeabilities = permeabilities
        self.state.defenses = defenses
        self.state.signals = signals
        self.state.is_alive = new_is_alive
        
        return {
            'positions': new_positions,
            'energies': new_energies,
            'signals': signals
        }
    
    def get_sensors(
        self,
        env: 'EnvironmentGPU'
    ) -> torch.Tensor:
        """
        批量获取传感器输入
        
        Args:
            env: GPU 环境
        Returns:
            Tensor [N, 13] - 传感器值
        """
        # 获取场值 (批量)
        field_values = env.get_field_values(self.state.positions)  # [N, 6-9]
        
        # 内部能量感知
        energy_norm = torch.clamp(self.state.energies / 200.0, 0, 1)  # [N]
        
        # 组合传感器
        sensors = torch.cat([
            field_values,      # [N, 6-9]
            energy_norm.unsqueeze(1)  # [N, 1]
        ], dim=1)
        
        return sensors  # [N, 7-10]
    
    def forward_brains(self, sensors: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播
        
        Args:
            sensors: [N, input_dim] 传感器输入
        Returns:
            Tensor [N, 5] 脑输出
        """
        if self.brain_matrix is None:
            # 默认大脑 (无学习)
            return torch.zeros(sensors.shape[0], 5, device=self.device)
        
        N = sensors.shape[0]
        
        # 方法1: 简单线性前向 (示例)
        # 实际应实现完整的 CNN/全连接网络
        
        # 展平输入
        inputs = sensors  # [N, input]
        
        # 批量矩阵乘法 (应用大脑权重)
        # [N, input] @ [input, hidden] -> [N, hidden]
        hidden = torch.matmul(inputs, self.brain_matrix[:, :inputs.shape[1], :32].transpose(1, 2))
        
        # 应用掩码
        mask = self.brain_masks[:, :inputs.shape[1], :32]
        hidden = hidden * mask.float()
        
        # ReLU 激活
        hidden = torch.relu(hidden)
        
        # 输出层 (简化)
        output = torch.matmul(hidden, self.brain_matrix[:, :32, :5].transpose(1, 2))
        
        # 取对角线元素作为输出
        output = output[:, torch.arange(min(N, output.shape[1])), torch.arange(5)[:min(N, output.shape[1])]]
        
        if output.shape[1] < 5:
            # 填充
            padding = torch.zeros(N, 5 - output.shape[1], device=self.device)
            output = torch.cat([output, padding], dim=1)
        
        return output  # [N, 5]


def benchmark_batched_agents(n_agents: int = 1000, n_steps: int = 100):
    """批量 Agent 性能基准测试"""
    import time
    
    print("\n" + "="*60)
    print(f"🎯 BatchedAgents 性能基准测试 ({n_agents} agents)")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建批量 Agent
    agents = BatchedAgents(
        n_agents=n_agents,
        env_width=100,
        env_height=100,
        device=device
    )
    
    # 创建环境
    from core.eoe.environment_gpu import EnvironmentGPU
    env = EnvironmentGPU(
        width=100, height=100,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True
    )
    
    # 预热
    print("\n预热 (10步)...")
    for _ in range(10):
        sensors = agents.get_sensors(env)
        outputs = agents.forward_brains(sensors)
        agents.step(outputs)
        env.step()
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 基准测试
    print(f"\n运行 {n_steps} 步...")
    start = time.perf_counter()
    
    for _ in range(n_steps):
        # 1. 批量传感器采样
        sensors = agents.get_sensors(env)  # [N, 13]
        
        # 2. 批量前向传播
        outputs = agents.forward_brains(sensors)  # [N, 5]
        
        # 3. 批量状态更新
        agents.step(outputs)
        
        # 4. 场更新
        env.step()
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    print(f"\n📊 结果:")
    print(f"  总耗时: {elapsed:.4f}s")
    print(f"  平均每步: {elapsed/n_steps*1000:.3f}ms")
    print(f"  吞吐量: {n_steps*n_agents/elapsed:.0f} agent-steps/sec")
    
    # 对比串行版本
    print("\n" + "="*60)
    print("📈 对比串行版本 (估算)")
    print("="*60)
    
    # 基于之前的测试数据
    sensor_per_agent = 0.024  # ms
    brain_per_agent = 0.084   # ms
    cpu_step = 3.2  # ms
    
    serial_time = n_steps * (n_agents * (sensor_per_agent + brain_per_agent) / 1000 + cpu_step / 1000)
    
    print(f"  串行估算: {serial_time:.2f}s")
    print(f"  批量GPU:  {elapsed:.2f}s")
    print(f"  🚀 加速比: {serial_time/elapsed:.1f}x")
    
    return agents, env


if __name__ == "__main__":
    benchmark_batched_agents(1000, 100)