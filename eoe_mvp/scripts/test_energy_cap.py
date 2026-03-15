#!/usr/bin/env python3
"""
能量上限实验 - 测试是否能涌现更复杂结构
==========================================
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType
from scripts.energy_cap_mod import EnergyCapPatcher


class DynamicEnergyField:
    """动态能量场 - 从 run_harsh_environment.py 复制"""
    
    def __init__(
        self, width: float = 100.0, height: float = 100.0,
        resolution: float = 1.0, device: str = 'cpu',
        n_patches: int = 5, patch_strength: float = 30.0,
        patch_sigma: float = 8.0, movement_speed: float = 0.5,
        global_decay: float = 0.95
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        self.n_patches = n_patches
        self.patch_strength = patch_strength
        self.patch_sigma = patch_sigma
        self.movement_speed = movement_speed
        self.global_decay = global_decay
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        if device == 'cpu':
            self.field = torch.zeros(1, 1, self.grid_height, self.grid_width, dtype=torch.float32)
        else:
            self.field = torch.zeros(1, 1, self.grid_height, self.grid_width, device=device, dtype=torch.float32)
        
        self._init_patches()
        self._inject_gaussian_patches()
    
    def _init_patches(self):
        if self.device == 'cpu':
            self.positions = torch.zeros(self.n_patches, 2)
            self.velocities = torch.zeros(self.n_patches, 2)
        else:
            self.positions = torch.zeros(self.n_patches, 2, device=self.device)
            self.velocities = torch.zeros(self.n_patches, 2, device=self.device)
        
        margin = 15.0
        for i in range(self.n_patches):
            self.positions[i, 0] = margin + torch.rand(1, device=self.device) * (self.width - 2*margin)
            self.positions[i, 1] = margin + torch.rand(1, device=self.device) * (self.height - 2*margin)
            self.velocities[i, 0] = (torch.rand(1, device=self.device) - 0.5) * self.movement_speed
            self.velocities[i, 1] = (torch.rand(1, device=self.device) - 0.5) * self.movement_speed
    
    def _inject_gaussian_patches(self):
        for i in range(self.n_patches):
            px, py = int(self.positions[i, 0].item()), int(self.positions[i, 1].item())
            sigma = int(self.patch_sigma)
            for dy in range(-sigma, sigma + 1):
                for dx in range(-sigma, sigma + 1):
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist <= sigma and 0 <= px+dx < self.grid_width and 0 <= py+dy < self.grid_height:
                        strength = self.patch_strength * (1 - dist / sigma)
                        self.field[0, 0, py+dy, px+dx] += strength
    
    def step(self):
        self.positions += self.velocities
        margin = 10.0
        for i in range(self.n_patches):
            if self.positions[i, 0] < margin or self.positions[i, 0] > self.width - margin:
                self.velocities[i, 0] *= -1
                self.positions[i, 0] = torch.clamp(self.positions[i, 0], margin, self.width - margin)
            if self.positions[i, 1] < margin or self.positions[i, 1] > self.height - margin:
                self.velocities[i, 1] *= -1
                self.positions[i, 1] = torch.clamp(self.positions[i, 1], margin, self.height - margin)
        
        if torch.rand(1).item() < 0.05:
            idx = torch.randint(0, self.n_patches, (1,)).item()
            self.velocities[idx, 0] += (torch.rand(1, device=self.device).item() - 0.5) * self.movement_speed
            self.velocities[idx, 1] += (torch.rand(1, device=self.device).item() - 0.5) * self.movement_speed
        
        self.field *= self.global_decay
        self._inject_gaussian_patches()


def create_cambrian_genome(min_nodes: int = None):
    g = OperatorGenome()
    if min_nodes is not None:
        n_nodes = min_nodes + np.random.randint(0, 3)
    else:
        n_nodes = np.random.randint(3, 7)
    
    types = [NodeType.SENSOR]
    for _ in range(n_nodes - 2):
        types.append(np.random.choice([NodeType.ADD, NodeType.MULTIPLY, NodeType.THRESHOLD, NodeType.DELAY]))
    types.append(NodeType.ACTUATOR)
    
    for j, t in enumerate(types):
        g.add_node(Node(j, t))
    
    for src in range(len(types) - 1):
        if np.random.random() < 0.7:
            g.add_edge(src, src+1, 0.001)
    
    g.energy = 80.0
    return g


def run_experiment(
    n_steps: int = 30000,
    initial_population: int = 500,
    device: str = 'cuda:0'
):
    print("="*70)
    print("🔬 能量上限实验 - 观测复杂结构涌现")
    print("="*70)
    
    # 配置
    config = PoolConfig()
    config.HEBBIAN_ENABLED = True
    config.SUPERNODE_ENABLED = True
    config.SUPERNODE_DETECTION_FREQUENCY = 500
    
    # 环境
    env = EnvironmentGPU(
        width=100.0, height=100.0, resolution=1.0,
        device=device,
        energy_field_enabled=True,
        seasons_enabled=True,
        season_length=3000
    )
    
    # 动态能量场（使用正确的类）
    env.energy_field = DynamicEnergyField(
        width=100.0, height=100.0,
        resolution=1.0, device=device,
        n_patches=5,
        patch_strength=40.0,
        patch_sigma=8.0,
        movement_speed=0.8,
        global_decay=0.95
    )
    env.energy_field_enabled = True
    
    # 代谢参数（更低）
    config.BASE_METABOLISM = 0.15
    config.MOTION_COST = 0.02
    
    # Agent池
    agents = BatchedAgents(
        initial_population=initial_population,
        max_agents=10000,
        env_width=100.0, env_height=100.0,
        device=device,
        init_energy=200.0,
        config=config,
        env=env
    )
    
    # 寒武纪初始化（从3节点开始）
    for i in range(initial_population):
        g = create_cambrian_genome()
        g.energy = 200.0  # 更高初始能量
        agents.genomes[i] = g
        agents.state.node_counts[i] = len(g.nodes)
    
    print(f"  🧬 寒武纪初始化: {initial_population} agents")
    
    # 能量上限补丁（仅限速，不加伤害）
    energy_patcher = EnergyCapPatcher(
        base_max_energy=1000.0,  # 更高能量上限
        energy_per_node=150.0,   # 每节点+150
        penalty_factor=0.8,      # 吃饱后80%速度（非常温和）
        enable_saturation_damage=False  # 禁用饱和伤害
    )
    
    # 渐进式压力
    WARMUP_STEPS = 4000
    PRESSURE_INTERVAL = 3000
    PRESSURE_RATE = 0.95
    
    print(f"\n🔥 参数:")
    print(f"   能量上限: {energy_patcher.base_max_energy} + 节点数 × {energy_patcher.energy_per_node}")
    print(f"   吃饱减益: {energy_patcher.penalty_factor}")
    print(f"   预热期: {WARMUP_STEPS} 步")
    print(f"   压力间隔: {PRESSURE_INTERVAL} 步")
    
    print(f"\n开始运行 {n_steps} 步...")
    
    current_strength = 35.0
    pressure_level = 0
    last_pressure = 0
    
    for step in range(n_steps):
        # Step
        result = agents.step(env=env, dt=0.1)
        
        # 更新能量场
        env.energy_field.step()
        
        # 应用能量上限补丁
        energy_patcher.patch(agents, env, device)
        
        # 奖励
        if step % 20 == 0:
            batch = agents.get_active_batch()
            if batch.n > 0:
                reward_mask = torch.rand(batch.n, device=device) < 0.1
                reward_indices = batch.indices[reward_mask]
                agents.state.energies[reward_indices] += 10.0
        
        # 渐进压力
        if step > WARMUP_STEPS and step - last_pressure >= PRESSURE_INTERVAL:
            current_strength *= PRESSURE_RATE
            env.energy_field.patch_strength = current_strength
            pressure_level += 1
            last_pressure = step
            print(f"\n🔥 [Step {step}] 压力升级 #{pressure_level}, 能量强度: {current_strength:.1f}")
        
        # 报告
        if step % 2500 == 0 and step > 0:
            batch = agents.get_active_batch()
            n_alive = result['n_alive']
            
            if n_alive == 0:
                print(f"\n⚠️ [Step {step}] 所有个体已死亡!")
                break
            
            avg_nodes = agents.state.node_counts[agents.alive_mask].float().mean().item()
            avg_energy = batch.energies.mean().item()
            max_energy = batch.energies.max().item()
            
            # 节点分布
            node_counts = agents.state.node_counts[agents.alive_mask].cpu().numpy()
            unique, counts = np.unique(node_counts, return_counts=True)
            
            # 复杂结构
            complex_count = (node_counts >= 5).sum()
            very_complex = (node_counts >= 6).sum()
            max_nodes = node_counts.max()
            
            print(f"\n{'='*60}")
            print(f"Step {step} | 存活: {n_alive}")
            print(f"平均节点: {avg_nodes:.1f} | 最大节点: {max_nodes}")
            print(f"平均能量: {avg_energy:.1f} | 最大能量: {max_energy:.1f}")
            print(f"复杂(≥5): {complex_count} ({complex_count/len(node_counts)*100:.1f}%)")
            print(f"极复杂(≥6): {very_complex} ({very_complex/len(node_counts)*100:.1f}%)")
            
            # 节点分布
            print("节点分布:")
            for n, c in sorted(zip(unique, counts)):
                pct = c / len(node_counts) * 100
                bar = '█' * int(pct / 2)
                print(f"  {n}节点: {c:4d} ({pct:5.1f}%) {bar}")
    
    # 最终统计
    print("\n" + "="*70)
    print("📊 最终统计")
    print("="*70)
    
    batch = agents.get_active_batch()
    n_alive = batch.n
    
    avg_nodes = agents.state.node_counts[agents.alive_mask].float().mean().item()
    avg_energy = batch.energies.mean().item()
    
    print(f"存活: {n_alive}")
    print(f"平均能量: {avg_energy:.1f}")
    print(f"平均节点: {avg_nodes:.1f}")
    
    # 节点分布
    node_counts = agents.state.node_counts[agents.alive_mask].cpu().numpy()
    unique, counts = np.unique(node_counts, return_counts=True)
    
    print("\n节点分布:")
    for n, c in sorted(zip(unique, counts)):
        pct = c / len(node_counts) * 100
        print(f"  {n}节点: {c} ({pct:.1f}%)")
    
    # 复杂结构
    complex_count = (node_counts >= 5).sum()
    very_complex = (node_counts >= 6).sum()
    ultra_complex = (node_counts >= 8).sum()
    
    print(f"\n🔬 复杂结构 (≥5节点): {complex_count} ({complex_count/len(node_counts)*100:.1f}%)")
    print(f"🔬 极复杂 (≥6节点): {very_complex} ({very_complex/len(node_counts)*100:.1f}%)")
    print(f"🔬 超复杂 (≥8节点): {ultra_complex} ({ultra_complex/len(node_counts)*100:.1f}%)")
    
    # Top 10 冠军
    print("\n🏆 Top 10 能量冠军:")
    top_indices = torch.argsort(batch.energies, descending=True)[:10]
    
    for rank, idx in enumerate(top_indices):
        agent_id = batch.indices[idx].item()
        nc = agents.state.node_counts[agent_id].item()
        en = batch.energies[idx].item()
        
        genome = agents.genomes.get(agent_id)
        n_edges = len(genome.edges) if genome else 0
        
        print(f"#{rank+1:2d} | ID:{agent_id:4d} | 能量:{en:8.1f} | 节点:{nc} | 边:{n_edges}")
    
    # Top 10 复杂个体
    print("\n🧠 Top 10 复杂个体:")
    top_complex = torch.argsort(agents.state.node_counts[agents.alive_mask], descending=True)[:10]
    
    for rank, idx in enumerate(top_complex):
        agent_id = batch.indices[idx].item()
        nc = agents.state.node_counts[agent_id].item()
        en = batch.energies[idx].item()
        
        genome = agents.genomes.get(agent_id)
        n_edges = len(genome.edges) if genome else 0
        
        print(f"#{rank+1:2d} | ID:{agent_id:4d} | 节点:{nc} | 能量:{en:8.1f} | 边:{n_edges}")
    
    # 能量补丁统计
    stats = energy_patcher.get_stats()
    print(f"\n📈 能量补丁统计:")
    print(f"   平均吃饱个体: {stats['avg_saturated']:.1f}")
    
    return {
        'avg_nodes': avg_nodes,
        'complex_pct': complex_count/len(node_counts)*100,
        'very_complex_pct': very_complex/len(node_counts)*100,
        'ultra_complex_pct': ultra_complex/len(node_counts)*100,
        'max_nodes': int(node_counts.max()),
    }


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    result = run_experiment(n_steps=n_steps, device=device)