#!/usr/bin/env python3
"""
v13.0 Neural Evolution - First Generation Selection
=====================================================
从"随机游走"到"神经接管"的完整进化闭环

三步流程:
1. 神经接管: 1000个异构大脑的GPU并行前向传播
2. 热力学淘汰: 1500-2000步纯能量筛选
3. 跨代遗传: 精英交叉变异 + ISF生态印记

运行:
    python scripts/test_neural_evolution_v13.py --generations 10 --steps 1500
"""

import argparse
import sys
import os
import time
import random
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.integrated_simulation import ThermodynamicLaw


# ============================================================================
# 神经网络前向传播 (异构大脑)
# ============================================================================

class HeterogeneousBrain:
    """
    异构大脑前向传播器
    ==================
    支持不同拓扑结构的神经网络并行计算
    """
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
    
    def forward(
        self, 
        sensors: torch.Tensor, 
        brain_weights: torch.Tensor,
        brain_masks: torch.Tensor,
        node_types: torch.Tensor
    ) -> torch.Tensor:
        """
        批量前向传播
        
        使用简化的全连接网络:
        Input(7) -> Hidden(16) -> Output(5)
        
        Args:
            sensors: [N, input_dim] 传感器输入
            brain_weights: [N, max_nodes, max_nodes] 大脑权重矩阵
            brain_masks: [N, max_nodes, max_nodes] 连接掩码
            node_types: [N, max_nodes] 节点类型
            
        Returns:
            Tensor [N, 5] 致动器输出
        """
        N = sensors.shape[0]
        input_dim = sensors.shape[1]
        
        # 固定网络结构
        hidden_dim = 16
        output_dim = 5
        
        # 从 brain_weights 中提取权重并应用掩码
        # W1: [N, input_dim, hidden_dim] - 使用 brain_weights 的前 input_dim 行和前 hidden_dim 列
        W1 = brain_weights[:, :input_dim, :hidden_dim]
        M1 = brain_masks[:, :input_dim, :hidden_dim]
        W1_masked = W1 * M1  # 应用连接掩码
        
        # 隐藏层
        hidden = torch.bmm(sensors.unsqueeze(1), W1_masked).squeeze(1)  # [N, hidden_dim]
        hidden = F.relu(hidden)
        
        # W2: [N, hidden_dim, output_dim]
        W2 = brain_weights[:, :hidden_dim, :output_dim]
        M2 = brain_masks[:, :hidden_dim, :output_dim]
        W2_masked = W2 * M2  # 应用输出层掩码
        
        # 输出层
        output = torch.bmm(hidden.unsqueeze(1), W2_masked).squeeze(1)  # [N, output_dim]
        
        return output


def create_random_brains(n_agents: int, device: str = 'cuda:0') -> tuple:
    """
    创建随机大脑矩阵 (初始化)
    
    Returns:
        brain_weights: [N, max_nodes, max_nodes]
        brain_masks: [N, max_nodes, max_nodes] 
        node_types: [N, max_nodes]
    """
    max_nodes = 16
    
    # 随机权重 [-1, 1]
    brain_weights = torch.randn(n_agents, max_nodes, max_nodes, device=device) * 0.5
    
    # 随机掩码 (30% 连接率)
    brain_masks = (torch.rand(n_agents, max_nodes, max_nodes, device=device) > 0.7).float()
    
    # 节点类型 (0=input, 1=hidden, 2=output)
    node_types = torch.randint(0, 3, (n_agents, max_nodes), device=device)
    
    return brain_weights, brain_masks, node_types


def crossover(parent1_weights, parent2_weights, device):
    """精英交叉
    
    Args:
        parent1_weights: [max_nodes, max_nodes]
        parent2_weights: [max_nodes, max_nodes]
    Returns:
        child_weights: [max_nodes, max_nodes]
    """
    max_nodes = parent1_weights.shape[0]
    
    # 随机选择交叉点
    crossover_mask = torch.rand(max_nodes, max_nodes, device=device) > 0.5
    
    child_weights = torch.where(crossover_mask, parent1_weights, parent2_weights)
    
    return child_weights


def mutate(brain_weights, mutation_rate=0.1, mutation_strength=0.5, device='cuda:0'):
    """变异
    
    Args:
        brain_weights: [max_nodes, max_nodes] 2D matrix
    Returns:
        mutated: [max_nodes, max_nodes]
    """
    max_nodes = brain_weights.shape[0]
    
    # 随机变异
    mutation_mask = torch.rand(max_nodes, max_nodes, device=device) < mutation_rate
    
    # 添加高斯噪声
    noise = torch.randn_like(brain_weights) * mutation_strength
    
    # 应用变异
    brain_weights = brain_weights + mutation_mask.float() * noise
    
    # 裁剪权重
    brain_weights = torch.clamp(brain_weights, -3, 3)
    
    return brain_weights


# ============================================================================
# 主进化流程
# ============================================================================

def run_neural_evolution(
    n_agents: int = 1000,
    n_steps: int = 1500,
    n_generations: int = 10,
    device: str = 'cuda:0',
    elite_ratio: float = 0.1,
    mutation_rate: float = 0.1,
    isf_decay: float = 0.5,
    visualize_every: int = 300
):
    """
    神经进化主循环
    
    Args:
        n_agents: Agent数量
        n_steps: 每代模拟步数
        n_generations: 进化代数
        device: 计算设备
        elite_ratio: 精英保留比例
        mutation_rate: 变异率
        isf_decay: ISF场衰减率 (0.5 = 保留50%)
        visualize_every: 可视化间隔
    """
    print("\n" + "="*70)
    print("NEURAL EVOLUTION v13.0 - First Generation Selection")
    print("="*70)
    print(f"Agents: {n_agents}")
    print(f"Steps per generation: {n_steps}")
    print(f"Generations: {n_generations}")
    print(f"Device: {device}")
    print(f"Elite ratio: {elite_ratio}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"ISF decay: {isf_decay}")
    print("="*70 + "\n")
    
    # 检查设备
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[!] CUDA not available, switching to CPU")
        device = 'cpu'
    
    # 初始化大脑
    brain_weights, brain_masks, node_types = create_random_brains(n_agents, device)
    print(f"[+] Initialized random brains: {brain_weights.shape}")
    
    # 记录进化历史
    history = {
        'generation': [],
        'max_energy': [],
        'mean_energy': [],
        'alive_count': [],
        'top_energy': []
    }
    
    # 进化主循环
    for gen in range(n_generations):
        print(f"\n{'='*70}")
        print(f"GENERATION {gen + 1}/{n_generations}")
        print(f"{'='*70}")
        
        # === 步骤1: 初始化环境 ===
        env = EnvironmentGPU(
            width=100.0,
            height=100.0,
            resolution=1.0,
            device=device,
            energy_field_enabled=True,
            impedance_field_enabled=True,
            stigmergy_field_enabled=True
        )
        
        # ISF 场衰减 (生态印记)
        if gen > 0:
            env.stigmergy_field.field *= isf_decay
            print(f"[+] ISF decay: {isf_decay * 100:.0f}% preserved from previous generation")
        
        # === 步骤2: 初始化Agent ===
        agents = BatchedAgents(
            n_agents=n_agents,
            env_width=100.0,
            env_height=100.0,
            device=device,
            init_energy=150.0
        )
        
        # 存储大脑到 agents
        agents.brain_weights = brain_weights
        agents.brain_masks = brain_masks
        agents.node_types = node_types
        
        # 初始化神经引擎
        brain_engine = HeterogeneousBrain(device=device)
        
        # 记录初始能量
        initial_energy = torch.sum(agents.state.energies).item()
        
        # === 步骤3: 模拟循环 (神经接管) ===
        print(f"\n[5/5] Running simulation with neural control...")
        print("-" * 70)
        
        gen_start = time.time()
        
        for step in range(n_steps):
            # 获取传感器输入
            sensors = agents.get_sensors(env)  # [N, input_dim]
            
            # 神经网络前向传播 (真实大脑!)
            try:
                brain_outputs = brain_engine.forward(
                    sensors, 
                    agents.brain_weights,
                    agents.brain_masks,
                    agents.node_types
                )
            except Exception as e:
                print(f"[DEBUG] sensors shape: {sensors.shape}")
                print(f"[DEBUG] brain_weights shape: {agents.brain_weights.shape}")
                print(f"[DEBUG] Error: {e}")
                raise
            
            # Agent 步进
            agents.step(brain_outputs, dt=1.0)
            
            # 热力学定律应用
            law = ThermodynamicLaw(device=device)
            alive_mask = agents.state.is_alive
            stats, new_alive_mask = law.apply(env, agents, alive_mask)
            agents.state.is_alive = new_alive_mask
            
            # 环境步进
            env.step()
            
            # 进度输出
            if (step + 1) % visualize_every == 0:
                alive_count = torch.sum(agents.state.is_alive).item()
                mean_e = torch.mean(agents.state.energies).item()
                max_e = torch.max(agents.state.energies).item()
                total_e = torch.sum(agents.state.energies).item()
                
                print(f"Step {step+1:4d}/{n_steps} | "
                      f"Alive: {alive_count:4d} | "
                      f"Mean E: {mean_e:6.2f} | "
                      f"Max E: {max_e:6.2f} | "
                      f"Total: {total_e:10.2f}")
        
        gen_time = time.time() - gen_start
        
        # === 步骤4: 热力学淘汰 - 读取结果 ===
        alive_mask = agents.state.is_alive
        energies = agents.state.energies  # [N]
        
        # 获取存活 Agent 的能量
        alive_energies = energies.clone()
        alive_energies[~alive_mask] = -1  # 死亡的标记为-1
        
        # 排序获取精英
        elite_count = int(n_agents * elite_ratio)
        
        # 按能量排序 (降序)
        sorted_indices = torch.argsort(alive_energies, descending=True)
        elite_indices = sorted_indices[:elite_count]
        
        # 记录统计
        max_energy = energies[alive_mask].max().item() if alive_mask.any() else 0
        mean_energy = energies[alive_mask].mean().item() if alive_mask.any() else 0
        alive_count = alive_mask.sum().item()
        
        history['generation'].append(gen + 1)
        history['max_energy'].append(max_energy)
        history['mean_energy'].append(mean_energy)
        history['alive_count'].append(alive_count)
        history['top_energy'].append(energies[elite_indices[0]].item())
        
        print(f"\n[+] Generation {gen + 1} complete in {gen_time:.1f}s")
        print(f"    Alive: {alive_count}/{n_agents}")
        print(f"    Mean energy: {mean_energy:.2f}")
        print(f"    Max energy: {max_energy:.2f}")
        print(f"    Top elite energy: {energies[elite_indices[0]]:.2f}")
        
        # === 步骤5: 跨代遗传 ===
        if gen < n_generations - 1:
            print(f"\n[*] Evolving to next generation...")
            
            # 获取精英大脑
            elite_weights = brain_weights[elite_indices]  # [elite, nodes, nodes]
            
            # 创建下一代大脑
            new_brain_weights = []
            
            for i in range(n_agents):
                # 选择两个精英父母
                p1_idx = random.randint(0, elite_count - 1)
                p2_idx = random.randint(0, elite_count - 1)
                
                parent1 = elite_weights[p1_idx].squeeze()  # 确保是2D
                parent2 = elite_weights[p2_idx].squeeze()
                
                # 交叉
                child = crossover(parent1, parent2, device)
                
                # 变异
                child = mutate(child, mutation_rate, 0.5, device)
                
                new_brain_weights.append(child)
            
            # 更新大脑
            brain_weights = torch.stack(new_brain_weights)
            
            # 保留精英的掩码
            brain_masks = brain_masks[elite_indices].repeat(n_agents // elite_count + 1, 1, 1)[:n_agents]
            node_types = node_types[elite_indices].repeat(n_agents // elite_count + 1, 1)[:n_agents]
            
            print(f"    [-] Created {n_agents} children from {elite_count} elites")
            print(f"    [-] Mutation rate: {mutation_rate}")
    
    # === 最终报告 ===
    print("\n" + "="*70)
    print("EVOLUTION COMPLETE")
    print("="*70)
    
    print("\nGeneration History:")
    print("-" * 50)
    print(f"{'Gen':>4} | {'Alive':>6} | {'Mean E':>10} | {'Max E':>10} | {'Top E':>10}")
    print("-" * 50)
    for i in range(len(history['generation'])):
        print(f"{history['generation'][i]:>4} | "
              f"{history['alive_count'][i]:>6} | "
              f"{history['mean_energy'][i]:>10.2f} | "
              f"{history['max_energy'][i]:>10.2f} | "
              f"{history['top_energy'][i]:>10.2f}")
    
    # 绘制进化曲线
    plot_evolution_history(history)
    
    return history


def plot_evolution_history(history: dict):
    """绘制进化历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    gens = history['generation']
    
    # 左图: 能量变化
    ax1 = axes[0]
    ax1.plot(gens, history['max_energy'], 'r-', label='Max Energy', linewidth=2)
    ax1.plot(gens, history['mean_energy'], 'b-', label='Mean Energy', linewidth=2)
    ax1.plot(gens, history['top_energy'], 'g--', label='Top Elite', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 存活数量
    ax2 = axes[1]
    ax2.bar(gens, history['alive_count'], color='purple', alpha=0.7)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Alive Count')
    ax2.set_title('Survival Rate')
    ax2.set_ylim(0, max(history['alive_count']) * 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evolution_history.png', dpi=150)
    print(f"\n[OK] Evolution chart saved to: evolution_history.png")


# ============================================================================
# 入口点
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Evolution v13.0')
    parser.add_argument('--agents', type=int, default=1000, help='Number of agents')
    parser.add_argument('--steps', type=int, default=1500, help='Steps per generation')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--elite-ratio', type=float, default=0.1, help='Elite selection ratio')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--isf-decay', type=float, default=0.5, help='ISF field decay (0.5 = 50% preserved)')
    parser.add_argument('--visualize-every', type=int, default=300, help='Progress output interval')
    
    args = parser.parse_args()
    
    history = run_neural_evolution(
        n_agents=args.agents,
        n_steps=args.steps,
        n_generations=args.generations,
        device=args.device,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        isf_decay=args.isf_decay,
        visualize_every=args.visualize_every
    )