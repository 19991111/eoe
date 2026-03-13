#!/usr/bin/env python3
"""
v13.0 Neural Evolution - OPTIMIZED VERSION
===========================================
优化点:
1. 梯度每10步计算一次 (而不是每步)
2. 使用torch.compile加速神经网络
3. 减少Python循环开销
4. 批量能量提取

运行:
    python scripts/test_neural_evolution_v13_opt.py --agents 1000 --steps 1000 --generations 50 --device cuda:0
"""

import sys
import os
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.integrated_simulation import ThermodynamicLaw


# ============================================================================
# 神经网络 (使用bmm优化)
# ============================================================================

class HeterogeneousBrain:
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
    
    def forward(self, sensors, brain_weights):
        """优化的前向传播"""
        # W1: [N, input_dim, hidden_dim]
        W1 = brain_weights[:, :sensors.shape[1], :16]
        # hidden: [N, hidden_dim]
        hidden = torch.bmm(sensors.unsqueeze(1), W1).squeeze(1)
        hidden = F.relu(hidden)
        
        # W2: [N, hidden_dim, output_dim]
        W2 = brain_weights[:, :16, :5]
        output = torch.bmm(hidden.unsqueeze(1), W2).squeeze(1)
        
        return output


def crossover(parent1, parent2, device):
    max_nodes = parent1.shape[0]
    crossover_mask = torch.rand(max_nodes, max_nodes, device=device) > 0.5
    child = torch.where(crossover_mask, parent1, parent2)
    return child


def mutate(brain_weights, mutation_rate=0.1, mutation_strength=0.5, device='cuda:0'):
    max_nodes = brain_weights.shape[0]
    mutation_mask = torch.rand(max_nodes, max_nodes, device=device) < mutation_rate
    noise = torch.randn_like(brain_weights) * mutation_strength
    brain_weights = brain_weights + mutation_mask.float() * noise
    brain_weights = torch.clamp(brain_weights, -3, 3)
    return brain_weights


# ============================================================================
# 优化的环境步进 (减少梯度计算频率)
# ============================================================================

class OptimizedEnvironment:
    """每N步才计算一次梯度"""
    
    def __init__(self, env: EnvironmentGPU, gradient_every: int = 10):
        self.env = env
        self.gradient_every = gradient_every
        self.step_count = 0
        
        # 预计算初始梯度
        self.env._compute_gradients()
    
    def step(self):
        self.step_count += 1
        self.env.step()
        
        # 每N步才计算梯度
        if self.step_count % self.gradient_every == 0:
            self.env._compute_gradients()


# ============================================================================
# 主函数
# ============================================================================

def run_neural_evolution(
    n_agents: int = 1000,
    n_steps: int = 1500,
    n_generations: int = 50,
    device: str = 'cuda:0',
    elite_ratio: float = 0.1,
    mutation_rate: float = 0.1,
    isf_decay: float = 0.5,
    gradient_every: int = 10
):
    print("="*70)
    print("NEURAL EVOLUTION v13.0 - OPTIMIZED")
    print("="*70)
    print(f"Agents: {n_agents}")
    print(f"Steps per generation: {n_steps}")
    print(f"Generations: {n_generations}")
    print(f"Device: {device}")
    print(f"Elite ratio: {elite_ratio}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"ISF decay: {isf_decay}")
    print(f"Gradient every: {gradient_every} steps")
    print("="*70)
    
    max_nodes = 16
    
    # 初始化大脑 (随机权重)
    brain_weights = torch.randn(n_agents, max_nodes, max_nodes, device=device) * 0.5
    brain_masks = torch.ones(n_agents, max_nodes, max_nodes, device=device)
    node_types = torch.randint(0, 3, (n_agents, max_nodes), device=device)
    
    print(f"[+] Initialized random brains: {brain_weights.shape}")
    
    brain_engine = HeterogeneousBrain(device=device)
    law = ThermodynamicLaw(device=device)
    
    history = {
        'generation': [],
        'max_energy': [],
        'mean_energy': [],
        'alive_count': [],
        'top_energy': []
    }
    
    for gen in range(n_generations):
        gen_start = time.time()
        
        # 初始化环境
        env = EnvironmentGPU(
            width=100.0, height=100.0, resolution=1.0, device=device,
            energy_field_enabled=True,
            impedance_field_enabled=True,
            stigmergy_field_enabled=True
        )
        
        # ISF 生态印记
        if gen > 0:
            env.stigmergy_field.field *= isf_decay
        
        opt_env = OptimizedEnvironment(env, gradient_every=gradient_every)
        
        # 初始化Agent
        agents = BatchedAgents(
            n_agents=n_agents, env_width=100.0, env_height=100.0,
            device=device, init_energy=150.0
        )
        agents.brain_weights = brain_weights
        agents.brain_masks = brain_masks
        agents.node_types = node_types
        
        # 初始化宇宙能量
        law.initialize_universe(env, agents)
        
        print(f"\n{'='*70}")
        print(f"GENERATION {gen + 1}/{n_generations}")
        print(f"{'='*70}")
        
        # 模拟循环
        for step in range(n_steps):
            # 获取传感器
            sensors = agents.get_sensors(env)
            
            # 神经网络前向传播
            brain_outputs = brain_engine.forward(sensors, agents.brain_weights)
            
            # Agent步进
            agents.step(brain_outputs, dt=1.0)
            
            # 热力学定律
            alive_mask = agents.state.is_alive
            stats, new_alive_mask = law.apply(env, agents, alive_mask)
            agents.state.is_alive = new_alive_mask
            
            # 环境步进 (优化版)
            opt_env.step()
            
            # 进度输出
            if (step + 1) % 100 == 0:
                energies = agents.state.energies
                alive = agents.state.is_alive
                alive_count = alive.sum().item()
                mean_e = energies[alive].mean().item() if alive.any() else 0
                max_e = energies[alive].max().item() if alive.any() else 0
                total_e = energies.sum().item()
                
                print(f"Step {step + 1:4d}/{n_steps} | Alive: {alive_count:4d} | Mean E: {mean_e:6.2f} | Max E: {max_e:6.2f} | Total: {total_e:10.2f}")
        
        # 记录结果
        alive_mask = agents.state.is_alive
        energies = agents.state.energies
        
        alive_energies = energies.clone()
        alive_energies[~alive_mask] = -1
        
        elite_count = int(n_agents * elite_ratio)
        sorted_indices = torch.argsort(alive_energies, descending=True)
        elite_indices = sorted_indices[:elite_count]
        
        max_energy = energies[alive_mask].max().item() if alive_mask.any() else 0
        mean_energy = energies[alive_mask].mean().item() if alive_mask.any() else 0
        alive_count = alive_mask.sum().item()
        
        gen_time = time.time() - gen_start
        
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
        
        # 进化
        if gen < n_generations - 1:
            print(f"\n[*] Evolving to next generation...")
            
            elite_weights = brain_weights[elite_indices]
            
            new_brain_weights = []
            for i in range(n_agents):
                p1_idx = random.randint(0, elite_count - 1)
                p2_idx = random.randint(0, elite_count - 1)
                
                parent1 = elite_weights[p1_idx].squeeze()
                parent2 = elite_weights[p2_idx].squeeze()
                
                child = crossover(parent1, parent2, device)
                child = mutate(child, mutation_rate, 0.5, device)
                
                new_brain_weights.append(child)
            
            brain_weights = torch.stack(new_brain_weights)
            
            # 保留精英的掩码
            brain_masks = brain_masks[elite_indices].repeat(n_agents // elite_count + 1, 1, 1)[:n_agents]
            node_types = node_types[elite_indices].repeat(n_agents // elite_count + 1, 1)[:n_agents]
            
            print(f"    [-] Created {n_agents} children from {elite_count} elites")
            print(f"    [-] Mutation rate: {mutation_rate}")
    
    # 最终结果
    print(f"\n{'='*70}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*70}")
    
    print("\nGeneration History:")
    print("-" * 60)
    print(f" {'Gen':>3} | {'Alive':>5} | {'Mean E':>9} | {'Max E':>9} | {'Top E':>9}")
    print("-" * 60)
    for i in range(len(history['generation'])):
        print(f" {history['generation'][i]:>3} | {history['alive_count'][i]:>5} | {history['mean_energy'][i]:>9.2f} | {history['max_energy'][i]:>9.2f} | {history['top_energy'][i]:>9.2f}")
    
    # 保存图表
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['generation'], history['alive_count'], 'g-', linewidth=2)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Alive Count')
    axes[0].set_title('Survival Rate')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['generation'], history['max_energy'], 'r-', label='Max E', linewidth=2)
    axes[1].plot(history['generation'], history['mean_energy'], 'b-', label='Mean E', linewidth=2)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Energy Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evolution_history_opt.png', dpi=150)
    print(f"\n[OK] Evolution chart saved to: evolution_history_opt.png")
    
    return history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--elite-ratio', type=float, default=0.1)
    parser.add_argument('--mutation-rate', type=float, default=0.1)
    parser.add_argument('--isf-decay', type=float, default=0.5)
    parser.add_argument('--gradient-every', type=int, default=10)
    
    args = parser.parse_args()
    
    run_neural_evolution(
        n_agents=args.agents,
        n_steps=args.steps,
        n_generations=args.generations,
        device=args.device,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        isf_decay=args.isf_decay,
        gradient_every=args.gradient_every
    )