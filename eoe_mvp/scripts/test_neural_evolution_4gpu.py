#!/usr/bin/env python3
"""
v13.0 Neural Evolution - 4-GPU Parallel Run
============================================
在4张GPU上并行运行100代进化

运行:
    python scripts/test_neural_evolution_4gpu.py
"""

import sys
import os
import time
import random
from pathlib import Path
import subprocess
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.integrated_simulation import ThermodynamicLaw


# ============================================================================
# 神经网络前向传播
# ============================================================================

class HeterogeneousBrain:
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
    
    def forward(self, sensors, brain_weights, brain_masks, node_types):
        N = sensors.shape[0]
        input_dim = sensors.shape[1]
        
        hidden_dim = 16
        output_dim = 5
        
        W1 = brain_weights[:, :input_dim, :hidden_dim]
        hidden = torch.bmm(sensors.unsqueeze(1), W1).squeeze(1)
        hidden = F.relu(hidden)
        
        W2 = brain_weights[:, :hidden_dim, :output_dim]
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
# 单GPU运行函数
# ============================================================================

def run_evolution_on_gpu(
    gpu_id: int,
    n_agents: int = 500,
    n_steps: int = 1500,
    n_generations: int = 100,
    elite_ratio: float = 0.1,
    mutation_rate: float = 0.1,
    isf_decay: float = 0.5,
    result_queue=None
):
    """在单个GPU上运行进化"""
    device = f'cuda:{gpu_id}'
    print(f"[GPU {gpu_id}] Starting evolution on {device}")
    
    # 固定随机种子
    random.seed(42 + gpu_id)
    torch.manual_seed(42 + gpu_id)
    
    # 初始化大脑
    max_nodes = 16
    brain_weights = torch.randn(n_agents, max_nodes, max_nodes, device=device) * 0.5
    brain_masks = torch.ones(n_agents, max_nodes, max_nodes, device=device)
    node_types = torch.randint(0, 3, (n_agents, max_nodes), device=device)
    
    history = {
        'generation': [],
        'max_energy': [],
        'mean_energy': [],
        'alive_count': [],
        'top_energy': []
    }
    
    brain_engine = HeterogeneousBrain(device=device)
    
    for gen in range(n_generations):
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
        
        # 初始化Agent
        agents = BatchedAgents(
            n_agents=n_agents, env_width=100.0, env_height=100.0,
            device=device, init_energy=150.0
        )
        agents.brain_weights = brain_weights
        agents.brain_masks = brain_masks
        agents.node_types = node_types
        
        law = ThermodynamicLaw(device=device)
        
        # 模拟循环
        for step in range(n_steps):
            sensors = agents.get_sensors(env)
            brain_outputs = brain_engine.forward(sensors, agents.brain_weights, agents.brain_masks, agents.node_types)
            agents.step(brain_outputs, dt=1.0)
            
            alive_mask = agents.state.is_alive
            stats, new_alive_mask = law.apply(env, agents, alive_mask)
            agents.state.is_alive = new_alive_mask
            
            env.step()
        
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
        
        history['generation'].append(gen + 1)
        history['max_energy'].append(max_energy)
        history['mean_energy'].append(mean_energy)
        history['alive_count'].append(alive_count)
        history['top_energy'].append(energies[elite_indices[0]].item())
        
        if (gen + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] Gen {gen+1:3d} | Alive: {alive_count:3d}/{n_agents} | Mean E: {mean_energy:6.2f} | Max E: {max_energy:6.2f}")
        
        # 进化
        if gen < n_generations - 1:
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
    
    # 保存最佳大脑
    best_idx = sorted_indices[0]
    best_brain = brain_weights[best_idx].cpu().numpy()
    
    print(f"[GPU {gpu_id}] Complete! Best energy: {history['top_energy'][-1]:.2f}")
    
    if result_queue:
        result_queue.put({
            'gpu_id': gpu_id,
            'history': history,
            'best_brain': best_brain,
            'best_energy': history['top_energy'][-1]
        })
    
    return history, best_brain


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("NEURAL EVOLUTION v13.0 - 4-GPU Parallel Run (100 Generations)")
    print("="*70)
    
    n_agents = 500      # 每GPU agent数量
    n_steps = 1500      # 每代步数
    n_generations = 100 # 代数
    elite_ratio = 0.1   # 精英比例
    mutation_rate = 0.1 # 变异率
    
    result_queue = []
    threads = []
    
    start_time = time.time()
    
    # 启动4个GPU线程
    for gpu_id in range(4):
        t = threading.Thread(
            target=lambda gid: run_evolution_on_gpu(
                gpu_id=gid,
                n_agents=n_agents,
                n_steps=n_steps,
                n_generations=n_generations,
                elite_ratio=elite_ratio,
                mutation_rate=mutation_rate,
                result_queue=result_queue
            ),
            args=(gpu_id,)
        )
        t.start()
        threads.append(t)
    
    # 等待完成
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("ALL 4 GPUs COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time:.1f} seconds")
    
    # 收集结果
    results = result_queue
    results.sort(key=lambda x: x['best_energy'], reverse=True)
    
    print("\nGPU Rankings by Best Energy:")
    print("-" * 50)
    for i, r in enumerate(results):
        print(f"Rank {i+1}: GPU {r['gpu_id']} | Best Energy: {r['best_energy']:.2f}")
    
    # 保存最佳大脑
    best = results[0]
    np.save('best_brain_gpu%d.npy' % best['gpu_id'], best['best_brain'])
    print(f"\n[OK] Best brain saved to: best_brain_gpu{best['gpu_id']}.npy")
    
    # 绘制所有GPU的历史
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, r in enumerate(results):
        ax = axes[i // 2, i % 2]
        h = r['history']
        ax.plot(h['generation'], h['max_energy'], 'r-', label='Max E', alpha=0.7)
        ax.plot(h['generation'], h['mean_energy'], 'b-', label='Mean E', alpha=0.7)
        ax.set_title(f'GPU {r["gpu_id"]} (Best: {r["best_energy"]:.1f})')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4gpu_evolution_history.png', dpi=150)
    print("[OK] 4GPU evolution chart saved to: 4gpu_evolution_history.png")