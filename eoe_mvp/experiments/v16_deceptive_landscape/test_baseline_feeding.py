#!/usr/bin/env python3
"""
v16.1 基准测试：验证向量化能量消耗机制
- 不闪烁版本，验证基础代谢平衡
- 预期：avg_nodes 应回到 ~4.0
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import os
os.environ['PYTHONUNBUFFERED'] = '1'

print("DEBUG: Starting baseline test", flush=True)

import torch
import numpy as np
import time
import json

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU

print("DEBUG: Imports done", flush=True)


def run_baseline_test():
    print("=" * 60, flush=True)
    print("v16.1 Baseline Test - Vectorized Energy Consumption", flush=True)
    print("=" * 60, flush=True)
    
    device = 'cuda:0'
    steps = 2000
    initial_pop = 50
    
    config = PoolConfig()
    # 启用预加载脑结构
    config.PRETRAINED_INIT = True
    config.PRETRAINED_STRUCTURES_FILE = 'experiments/v15_pretrained/saved_structures/top_k_structures.json'
    config.PRETRAINED_TOP_N = 10
    
    # 不闪烁的普通能量场
    env = EnvironmentGPU(
        width=100.0, height=100.0, resolution=1.0, device=device,
        energy_field_enabled=True,
        flickering_energy_enabled=False,  # 不闪烁
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
    )
    
    # 调整能量场参数
    if hasattr(env, 'energy_field'):
        env.energy_field.n_sources = 30
        env.energy_field.source_strength = 150.0
        env.energy_field.sources = torch.zeros(30, 6, device=device)
        env.energy_field._init_sources()
    print("Environment ready (no flickering)", flush=True)
    
    max_agents_limit = min(initial_pop * 4, 80)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents_limit,
        env_width=100.0, env_height=100.0,
        device=device, init_energy=150.0,
        config=config, env=env
    )
    print(f"Initial agents: {agents.alive_mask.sum().item()}", flush=True)
    
    # 初始化基因组并设置大脑
    if config.PRETRAINED_INIT and config.PRETRAINED_STRUCTURES_FILE:
        # 预加载机制：从文件加载结构
        alive_genomes = agents._load_pretrained_genomes(agents.alive_mask.sum().item())
        if alive_genomes:
            print(f"[主程序] 使用预加载脑结构: {len(alive_genomes)} 种", flush=True)
        else:
            print("[主程序] 预加载失败，使用寒武纪初始化", flush=True)
            alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    else:
        # 寒武纪初始化
        alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
        print(f"[主程序] 寒武纪初始化: {len(alive_genomes)} 个Agent", flush=True)
    
    # 设置大脑矩阵
    agents.set_brains(alive_genomes)
    
    stats = {'steps': [], 'population': [], 'avg_nodes': [], 'complex_structures': [], 'total_energy': []}
    
    start_time = time.time()
    
    for step in range(steps):
        env.step()
        step_stats = agents.step(env=env, dt=0.1)
        
        # 记录能量场总能量
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            total_energy = env.energy_field.field.sum().item()
        else:
            total_energy = 0
        
        if (step + 1) % 100 == 0:
            n_alive = step_stats['n_alive']
            stats['steps'].append(step)
            stats['population'].append(n_alive)
            stats['total_energy'].append(total_energy)
            
            if n_alive > 0:
                alive_indices = agents.alive_mask.nonzero(as_tuple=True)[0]
                avg_nodes = agents.state.node_counts[alive_indices].float().mean().item()
                stats['avg_nodes'].append(avg_nodes)
                complex_count = (agents.state.node_counts[alive_indices] > 4).sum().item()
                stats['complex_structures'].append(complex_count)
            else:
                stats['avg_nodes'].append(0)
                stats['complex_structures'].append(0)
        
        if (step + 1) % 200 == 0:
            n_alive = agents.alive_mask.sum().item()
            avg_nodes = stats['avg_nodes'][-1] if stats['avg_nodes'] else 0
            complex_count = stats['complex_structures'][-1] if stats['complex_structures'] else 0
            elapsed = time.time() - start_time
            
            print(f"Step {step+1:5d} | Pop: {n_alive:4d} | "
                  f"Nodes: {avg_nodes:.2f} | Complex: {complex_count:3d} | "
                  f"Energy: {total_energy:.1f} | Time: {elapsed:.1f}s", flush=True)
        
        if agents.alive_mask.sum().item() == 0:
            print(f"Extinction at step {step+1}!", flush=True)
            break
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60, flush=True)
    print("Baseline Test Complete!", flush=True)
    print(f"Steps: {step+1}, Time: {elapsed:.1f}s", flush=True)
    print(f"Final pop: {agents.alive_mask.sum().item()}", flush=True)
    
    # 最终统计
    if stats['avg_nodes']:
        final_avg_nodes = stats['avg_nodes'][-1]
        print(f"Final avg_nodes: {final_avg_nodes:.2f}", flush=True)
        
        # 验证标准
        if final_avg_nodes >= 3.5:
            print("✅ BASELINE PASSED: avg_nodes >= 3.5", flush=True)
        else:
            print("❌ BASELINE FAILED: avg_nodes < 3.5", flush=True)
    
    results = {
        'stats': stats, 
        'final': {
            'step': step+1, 
            'population': agents.alive_mask.sum().item(),
            'avg_nodes': stats['avg_nodes'][-1] if stats['avg_nodes'] else 0
        }
    }
    with open("experiments/v16_deceptive_landscape/baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_baseline_test()