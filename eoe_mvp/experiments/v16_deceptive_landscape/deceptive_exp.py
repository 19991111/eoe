#!/usr/bin/env python3
"""v16.1 Deceptive Landscape Evolution"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import os
os.environ['PYTHONUNBUFFERED'] = '1'

print("DEBUG: Starting", flush=True)

import torch
import numpy as np
import time
import json

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU

print("DEBUG: Imports done", flush=True)


def run_experiment():
    print("=" * 60, flush=True)
    print("v16.1 Deceptive Landscape Evolution", flush=True)
    print("=" * 60, flush=True)
    
    device = 'cuda:0'
    steps = 500
    initial_pop = 50
    
    config = PoolConfig()
    
    env = EnvironmentGPU(
        width=100.0, height=100.0, resolution=1.0, device=device,
        energy_field_enabled=False,
        flickering_energy_enabled=True,
        flickering_period=25,
        flickering_invisible_moves=75,
        flickering_speed=0.5,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
    )
    print("Environment ready", flush=True)
    
    # Limit max_agents to avoid hanging at >= 100
    max_agents_limit = min(initial_pop * 4, 80)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents_limit,
        env_width=100.0, env_height=100.0,
        device=device, init_energy=150.0,
        config=config, env=env
    )
    print(f"Initial agents: {agents.alive_mask.sum().item()}", flush=True)
    
    stats = {'steps': [], 'population': [], 'avg_nodes': [], 'complex_structures': []}
    
    start_time = time.time()
    
    for step in range(steps):
        env.step()
        step_stats = agents.step(env=env, dt=0.1)
        
        if (step + 1) % 100 == 0:
            n_alive = step_stats['n_alive']
            stats['steps'].append(step)
            stats['population'].append(n_alive)
            
            if n_alive > 0:
                alive_indices = agents.alive_mask.nonzero(as_tuple=True)[0]
                avg_nodes = agents.state.node_counts[alive_indices].float().mean().item()
                stats['avg_nodes'].append(avg_nodes)
                complex_count = (agents.state.node_counts[alive_indices] > 4).sum().item()
                stats['complex_structures'].append(complex_count)
        
        if (step + 1) % 200 == 0:
            energy_stats = env.flickering_energy_field.get_stats() if env.flickering_energy_enabled else {}
            n_alive = agents.alive_mask.sum().item()
            avg_nodes = np.mean(stats['avg_nodes'][-5:]) if stats['avg_nodes'] else 0
            complex_count = stats['complex_structures'][-1] if stats['complex_structures'] else 0
            elapsed = time.time() - start_time
            
            print(f"Step {step+1:5d} | Pop: {n_alive:4d} | "
                  f"Nodes: {avg_nodes:.2f} | Complex: {complex_count:3d} | "
                  f"Visible: {energy_stats.get('visible_sources', 0)}/{energy_stats.get('total_sources', 0)} | "
                  f"Time: {elapsed:.1f}s", flush=True)
        
        if agents.alive_mask.sum().item() == 0:
            print(f"Extinction at step {step+1}!", flush=True)
            break
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60, flush=True)
    print("Complete!", flush=True)
    print(f"Steps: {step+1}, Time: {elapsed:.1f}s", flush=True)
    print(f"Final pop: {agents.alive_mask.sum().item()}", flush=True)
    
    results = {'stats': stats, 'final': {'step': step+1, 'population': agents.alive_mask.sum().item()}}
    with open("experiments/v16_deceptive_landscape/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_experiment()