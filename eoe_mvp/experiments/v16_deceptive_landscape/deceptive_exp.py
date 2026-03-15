#!/usr/bin/env python3
"""v16.7 Deceptive Landscape Evolution - 极端不对称版

核心策略: 彻底切断"免费午餐"
- 可见能量: 0.1x (垃圾食品)
- 隐身能量: 20x (超级大补丸)  
- 可见时间: 5% (原来10%)
- 主动感知: 不动就看不见

目标: 打破"伏地魔"策略，迫使Agent移动捕食
"""

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
    print("v16.7 Deceptive Landscape Evolution - 极端不对称版", flush=True)
    print("  [GOAL] 彻底饿死伏地魔!", flush=True)
    print("  🗑️ 可见=0.1x (垃圾食品) | ⚡ 隐身=20x (大补丸)", flush=True)
    print("  📉 可见时间: 5% (原来10%)", flush=True)
    print("=" * 60, flush=True)
    
    device = 'cuda:0'
    steps = 3000  # v16.5 快速测试
    initial_pop = 50
    
    config = PoolConfig()
    # 启用预加载脑结构
    config.PRETRAINED_INIT = True
    config.PRETRAINED_STRUCTURES_FILE = 'experiments/v15_pretrained/saved_structures/top_k_structures.json'
    config.PRETRAINED_TOP_N = 10
    
    # 临时: 如果预加载失败则使用寒武纪
    config.CAMBRIAN_INIT = True
    
    # v16.7 极端不对称参数 (核心!)
    config.VISIBLE_REWARD_MULTIPLIER = 0.1     # 可见能量0.1x (垃圾食品!)
    config.INVISIBLE_REWARD_MULTIPLIER = 20.0  # 隐身能量20x (超级大补丸!)
    config.COGNITIVE_PREMIUM_MULTIPLIER = 10.0 # 保持兼容
    config.ENABLE_INVISIBLE_SENSING = True     # 允许感知隐身能量
    config.COGNITIVE_PREMIUM_ONLY_INVISIBLE = True
    
    # v16.5 主动感知参数
    config.ACTIVE_SENSING_ENABLED = True
    config.ACTIVE_SENSING_THRESHOLD = 0.3       # 达到此速度则100%感知
    config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.05 # 静止时最低5%感知
    config.INVISIBLE_SENSING_BOOST = 2.0        # 隐身能量需要2x感知效率
    
    # 可见时间降至5%
    config.RESOURCE_CYCLE_ENABLED = True
    config.RESOURCE_CYCLE_LENGTH = 500  # 周期长度
    config.RESOURCE_FADE_STEPS = 475    # 475步不可见，25步可见 (5%)
    
    # 提高基础代谢 (让躺平更难过活)
    config.BASE_METABOLISM = 0.08
    
    # 降低运动惩罚，让Agent能移动
    config.MOVEMENT_PENALTY = 0.01
    
    # 提高基础代谢增加生存压力
    config.BASE_METABOLISM = 0.15  # 3倍于原来
    config.AGE_ENABLED = True       # 启用年龄惩罚
    
    env = EnvironmentGPU(
        width=100.0, height=100.0, resolution=1.0, device=device,
        energy_field_enabled=False,
        flickering_energy_enabled=True,
        flickering_period=10,       # 只10步可见(10%)
        flickering_invisible_moves=90,  # 90步不可见(90%)
        flickering_speed=0.5,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
    )
    
    # 启用隐身期曲线运动(减少随机碰撞)
    if hasattr(env.flickering_energy_field, 'set_invisible_motion_curved'):
        env.flickering_energy_field.set_invisible_motion_curved(True)
        print("  ✅ 隐身期曲线运动已启用", flush=True)
    
    print("Environment ready", flush=True)
    
    # Limit max_agents to avoid hanging at >= 100
    max_agents_limit = min(initial_pop * 4, 80)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents_limit,
        env_width=100.0, env_height=100.0,
        device=device, init_energy=40.0,  # 降低初始能量
        config=config, env=env
    )
    print(f"Initial agents: {agents.alive_mask.sum().item()}", flush=True)
    
    # 初始化基因组并设置大脑
    if config.PRETRAINED_INIT and config.PRETRAINED_STRUCTURES_FILE:
        alive_genomes = agents._load_pretrained_genomes(agents.alive_mask.sum().item())
        if alive_genomes:
            print(f"[主程序] 使用预加载脑结构: {len(alive_genomes)} 种", flush=True)
        else:
            alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    else:
        alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
        print(f"[主程序] 寒武纪初始化: {len(alive_genomes)} 个Agent", flush=True)
    
    agents.set_brains(alive_genomes)
    
    stats = {'steps': [], 'population': [], 'avg_nodes': [], 'complex_structures': [], 'avg_velocity': []}
    
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
                
                # v16.2: 计算平均速度 (监控运动惩罚效果)
                alive_velocities = agents.state.linear_velocity[alive_indices]
                avg_vel = alive_velocities.norm(dim=-1).mean().item()
                stats['avg_velocity'].append(avg_vel)
        
        if (step + 1) % 150 == 0:
            energy_stats = env.flickering_energy_field.get_stats() if env.flickering_energy_enabled else {}
            n_alive = agents.alive_mask.sum().item()
            avg_nodes = np.mean(stats['avg_nodes'][-5:]) if stats['avg_nodes'] else 0
            complex_count = stats['complex_structures'][-1] if stats['complex_structures'] else 0
            avg_vel = stats['avg_velocity'][-1] if stats['avg_velocity'] else 0
            elapsed = time.time() - start_time
            
            print(f"Step {step+1:5d} | Pop: {n_alive:4d} | "
                  f"Nodes: {avg_nodes:.2f} | Complex: {complex_count:3d} | "
                  f"Vel: {avg_vel:.3f} | "
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