#!/usr/bin/env python3
"""
v16.0 构成性环境10000代演化实验
===============================
测试:
1. MatterGrid 建造/破坏机制
2. WindField 挡风墙行为涌现
3. SuperNode 结构演化
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np
import time
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType

# ==================== 配置 ====================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GENERATIONS = 10000
STEPS_PER_GEN = 100
INITIAL_POP = 300
MAX_AGENTS = 2000

print("=" * 60)
print("v16.0 构成性环境 10000代演化实验")
print("=" * 60)
print(f"Device: {device}")
print(f"Generations: {GENERATIONS}")
print(f"Steps/Gen: {STEPS_PER_GEN}")
print(f"Initial Pop: {INITIAL_POP}")
print("-" * 60)

# v16.0 配置
config = PoolConfig()
config.MATTER_GRID_ENABLED = True
config.N_BRAIN_OUTPUTS_V16 = 7
config.BASE_METABOLISM = 0.008  # 极低代谢
config.REPRODUCTION_THRESHOLD = 100.0  # 提高分裂阈值
config.ENERGY_RECIRCULATION_ENABLED = True
config.ENERGY_RECIRCULATION_RATIO = 0.5
config.SUPERNODE_ENABLED = True
config.HEBBIAN_ENABLED = True
config.FOOD_ENERGY = 50.0  # 增加食物能量
config.N_FOOD = 50  # 增加食物数量

# 创建环境 - 启用所有v16特性
env = EnvironmentGPU(
    width=100.0,
    height=100.0,
    resolution=1.0,
    device=device,
    energy_field_enabled=True,
    impedance_field_enabled=False,
    stigmergy_field_enabled=False,
    danger_field_enabled=False,  # 关闭危险场
    matter_grid_enabled=True,
    wind_field_enabled=True,
    wind_direction=np.pi,  # 西风
    wind_damage_rate=0.01,  # 极温和伤害
    seasons_enabled=False  # 关闭季节
)

print(f"MatterGrid: enabled")
print(f"WindField: direction=π, damage=0.03")
print("-" * 60)

# 创建智能体池
agents = BatchedAgents(
    initial_population=INITIAL_POP,
    max_agents=MAX_AGENTS,
    env_width=100.0,
    env_height=100.0,
    device=device,
    init_energy=180.0,
    config=config,
    env=env
)

# 寒武纪初始化
def create_genomes(n):
    genomes = {}
    for i in range(n):
        g = OperatorGenome()
        g.add_node(Node(node_id=0, node_type=NodeType.SENSOR))
        g.add_node(Node(node_id=1, node_type=NodeType.THRESHOLD))
        g.add_node(Node(node_id=2, node_type=NodeType.MULTIPLY))
        g.add_node(Node(node_id=3, node_type=NodeType.ACTUATOR_THRUST_X))
        g.add_node(Node(node_id=4, node_type=NodeType.ACTUATOR_THRUST_Y))
        genomes[i] = g
    return genomes

genomes = create_genomes(agents.alive_mask.sum().item())
for idx, g in genomes.items():
    agents.genomes[idx] = g
    agents.state.node_counts[idx] = len(g.nodes)

alive_genomes = [agents.genomes[i] for i in range(agents.alive_mask.sum().item())]
agents.set_brains(alive_genomes)

print(f"初始智能体: {agents.alive_mask.sum().item()}")
print("-" * 60)

# 统计
stats = {
    'population': [],
    'avg_nodes': [],
    'avg_energy': [],
    'wall_count': [],
    'births': 0,
    'deaths': 0,
}

# ==================== 主循环 ====================
start_time = time.time()
total_steps = 0

for gen in range(GENERATIONS):
    gen_start = time.time()
    
    # 每个generation的步进
    for step in range(STEPS_PER_GEN):
        env.step()
        step_stats = agents.step(env=env, dt=0.1)
        total_steps += 1
        
        # 统计
        if (step + 1) % 50 == 0:
            n_alive = step_stats['n_alive']
            stats['population'].append(n_alive)
            stats['births'] += step_stats['births']
            stats['deaths'] += step_stats['deaths']
            
            # 平均节点数
            if n_alive > 0:
                alive_indices = agents.alive_mask.nonzero(as_tuple=True)[0]
                avg_nodes = agents.state.node_counts[alive_indices].float().mean().item()
                stats['avg_nodes'].append(avg_nodes)
                
                # 平均能量
                avg_energy = agents.state.energies[alive_indices].mean().item()
                stats['avg_energy'].append(avg_energy)
            else:
                stats['avg_nodes'].append(0)
                stats['avg_energy'].append(0)
            
            # 墙壁数量
            if env.matter_grid is not None:
                wall_count = (env.matter_grid == 1).sum().item()
                stats['wall_count'].append(wall_count)
    
    # 复杂结构检测 (简化版)
    if (gen + 1) % 10 == 0:  # 每10代检测一次
        alive_idx = agents.alive_mask.nonzero(as_tuple=True)[0]
        if len(alive_idx) > 0:
            node_counts = agents.state.node_counts[alive_idx].float().mean().item()
            if node_counts > 5:
                print(f"  🧬 Gen {gen+1}: Avg nodes = {node_counts:.2f}")
    
    # 输出
    gen_time = time.time() - gen_start
    n_alive = agents.alive_mask.sum().item()
    wall_count = (env.matter_grid == 1).sum().item() if env.matter_grid is not None else 0
    
    if (gen + 1) % 100 == 0 or gen == 0:
        avg_energy = np.mean(stats['avg_energy'][-10:]) if stats['avg_energy'] else 0
        print(f"Gen {gen+1:5d} | Pop: {n_alive:4d} | Walls: {wall_count:3d} | "
              f"Energy: {avg_energy:.1f} | Time: {gen_time:.1f}s")
    
    # 检查灭绝
    if n_alive == 0:
        print(f"\n⚠️  种群灭绝于第 {gen+1} 代!")
        break

# ==================== 结果 ====================
elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("实验完成!")
print("=" * 60)
print(f"运行代数: {gen+1}")
print(f"总步数: {total_steps}")
print(f"存活时间: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"最终人口: {agents.alive_mask.sum().item()}")
print(f"总出生: {stats['births']}")
print(f"总死亡: {stats['deaths']}")

if env.matter_grid is not None:
    final_walls = (env.matter_grid == 1).sum().item()
    print(f"最终墙壁: {final_walls}")

# 保存结果
import json
results = {
    'config': {
        'generations': GENERATIONS,
        'steps_per_gen': STEPS_PER_GEN,
        'matter_grid': True,
        'wind_field': True
    },
    'stats': {
        'population': stats['population'],
        'avg_nodes': stats['avg_nodes'],
        'avg_energy': stats['avg_energy'],
        'wall_count': stats['wall_count'],
        'births': stats['births'],
        'deaths': stats['deaths']
    },
    'final': {
        'generation': gen + 1,
        'population': agents.alive_mask.sum().item(),
        'walls': final_walls if env.matter_grid is not None else 0,
        'elapsed_seconds': elapsed
    }
}

import os
os.makedirs("experiments/v16_compositional", exist_ok=True)
with open("experiments/v16_compositional/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: experiments/v16_compositional/results.json")