#!/usr/bin/env python3
"""
v15.2 认知溢价实验 - 预加载脑结构
===================================
使用保存的复杂结构初始化种群，观察涌现效果

运行:
    python scripts/run_v15_pretrained.py --structures FILE --steps N
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np
import argparse

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.genome import OperatorGenome
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.t_maze import TMazeEnvironment
from core.eoe.intelligent_prey import IntelligentPreyAdapter, IntelligentPreyConfig
# 使用scripts目录下的加载器
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp/scripts')
from load_brain_structures import BrainStructureLoader

from core.eoe.complexity_tracker import ComplexityTracker


def create_initial_genomes_from_loader(loader: BrainStructureLoader, n_agents: int):
    """从加载器创建初始基因组"""
    # 使用Top 20结构创建种群
    unique_genomes = loader.load_top_n_genomes(n=20)
    
    if not unique_genomes:
        # 回退到随机初始化
        print("⚠️ 加载失败，使用随机初始化")
        return None
    
    # 复制填充到n_agents
    population = {}
    from copy import deepcopy
    
    for i in range(n_agents):
        template = unique_genomes[i % len(unique_genomes)]
        population[i] = deepcopy(template)
    
    print(f"✅ 预加载种群: {len(population)} Agent, {len(unique_genomes)} 种结构")
    return population


def run_experiment(
    structures_file: str,
    steps: int = 5000,
    n_agents: int = 200,
    device: str = 'cpu',
    enable_t_maze: bool = True,
    enable_red_queen: bool = True,
):
    """运行预加载实验"""
    
    print("=" * 60)
    print("v15.2 预加载脑结构实验")
    print("=" * 60)
    print(f"结构文件: {structures_file}")
    print(f"步数: {steps}")
    print(f"种群: {n_agents}")
    print("-" * 60)
    
    # 加载脑结构
    loader = BrainStructureLoader(structures_file)
    
    # 创建环境
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
    )
    
    # 能量场参数
    env.energy_field.n_sources = 15
    env.energy_field.source_strength = 300.0
    env.energy_field.source_capacity = 1500.0
    
    # 实验参数 (修改类变量)
    PoolConfig.REPRODUCTION_THRESHOLD = 40.0
    PoolConfig.MUTATION_RATE = 0.10  # 较低变异率，保留预加载结构
    PoolConfig.SUPERNODE_ENABLED = True
    PoolConfig.HEBBIAN_ENABLED = True
    PoolConfig.BASE_METABOLISM = 0.015  # 更低代谢
    PoolConfig.CAMBRIAN_INIT = False  # 禁用随机初始化
    
    agents = BatchedAgents(
        initial_population=n_agents,
        config=PoolConfig,
        device=device,
    )
    
    # 预加载基因组
    population = create_initial_genomes_from_loader(loader, n_agents)
    
    if population:
        # 批量添加Agent - 直接设置genomes字典
        for agent_id, genome in population.items():
            agents.genomes[agent_id] = genome
            agents.state.node_counts[agent_id] = len(genome.nodes)
            agents.state.energies[agent_id] = 75.0
            agents.alive_mask[agent_id] = True
    else:
        # 回退: 随机初始化
        for i in range(n_agents):
            g = OperatorGenome()
            agents.genomes[i] = g
            agents.state.node_counts[i] = len(g.nodes)
            agents.state.energies[i] = 75.0
            agents.alive_mask[i] = True
    
    # 设置大脑矩阵
    alive_genomes = [agents.genomes[i] for i in range(n_agents)]
    agents.set_brains(alive_genomes)
    
    print(f"初始种群: {agents.alive_mask.sum().item()}")
    
    # T迷宫
    if enable_t_maze:
        PoolConfig.T_MAZE_ENABLED = True
        agents.state.t_maze_signal = torch.zeros(5000, device=device, dtype=torch.long)
        agents.state.t_maze_signal_timer = torch.zeros(5000, device=device, dtype=torch.long)
        agents.state.t_maze_correct_dir = torch.zeros(5000, device=device, dtype=torch.long)
        agents.state.t_maze_episode_step = torch.zeros(5000, device=device, dtype=torch.long)
        agents.state.t_maze_decision_made = torch.zeros(5000, device=device, dtype=torch.bool)
    
    # Red Queen
    prey_adapter = None
    if enable_red_queen:
        PoolConfig.RED_QUEEN_ENABLED = True
        # 使用默认配置
        prey_adapter = IntelligentPreyAdapter(env.energy_field, IntelligentPreyConfig)
    
    # 复杂结构追踪
    complexity_tracker = ComplexityTracker(
        save_dir="experiments/v15_pretrained/saved_structures",
        top_k=50,
        min_complexity=4.0,
        save_interval=500
    )
    
    # 主循环
    import time
    start_time = time.time()
    
    for step in range(steps):
        # 大脑推理
        def brain_fn(batch):
            sensors = agents.get_sensors(env)
            return agents.forward_brains(sensors)
        
        result = agents.step(env=env, dt=1.0, brain_fn=brain_fn)
        
        # Red Queen
        if enable_red_queen and prey_adapter:
            batch = agents.get_active_batch()
            if batch.n > 0:
                prey_adapter.update(batch.positions, batch.linear_velocity, dt=1.0)
        
        # 复杂结构追踪
        if step % 500 == 0:
            current_alive = agents.alive_mask.sum().item()
            if current_alive > 0:
                alive_genomes = []
                fitnesses = []
                batch = agents.get_active_batch()
                for idx in batch.indices.tolist():
                    if idx in agents.genomes:
                        alive_genomes.append(agents.genomes[idx])
                        fitnesses.append(agents.state.energies[idx].item())
                if alive_genomes:
                    complexity_tracker.update(alive_genomes, fitnesses, step)
        
        # 统计
        if step % 100 == 0:
            n_alive = agents.alive_mask.sum().item()
            if n_alive > 0:
                avg_nodes = agents.state.node_counts[agents.alive_mask].float().mean().item()
                avg_energy = agents.state.energies[agents.alive_mask].mean().item()
            else:
                avg_nodes = 0
                avg_energy = 0
            
            elapsed = time.time() - start_time
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            
            print(f"步{step:>5} | 人口{n_alive:>4} | 节点{avg_nodes:.1f} | "
                  f"能量{avg_energy:.1f} | 速度{speed:.0f}步/秒")
            
            # 提前终止
            if n_alive == 0:
                print(f"\n⚠️ 种群在步 {step} 灭绝")
                break
    
    # 最终统计
    print("-" * 60)
    print("最终统计:")
    final_alive = agents.alive_mask.sum().item()
    print(f"  最终人口: {final_alive}")
    if final_alive > 0:
        print(f"  平均节点: {agents.state.node_counts[agents.alive_mask].float().mean().item():.2f}")
        print(f"  平均能量: {agents.state.energies[agents.alive_mask].mean().item():.2f}")
    
    # 保存复杂结构
    complexity_tracker.save(steps)
    complexity_tracker.print_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--structures', 
                       default='experiments/v15_cognitive_premium/saved_structures/complexity_step30000.json')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--population', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-t-maze', action='store_true')
    parser.add_argument('--no-red-queen', action='store_true')
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else 'cuda'
    
    run_experiment(
        structures_file=args.structures,
        steps=args.steps,
        n_agents=args.population,
        device=device,
        enable_t_maze=not args.no_t_maze,
        enable_red_queen=not args.no_red_queen,
    )