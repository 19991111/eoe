#!/usr/bin/env python3
"""
v16.0 构成性环境 10000步实验
======================
测试v15新机制:
1. 非线性代谢 (Q2)
2. T型迷宫 POMDP (Q3)
3. Red Queen 智能猎物 (Q4)

运行方式:
    python scripts/run_v15_experiment.py [--t-maze] [--red-queen] [--steps N]
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import time
import numpy as np
import argparse
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.intelligent_prey import IntelligentPreyAdapter, IntelligentPreyConfig
from core.eoe.complexity_tracker import ComplexityTracker


def create_initial_genomes(n_agents: int, config=None):
    """寒武纪初始化"""
    if config is None:
        config = PoolConfig()
    
    genomes = {}
    for i in range(n_agents):
        g = OperatorGenome()
        
        if config.CAMBRIAN_INIT:
            n_nodes = np.random.randint(config.CAMBRIAN_MIN_NODES, config.CAMBRIAN_MAX_NODES + 1)
            
            node_types = [NodeType.SENSOR]
            for _ in range(n_nodes - 2):
                rt = np.random.random()
                if rt < config.CAMBRIAN_DELAY_PROB:
                    node_types.append(NodeType.DELAY)
                elif rt < config.CAMBRIAN_DELAY_PROB + config.CAMBRIAN_MULTIPLY_PROB:
                    node_types.append(NodeType.MULTIPLY)
                else:
                    node_types.append(NodeType.THRESHOLD)
            node_types.append(NodeType.ACTUATOR)
            
            for j, nt in enumerate(node_types):
                g.add_node(Node(node_id=j, node_type=nt))
            
            for src in range(len(node_types) - 1):
                if np.random.random() < 0.7:
                    tgt = np.random.randint(src + 1, len(node_types))
                    weight = np.random.uniform(-0.5, 0.5)
                    if config.SILENT_MUTATION:
                        weight = config.SILENT_WEIGHT
                    g.add_edge(src, tgt, weight=weight)
            
            if not any(e['source_id'] == 0 for e in g.edges):
                tgt = np.random.randint(1, len(node_types))
                g.add_edge(0, tgt, weight=config.SILENT_WEIGHT)
        
        genomes[i] = g
    
    return genomes


def run_experiment(
    steps: int = 5000,
    enable_t_maze: bool = False,
    enable_red_queen: bool = False,
    enable_nonlinear: bool = True,
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
):
    """运行v15实验"""
    
    # 配置
    config = PoolConfig()
    
    # v15核心机制
    config.NONLINEAR_METABOLISM = enable_nonlinear
    config.T_MAZE_ENABLED = enable_t_maze
    config.RED_QUEEN_ENABLED = enable_red_queen
    
    # 实验参数 - 降低代谢延长生存
    config.REPRODUCTION_THRESHOLD = 40.0  # 降低阈值
    config.MUTATION_RATE = 0.12           # 变异率
    config.SUPERNODE_ENABLED = True
    config.HEBBIAN_ENABLED = True
    config.BASE_METABOLISM = 0.02         # 极低基础代谢延长寿命
    
    if enable_t_maze:
        config.T_MAZE_SIGNAL_DURATION = 5
        config.T_MAZE_BLIND_ZONE = 15  # 缩短便于观察
        config.T_MAZE_DECISION_DELAY = 20
    
    print("=" * 60)
    print("v15 认知溢价实验")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"步数: {steps}")
    print(f"非线性代谢: {enable_nonlinear}")
    print(f"T型迷宫: {enable_t_maze}")
    print(f"Red Queen: {enable_red_queen}")
    print("-" * 60)
    
    # 创建环境 - v16.0 构成性环境
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        seasons_enabled=False,  # 简化实验
        # v16.0 新增
        matter_grid_enabled=True,
        matter_resolution=1.0,
        wind_field_enabled=True,
        wind_direction=np.pi,
        wind_damage_rate=0.01
    )
    
    print("v16.0 特性: MatterGrid + WindField 启用")
    
    # 配置能量场参数 - 封闭系统能量守恒
    env.energy_field.n_sources = 30          # 更多能量源
    env.energy_field.source_strength = 600.0 # 强脉冲
    env.energy_field.source_capacity = 5000.0 # 大容量
    env.energy_field.decay_rate = 1.0         # 无衰减 (真正守恒)
    env.energy_field.respawn_threshold = 0.20 # 20%时重生
    
    # 启用100%能量循环
    config.ENERGY_RECIRCULATION_ENABLED = True
    config.ENERGY_RECIRCULATION_RATIO = 1.0   # 100%代谢能量回归
    
    # v16.0 启用MatterGrid建造
    config.MATTER_GRID_ENABLED = True
    
    print(f"  能量循环: {config.ENERGY_RECIRCULATION_RATIO*100}% 代谢回归")
    print(f"  MatterGrid: {config.MATTER_GRID_ENABLED}")
    print(f"  能量衰减: {env.energy_field.decay_rate}")
    
    # Red Queen: 添加智能猎物
    prey_adapter = None
    if enable_red_queen:
        prey_config = IntelligentPreyConfig()
        prey_config.detection_range = 25.0
        prey_config.escape_trigger_distance = 15.0
        prey_adapter = IntelligentPreyAdapter(env.energy_field, prey_config)
    
    # 创建Agent池
    agents = BatchedAgents(
        initial_population=200,
        max_agents=5000,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=150.0,
        config=config,
        env=env
    )
    
    # 初始化基因组
    genomes = create_initial_genomes(agents.alive_mask.sum().item(), config)
    for idx, g in genomes.items():
        agents.genomes[idx] = g
        agents.state.node_counts[idx] = len(g.nodes)
    
    # 设置大脑矩阵 (编译基因组)
    alive_genomes = [agents.genomes[i] for i in range(agents.alive_mask.sum().item())]
    agents.set_brains(alive_genomes)
    
    # 统计
    stats = {
        'population': [],
        'avg_nodes': [],
        'avg_energy': [],
        'births': 0,
        'deaths': 0,
    }
    
    # 复杂结构追踪器
    complexity_tracker = ComplexityTracker(
        save_dir="experiments/v15_cognitive_premium/saved_structures",
        top_k=50,
        min_complexity=4.0,
        save_interval=500
    )
    
    if enable_t_maze:
        stats['t_maze_episodes'] = []
        stats['t_maze_correct'] = []
    
    print(f"初始种群: {agents.alive_mask.sum().item()}")
    start_time = time.time()
    
    # 主循环
    for step in range(steps):
        # 大脑推理
        def brain_fn(batch):
            # 获取传感器输入
            sensors = agents.get_sensors(env)
            # 前向传播
            return agents.forward_brains(sensors)
        
        # 步进
        result = agents.step(env=env, dt=1.0, brain_fn=brain_fn)
        
        # Red Queen: 更新智能猎物
        if enable_red_queen and prey_adapter is not None:
            active_batch = agents.get_active_batch()
            if active_batch.n > 0:
                prey_adapter.update(
                    active_batch.positions,
                    active_batch.linear_velocity,
                    dt=1.0
                )
        
        # 复杂结构追踪 (每500步)
        if step % 500 == 0:
            current_alive = agents.alive_mask.sum().item()
            if current_alive > 0:
                # 从pool获取活着的基因组
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
            avg_nodes = agents.state.node_counts[agents.alive_mask].float().mean().item()
            avg_energy = agents.state.energies[agents.alive_mask].mean().item()
            
            stats['population'].append(n_alive)
            stats['avg_nodes'].append(avg_nodes)
            stats['avg_energy'].append(avg_energy)
            stats['births'] += result.get('births', 0)
            stats['deaths'] += result.get('deaths', 0)
            
            if enable_t_maze:
                episodes = agents.state.t_maze_episodes[agents.alive_mask].sum().item()
                correct = agents.state.t_maze_correct[agents.alive_mask].sum().item()
                stats['t_maze_episodes'].append(episodes)
                stats['t_maze_correct'].append(correct)
            
            elapsed = time.time() - start_time
            speed = (step + 1) / elapsed
            
            print(f"步{step:>5} | 人口{n_alive:>4} | 节点{avg_nodes:.1f} | "
                  f"能量{avg_energy:.1f} | 速度{speed:.0f}步/秒")
    
    # 最终统计
    print("-" * 60)
    print("最终统计:")
    print(f"  总出生: {stats['births']}")
    print(f"  总死亡: {stats['deaths']}")
    print(f"  最终人口: {stats['population'][-1]}")
    print(f"  平均节点: {stats['avg_nodes'][-1]:.2f}")
    print(f"  平均能量: {stats['avg_energy'][-1]:.2f}")
    
    if enable_t_maze and stats['t_maze_episodes']:
        total_ep = stats['t_maze_episodes'][-1]
        total_corr = stats['t_maze_correct'][-1]
        if total_ep > 0:
            accuracy = total_corr / total_ep
            print(f"  T迷宫准确率: {accuracy:.1%} ({total_corr}/{total_ep})")
    
    # 保存复杂结构并打印摘要
    complexity_tracker.save(steps)
    complexity_tracker.print_summary()
    
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='v15认知溢价实验')
    parser.add_argument('--steps', type=int, default=10000, help='运行步数')
    parser.add_argument('--t-maze', action='store_true', help='启用T型迷宫')
    parser.add_argument('--red-queen', action='store_true', help='启用Red Queen')
    parser.add_argument('--no-nonlinear', action='store_true', help='禁用非线性代谢')
    parser.add_argument('--cpu', action='store_true', help='使用CPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    run_experiment(
        steps=args.steps,
        enable_t_maze=args.t_maze,
        enable_red_queen=args.red_queen,
        enable_nonlinear=not args.no_nonlinear,
        device=device
    )