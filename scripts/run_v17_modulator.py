#!/usr/bin/env python3
"""
v17.0 MODULATOR 算子 1000步实验
===============================
测试新引入的 MODULATOR 门控算子是否能涌现出更复杂的能力

运行方式:
    python scripts/run_v17_modulator.py [--steps N]
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import time
import numpy as np
from collections import Counter
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType
from core.eoe.environment_gpu import EnvironmentGPU


class V17Config(PoolConfig):
    """v17.0 实验配置 (平衡版)"""
    # 池大小
    MAX_AGENTS = 300                  # 适中池子
    
    # 启用 MODULATOR
    CAMBRIAN_MODULATOR_PROB = 0.10
    
    # 代谢成本
    COST_MODULATOR_ACTIVATION = 0.0
    COST_MULTIPLY_ACTIVATION = 0.0
    
    # 平衡代谢
    BASE_METABOLISM = 0.005           # 适中代谢
    NONLINEAR_METABOLISM = False      # 禁用
    AGE_ENABLED = False
    BASAL_COST = 0.0
    NEURAL_COST = 0.0
    
    # 繁殖阈值
    REPRODUCTION_THRESHOLD = 40.0     # 平衡阈值
    CHILD_ENERGY_RATIO = 0.5
    
    # 初始能量
    INITIAL_ENERGY = 80.0
    
    # 突变率
    MUTATION_RATE = 0.5
    ADD_NODE_PROB = 0.3
    
    # 生态
    CROWDING_PENALTY_ENABLED = False
    SOFT_CARRYING_CAP = False


def create_initial_genomes(n_agents: int, config=None):
    """寒武纪初始化 - v17.0 版本 (含 MODULATOR)"""
    if config is None:
        config = V17Config()
    
    genomes = {}
    for i in range(n_agents):
        g = OperatorGenome()
        
        if config.CAMBRIAN_INIT:
            n_nodes = np.random.randint(config.CAMBRIAN_MIN_NODES, config.CAMBRIAN_MAX_NODES + 1)
            
            # v17.0: 包含 MODULATOR 的节点类型选择
            node_types = [NodeType.SENSOR]
            for _ in range(n_nodes - 2):
                rt = np.random.random()
                if rt < config.CAMBRIAN_DELAY_PROB:
                    node_types.append(NodeType.DELAY)
                elif rt < config.CAMBRIAN_DELAY_PROB + config.CAMBRIAN_MULTIPLY_PROB:
                    node_types.append(NodeType.MULTIPLY)
                elif rt < config.CAMBRIAN_DELAY_PROB + config.CAMBRIAN_MULTIPLY_PROB + config.CAMBRIAN_MODULATOR_PROB:
                    node_types.append(NodeType.MODULATOR)  # v17.0 新增
                else:
                    node_types.append(NodeType.THRESHOLD)
            node_types.append(NodeType.ACTUATOR)
            
            for j, nt in enumerate(node_types):
                g.add_node(Node(node_id=j, node_type=nt))
            
            # 添加边
            for src in range(len(node_types) - 1):
                if np.random.random() < 0.7:
                    tgt = np.random.randint(src + 1, len(node_types))
                    weight = np.random.uniform(-0.5, 0.5)
                    if config.SILENT_MUTATION:
                        weight = config.SILENT_WEIGHT
                    g.add_edge(src, tgt, weight=weight, enabled=True)
        else:
            # 最小脑
            g.add_node(Node(0, NodeType.SENSOR))
            g.add_node(Node(1, NodeType.ADD))
            g.add_node(Node(2, NodeType.ACTUATOR))
            g.add_edge(0, 1, weight=0.1)
            g.add_edge(1, 2, weight=0.1)
        
        genomes[i] = g
    
    return genomes


def run_experiment(steps=1000, n_agents=40):
    """运行 v17.0 实验"""
    print("=" * 60)
    print(f"EOE v17.0 MODULATOR 实验 ({steps} 步, {n_agents} agents)")
    print("=" * 60)
    
    config = V17Config()
    config.MAX_POPULATION = n_agents
    config.MAX_STEPS = steps
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"设备: {config.DEVICE}")
    print(f"MODULATOR 概率: {config.CAMBRIAN_MODULATOR_PROB}")
    print(f"动态激活成本: MODULATOR={config.COST_MODULATOR_ACTIVATION}")
    
    # 初始化环境 (参考 v16 配置)
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=config.DEVICE,
        energy_field_enabled=True,
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        seasons_enabled=True,
    )
    
    # 配置能量场参数 - 增加能量供给
    env.energy_field.n_sources = 30
    env.energy_field.source_strength = 800.0
    env.energy_field.source_capacity = 5000.0
    env.energy_field.decay_rate = 1.0  # 无衰减
    env.energy_field.respawn_threshold = 0.10
    
    # 启用能量循环
    config.ENERGY_RECIRCULATION_ENABLED = True
    config.ENERGY_RECIRCULATION_RATIO = 0.8
    
    # 初始化智能体 (使用类似 run_v16 的方式)
    max_pool = getattr(config, 'MAX_AGENTS', 500)  # 使用配置的值
    agents = BatchedAgents(
        initial_population=n_agents,
        max_agents=max_pool,  # 使用配置池大小
        env_width=100.0,
        env_height=100.0,
        device=config.DEVICE,
        init_energy=getattr(config, 'INITIAL_ENERGY', 150.0),
        config=config,
        env=env
    )
    
    # 创建初始种群
    genomes = create_initial_genomes(n_agents, config)
    for idx, g in genomes.items():
        agents.genomes[idx] = g
        agents.state.node_counts[idx] = len(g.nodes)
    
    # 设置大脑矩阵
    alive_genomes = [agents.genomes[i] for i in range(n_agents)]
    agents.set_brains(alive_genomes)
    
    print(f"\n初始种群: {n_agents}")
    
    # 记录初始节点类型
    initial_types = Counter()
    for g in agents.genomes.values():
        if g:
            for n in g.nodes.values():
                initial_types[n.node_type.name] += 1
    
    print("初始节点分布:")
    for nt, cnt in sorted(initial_types.items(), key=lambda x: -x[1]):
        print(f"  {nt}: {cnt}")
    
    # 运行模拟
    print(f"\n开始模拟...")
    start_time = time.time()
    
    for step in range(steps):
        # 环境更新
        env.step()
        
        # 智能体更新
        agents.step(env)
        
        # 每100步输出状态
        if step % 100 == 0:
            alive = agents.alive_mask.sum().item()
            batch = agents.get_active_batch()
            alive_idx = agents.alive_mask[:n_agents].bool()
            avg_energy = agents.state.energies[:n_agents][alive_idx].mean().item() if alive > 0 else 0
            avg_nodes = agents.state.node_counts[:n_agents][alive_idx].float().mean().item() if alive > 0 else 0
            
            # 统计 MODULATOR 数量
            modulator_count = 0
            multiply_count = 0
            for i in range(n_agents):
                if agents.alive_mask[i] and agents.genomes[i]:
                    for n in agents.genomes[i].nodes.values():
                        if n.node_type == NodeType.MODULATOR:
                            modulator_count += 1
                        elif n.node_type == NodeType.MULTIPLY:
                            multiply_count += 1
            
            print(f"Step {step:4d}: 存活 {alive:2d}, "
                  f"平均节点 {avg_nodes:.1f}, "
                  f"MODULATOR: {modulator_count}, MULTIPLY: {multiply_count}")
    
    elapsed = time.time() - start_time
    
    # 最终统计
    print("\n" + "=" * 60)
    print("【实验结果】")
    print("=" * 60)
    
    final_alive = agents.alive_mask[:n_agents].sum().item()
    alive_idx = agents.alive_mask[:n_agents].bool()
    print(f"初始种群: {n_agents}")
    print(f"最终存活: {final_alive} ({final_alive/n_agents*100:.0f}%)")
    print(f"平均节点: {agents.state.node_counts[:n_agents][alive_idx].float().mean().item():.1f}")
    
    # 最终节点类型统计
    final_types = Counter()
    for i in range(n_agents):
        if agents.alive_mask[i] and agents.genomes[i]:
            for n in agents.genomes[i].nodes.values():
                final_types[n.node_type.name] += 1
    
    print("\n最终节点分布:")
    for nt, cnt in sorted(final_types.items(), key=lambda x: -x[1]):
        print(f"  {nt}: {cnt}")
    
    # MODULATOR 分析
    modulator_agents = 0
    modulator_per_agent = []
    for i in range(n_agents):
        if agents.alive_mask[i] and agents.genomes[i]:
            count = sum(1 for n in agents.genomes[i].nodes.values() if n.node_type == NodeType.MODULATOR)
            if count > 0:
                modulator_agents += 1
                modulator_per_agent.append(count)
    
    print(f"\nMODULATOR 分析:")
    print(f"  拥有 MODULATOR 的存活个体: {modulator_agents}/{final_alive}")
    if modulator_per_agent:
        print(f"  平均 MODULATOR 数量: {np.mean(modulator_per_agent):.1f}")
        print(f"  最多 MODULATOR 数量: {max(modulator_per_agent)}")
    
    print(f"\n耗时: {elapsed:.1f}秒")
    print("=" * 60)
    
    return {
        'final_alive': final_alive,
        'avg_nodes': avg_nodes,
        'modulator_count': modulator_count,
        'modulator_agents': modulator_agents,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--agents', type=int, default=40)
    args = parser.parse_args()
    
    run_experiment(steps=args.steps, n_agents=args.agents)