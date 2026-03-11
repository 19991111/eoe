#!/usr/bin/env python3
"""
EOE v0.26 行为能力分析
分析已训练网络的具体行为模式
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core


def analyze_behavior(env_name, env_cfg, gens=60):
    """训练并分析行为"""
    print(f"\n{'='*50}")
    print(f"🔬 {env_name}")
    print(f"{'='*50}")
    
    # 训练
    pop = core.Population(population_size=15, lifespan=80, use_champion=True)
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    env_cfg(pop.environment)
    
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    # 训练时记录行为
    food_eaten_history = []
    distance_history = []
    
    for gen in range(gens):
        pop.epoch(verbose=False)
        
        # 记录当代最佳
        best = max(pop.agents, key=lambda a: a.fitness)
        food_eaten_history.append(best.food_eaten)
        distance_history.append(getattr(best, 'total_distance', 0))
    
    # 统计节点类型
    best = max(pop.agents, key=lambda a: a.fitness)
    node_types = {}
    for n in best.genome.nodes.values():
        nt = n.node_type.name
        node_types[nt] = node_types.get(nt, 0) + 1
    
    # 计算边活跃度
    active_edges = 0
    for e in best.genome.edges:
        if e['enabled']:
            active_edges += 1
    
    print(f"  节点: {node_types}")
    print(f"  活跃边: {active_edges}/132")
    print(f"  食物: 总={sum(food_eaten_history)}, 代均={np.mean(food_eaten_history):.1f}")
    print(f"  移动: 最终={distance_history[-1]:.0f}")
    
    return {
        'env': env_name,
        'fitness': best.fitness,
        'food': sum(food_eaten_history),
        'food_per_gen': np.mean(food_eaten_history),
        'distance': distance_history[-1],
        'nodes': node_types,
        'active_edges': active_edges
    }


def cfg_base(e):
    e.n_food = 5


def cfg_rich(e):
    e.n_food = 10


def cfg_predator(e):
    e.n_boss_beasts = 2


def cfg_noisy(e):
    e.sensor_noise = 0.2


def cfg_large(e):
    e.width = 200
    e.height = 200


if __name__ == "__main__":
    configs = [
        ("基准", cfg_base),
        ("富食", cfg_rich),
        ("捕食者", cfg_predator),
        ("噪声", cfg_noisy),
        ("大型地图", cfg_large),
    ]
    
    results = []
    for name, cfg in configs:
        r = analyze_behavior(name, cfg)
        results.append(r)
    
    # 总结分析
    print("\n" + "="*50)
    print("📊 涌现能力分析")
    print("="*50)
    
    # 1. 通信能力
    print("\n【通信能力】")
    for r in results:
        comm_out = r['nodes'].get('COMM_OUT', 0)
        comm_in = r['nodes'].get('COMM_IN', 0)
        radar = r['nodes'].get('AGENT_RADAR_SENSOR', 0)
        print(f"  {r['env']:<10} COMM_OUT={comm_out}, COMM_IN={comm_in}, RADAR={radar}")
    
    # 2. 决策能力
    print("\n【决策能力】")
    for r in results:
        sw = r['nodes'].get('SWITCH', 0)
        th = r['nodes'].get('THRESHOLD', 0)
        print(f"  {r['env']:<10} SWITCH={sw}, THRESHOLD={th}, 复杂度={sw+th}")
    
    # 3. 记忆能力
    print("\n【记忆能力】")
    for r in results:
        delay = r['nodes'].get('DELAY', 0)
        pred = r['nodes'].get('PREDICTOR', 0)
        buff = r['nodes'].get('BUFFER', 0)
        print(f"  {r['env']:<10} DELAY={delay}, PREDICTOR={pred}, BUFFER={buff}")
    
    # 4. 学习能力
    print("\n【学习能力】")
    for r in results:
        upd = r['nodes'].get('UPDATE_WEIGHT', 0)
        mod = r['nodes'].get('MODULATOR', 0)
        rep = r['nodes'].get('REWARD_PREDICTOR', 0)
        print(f"  {r['env']:<10} UPDATE={upd}, MODULATOR={mod}, REWARD_PRED={rep}")
    
    # 5. 算术能力
    print("\n【算术/逻辑能力】")
    for r in results:
        add = r['nodes'].get('ADD', 0)
        mul = r['nodes'].get('MULTIPLY', 0)
        poly = r['nodes'].get('POLY', 0)
        print(f"  {r['env']:<10} ADD={add}, MULTIPLY={mul}, POLY={poly}")
    
    # 综合评估
    print("\n" + "="*50)
    print("🏆 环境vs能力矩阵")
    print("="*50)
    
    print(f"\n{'环境':<10} {'通信':>6} {'决策':>6} {'记忆':>6} {'学习':>6} {'算术':>6}")
    print("-"*50)
    
    for r in results:
        comm = (r['nodes'].get('COMM_OUT', 0) + r['nodes'].get('COMM_IN', 0)) * 2
        decision = (r['nodes'].get('SWITCH', 0) + r['nodes'].get('THRESHOLD', 0)) // 2
        memory = (r['nodes'].get('DELAY', 0) + r['nodes'].get('PREDICTOR', 0)) // 10
        learning = (r['nodes'].get('UPDATE_WEIGHT', 0) + r['nodes'].get('REWARD_PREDICTOR', 0) + r['nodes'].get('MODULATOR', 0)) * 10
        arithmetic = (r['nodes'].get('ADD', 0) + r['nodes'].get('MULTIPLY', 0)) // 3
        
        print(f"{r['env']:<10} {comm:>6} {decision:>6} {memory:>6} {learning:>6} {arithmetic:>6}")