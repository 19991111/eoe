#!/usr/bin/env python3
"""
EOE v0.26 泛化能力测试 - 直接在种群中测试
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core


def run_test(name, env_config_fn, train_gens=40, test_gens=20):
    """在不同环境中训练和测试"""
    print(f"\n{'='*50}")
    print(f"🌍 {name}")
    print(f"{'='*50}")
    
    # 创建种群
    pop = core.Population(population_size=15, lifespan=80, use_champion=True)
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    
    # 应用环境配置
    env_config_fn(pop.environment)
    
    # 启用所有边
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    # 训练
    best_train = 0
    for gen in range(train_gens):
        pop.epoch(verbose=False)
        best = max(pop.agents, key=lambda a: a.fitness)
        best_train = max(best_train, best.fitness)
    
    # 记录最佳代理
    best_agent = max(pop.agents, key=lambda a: a.fitness)
    
    # 节点统计
    nodes = {}
    for n in best_agent.genome.nodes.values():
        nodes[n.node_type.name] = nodes.get(n.node_type.name, 0) + 1
    
    # 继续运行更多代看稳定性
    stable_fit = best_train
    for gen in range(test_gens):
        pop.epoch(verbose=False)
        best = max(pop.agents, key=lambda a: a.fitness)
        stable_fit = best.fitness
    
    print(f"  训练: {best_train:.1f} → 稳定: {stable_fit:.1f}")
    print(f"  节点: SWITCH={nodes.get('SWITCH',0)} ADD={nodes.get('ADD',0)} DELAY={nodes.get('DELAY',0)}")
    
    return stable_fit, nodes


def cfg_base(env):
    env.sensor_noise = 0
    env.n_boss_beasts = 0
    env.food_toxicity = {}


def cfg_noisy(env):
    env.sensor_noise = 0.2
    env.n_boss_beasts = 0


def cfg_predator(env):
    env.sensor_noise = 0
    env.n_boss_beasts = 2


def cfg_toxic(env):
    env.sensor_noise = 0
    env.n_boss_beasts = 0
    n = env.n_food
    env.food_toxicity = {i: i < n * 0.5 for i in range(n)}


def cfg_hard(env):
    env.sensor_noise = 0.25
    env.n_boss_beasts = 1
    n = env.n_food
    env.food_toxicity = {i: i < n * 0.3 for i in range(n)}


if __name__ == "__main__":
    tests = [
        ("基准", cfg_base),
        ("噪声", cfg_noisy),
        ("捕食者", cfg_predator),
        ("有毒", cfg_toxic),
        ("困难", cfg_hard),
    ]
    
    results = []
    for name, cfg in tests:
        fit, nodes = run_test(name, cfg)
        results.append((name, fit, nodes))
    
    # 分析
    print("\n" + "="*55)
    print("📊 多环境适应结果")
    print("="*55)
    
    base = results[0][1]
    
    print(f"\n{'环境':<12} {'Fitness':>10} {'SWITCH':>8} {'ADD':>6}")
    print("-"*45)
    
    for name, fit, nodes in results:
        diff = (fit - base) / base * 100 if base > 0 else 0
        sw = nodes.get('SWITCH', 0)
        add = nodes.get('ADD', 0)
        print(f"{name:<12} {fit:>10.1f} {diff:>+7.1f}% {sw:>6} {add:>6}")
    
    best = max(results, key=lambda x: x[1])
    print(f"\n🏆 最佳: {best[0]} ({best[1]:.1f})")