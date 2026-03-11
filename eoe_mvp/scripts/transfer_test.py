#!/usr/bin/env python3
"""
EOE v0.26 迁移学习测试
在不同环境中训练，测试泛化能力
"""

import sys
import os
import time
import copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core


def train_in_env(name, env_modifier, gens=60):
    """在指定环境中训练"""
    print(f"\n{'='*50}")
    print(f"🎓 训练环境: {name}")
    print(f"{'='*50}")
    
    pop = core.Population(population_size=15, lifespan=80, use_champion=True)
    
    # 基础配置
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    
    # 应用环境修改
    env_modifier(pop.environment)
    
    # 启用所有边
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    best_agent = None
    t0 = time.time()
    
    for gen in range(gens):
        pop.epoch(verbose=False)
        best = max(pop.agents, key=lambda a: a.fitness)
        if best.fitness > (best_agent.fitness if best_agent else 0):
            best_agent = copy.deepcopy(best)
    
    elapsed = time.time() - t0
    
    # 统计节点
    nodes = {}
    for n in best_agent.genome.nodes.values():
        nodes[n.node_type.name] = nodes.get(n.node_type.name, 0) + 1
    
    print(f"  训练完成: Fit={best_agent.fitness:.1f}, Time={elapsed:.1f}s")
    print(f"  节点: {nodes}")
    
    return pop.environment, best_agent


def test_agent(env, agent, test_name):
    """在环境中测试代理"""
    # 创建新环境用于测试
    test_pop = core.Population(population_size=1, lifespan=80, use_champion=False)
    
    # 复制环境设置
    test_pop.environment.synaptic_pruning_enabled = False
    test_pop.environment.metabolic_beta = 0.0
    
    # 应用测试环境修改
    env_modifier(test_pop.environment)
    
    # 替换为测试代理
    test_pop.agents[0] = agent
    agent.fitness = 0
    agent.food_eaten = 0
    agent.x = test_pop.environment.width / 2
    agent.y = test_pop.environment.height / 2
    
    # 运行
    for _ in range(80):
        test_pop.environment.step()
    
    return agent.fitness


# 环境修改器
def mod_base(e): pass

def mod_dark(e):
    e.sensor_noise = 0.3

def mod_predator(e):
    e.n_boss_beasts = 2

def mod_toxic(e):
    e.food_toxicity = {i: True for i in range(e.n_food)}


if __name__ == "__main__":
    # 在基准环境训练
    print("\n" + "="*60)
    print("🔬 迁移学习测试")
    print("="*60)
    
    # 训练
    env_base, agent_base = train_in_env("基准", mod_base, gens=60)
    
    # 测试不同环境
    test_envs = [
        ("基准", mod_base),
        ("黑暗", mod_dark),
        ("捕食者", mod_predator),
        ("高毒", mod_toxic),
    ]
    
    print("\n" + "="*50)
    print("📊 基准训练的泛化测试")
    print("="*50)
    
    base_fit = None
    for test_name, mod in test_envs:
        # 复制环境
        test_pop = core.Population(population_size=1, lifespan=80, use_champion=False)
        test_pop.environment.synaptic_pruning_enabled = False
        test_pop.environment.metabolic_beta = 0.0
        mod(test_pop.environment)
        
        # 复制最佳代理
        test_agent = copy.deepcopy(agent_base)
        test_agent.x = test_pop.environment.width / 2
        test_agent.y = test_pop.environment.height / 2
        test_agent.fitness = 0
        test_agent.food_eaten = 0
        test_agent.steps_alive = 0
        
        # 测试
        for _ in range(80):
            test_pop.environment.agents = [test_agent]
            test_pop.environment.step()
            if not test_agent.is_alive:
                break
        
        fit = test_agent.fitness
        food = test_agent.food_eaten
        
        if base_fit is None:
            base_fit = fit
        
        diff = (fit - base_fit) / base_fit * 100 if base_fit > 0 else 0
        print(f"  {test_name:<10} Fit={fit:>8.1f} Food={food:>3}  {diff:>+7.1f}%")