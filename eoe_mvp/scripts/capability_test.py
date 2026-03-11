#!/usr/bin/env python3
"""
EOE v0.26 能力涌现测试
测试不同环境下训练的网络涌现出的具体能力
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import core


def train_agent(env_config_fn, gens=50):
    """训练代理并返回最佳"""
    pop = core.Population(population_size=15, lifespan=80, use_champion=True)
    pop.environment.synaptic_pruning_enabled = False
    pop.environment.metabolic_beta = 0.0
    env_config_fn(pop.environment)
    
    for agent in pop.agents:
        for edge in agent.genome.edges:
            edge['enabled'] = True
    
    for gen in range(gens):
        pop.epoch(verbose=False)
    
    return max(pop.agents, key=lambda a: a.fitness)


def test_capability(agent, test_fn, env_config_fn):
    """测试特定能力"""
    # 创建测试环境
    test_pop = core.Population(population_size=1, lifespan=80, use_champion=False)
    test_pop.environment.synaptic_pruning_enabled = False
    test_pop.environment.metabolic_beta = 0.0
    env_config_fn(test_pop.environment)
    
    # 重置代理
    test_agent = agent
    test_agent.x = test_pop.environment.width / 2
    test_agent.y = test_pop.environment.height / 2
    test_agent.fitness = 0
    test_agent.food_eaten = 0
    test_agent.steps_alive = 0
    test_agent.is_alive = True
    
    # 运行测试
    results = []
    test_pop.environment.agents = [test_agent]
    
    for step in range(80):
        if not test_agent.is_alive:
            break
        
        # 记录测试前状态
        pre_x, pre_y = test_agent.x, test_agent.y
        
        test_pop.environment.step()
        test_pop.environment.agents = [test_agent]
        
        # 获取测试指标
        result = test_fn(test_agent, step, pre_x, pre_y)
        results.append(result)
    
    return results


# ===== 能力测试函数 =====

def test_foraging(agent, step, pre_x, pre_y):
    """觅食能力测试"""
    return {
        'food_eaten': agent.food_eaten,
        'distance_moved': np.sqrt((agent.x - pre_x)**2 + (agent.y - pre_y)**2),
        'total_distance': getattr(agent, 'total_distance', 0)
    }


def test_predator_avoidance(agent, step, pre_x, pre_y):
    """捕食者逃避测试"""
    env = agent.environment if hasattr(agent, 'environment') else None
    nearest_beast = 999
    if env and hasattr(env, 'boss_beasts'):
        for beast in getattr(env, 'boss_beasts', []):
            dist = np.sqrt((agent.x - beast[0])**2 + (agent.y - beast[1])**2)
            nearest_beast = min(nearest_beast, dist)
    
    return {
        'nearest_beast': nearest_beast,
        'survived': agent.is_alive,
        'steps': agent.steps_alive
    }


def test_exploration(agent, step, pre_x, pre_y):
    """探索能力测试"""
    return {
        'x': agent.x,
        'y': agent.y,
        'distance_moved': np.sqrt((agent.x - pre_x)**2 + (agent.y - pre_y)**2),
    }


def test_communication(agent, step, pre_x, pre_y):
    """通信能力测试"""
    # 检查是否有COMM节点
    has_comm = any(n.node_type.name in ['COMM_OUT', 'COMM_IN'] 
                   for n in agent.genome.nodes.values())
    
    return {
        'has_comm': has_comm,
        'comm_nodes': sum(1 for n in agent.genome.nodes.values() 
                        if n.node_type.name in ['COMM_OUT', 'COMM_IN'])
    }


def test_decision_making(agent, step, pre_x, pre_y):
    """决策能力测试 - 统计SWITCH使用"""
    switch_count = sum(1 for n in agent.genome.nodes.values() 
                      if n.node_type.name == 'SWITCH')
    threshold_count = sum(1 for n in agent.genome.nodes.values() 
                         if n.node_type.name == 'THRESHOLD')
    
    return {
        'switch_nodes': switch_count,
        'threshold_nodes': threshold_count,
        'decision_complexity': switch_count + threshold_count
    }


# ===== 环境配置 =====

def cfg_base(e):
    e.n_food = 5
    e.n_boss_beasts = 0
    e.sensor_noise = 0


def cfg_predator(e):
    e.n_food = 5
    e.n_boss_beasts = 2
    e.sensor_noise = 0


def cfg_noisy(e):
    e.n_food = 5
    e.n_boss_beasts = 0
    e.sensor_noise = 0.25


def cfg_complex(e):
    e.n_food = 8
    e.n_boss_beasts = 1
    e.sensor_noise = 0.15


# ===== 主测试 =====

if __name__ == "__main__":
    print("="*60)
    print("🧪 能力涌现测试")
    print("="*60)
    
    envs = [
        ("基准环境", cfg_base),
        ("捕食者环境", cfg_predator),
        ("噪声环境", cfg_noisy),
        ("复杂环境", cfg_complex),
    ]
    
    capabilities = {
        "觅食": (test_foraging, cfg_base),
        "逃避": (test_predator_avoidance, cfg_predator),
        "探索": (test_exploration, cfg_base),
        "通信": (test_communication, cfg_base),
        "决策": (test_decision_making, cfg_base),
    }
    
    all_results = {}
    
    # 训练并测试
    for env_name, env_cfg in envs:
        print(f"\n🌍 环境: {env_name}")
        print("-"*40)
        
        # 训练
        agent = train_agent(env_cfg, gens=50)
        
        # 节点分析
        node_types = {}
        for n in agent.genome.nodes.values():
            nt = n.node_type.name
            node_types[nt] = node_types.get(nt, 0) + 1
        
        print(f"  节点结构: {node_types}")
        
        # 测试各项能力
        for cap_name, (test_fn, test_env_cfg) in capabilities.items():
            results = test_capability(agent, test_fn, test_env_cfg)
            
            # 汇总结果
            if cap_name == "觅食":
                total_food = sum(r['food_eaten'] for r in results)
                total_dist = results[-1]['total_distance'] if results else 0
                print(f"    觅食: 食物={total_food}, 总移动={total_dist:.1f}")
                all_results[(env_name, cap_name)] = (total_food, total_dist)
            
            elif cap_name == "逃避":
                survived = sum(1 for r in results if r.get('survived', False))
                avg_dist = np.mean([r['nearest_beast'] for r in results if r['nearest_beast'] < 999])
                print(f"    逃避: 存活={survived}/{len(results)}, 最近={avg_dist:.1f}")
                all_results[(env_name, cap_name)] = (survived, avg_dist)
            
            elif cap_name == "探索":
                # 计算覆盖区域
                positions = [(r['x'], r['y']) for r in results]
                if positions:
                    x_range = max(p[0] for p in positions) - min(p[0] for p in positions)
                    y_range = max(p[1] for p in positions) - min(p[1] for p in positions)
                    coverage = x_range * y_range / 10000
                    print(f"    探索: 覆盖区域={coverage:.1f}%")
                    all_results[(env_name, cap_name)] = (coverage,)
            
            elif cap_name == "通信":
                has_comm = any(r['has_comm'] for r in results)
                comm_count = max((r['comm_nodes'] for r in results), default=0)
                print(f"    通信: 有COMM={'是' if has_comm else '否'}, 节点数={comm_count}")
                all_results[(env_name, cap_name)] = (has_comm, comm_count)
            
            elif cap_name == "决策":
                avg_complex = np.mean([r['decision_complexity'] for r in results]) if results else 0
                max_complex = max((r['decision_complexity'] for r in results), default=0)
                print(f"    决策: 平均复杂度={avg_complex:.1f}, 最大={max_complex}")
                all_results[(env_name, cap_name)] = (avg_complex, max_complex)
    
    # 总结
    print("\n" + "="*60)
    print("📊 能力对比总结")
    print("="*60)
    
    for cap_name, _ in capabilities.items():
        print(f"\n【{cap_name}】")
        for env_name, _ in envs:
            key = (env_name, cap_name)
            if key in all_results:
                val = all_results[key]
                if cap_name == "觅食":
                    print(f"  {env_name}: 食物={val[0]}, 移动={val[1]:.0f}")
                elif cap_name == "逃避":
                    print(f"  {env_name}: 存活={val[0]}, 距离={val[1]:.1f}")
                elif cap_name == "探索":
                    print(f"  {env_name}: 覆盖={val[0]:.1f}%")
                elif cap_name == "通信":
                    print(f"  {env_name}: COMM={'有' if val[0] else '无'}, 节点={val[1]}")
                elif cap_name == "决策":
                    print(f"  {env_name}: 复杂度={val[1]}")