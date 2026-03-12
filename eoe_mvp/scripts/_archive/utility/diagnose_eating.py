#!/usr/bin/env python3
"""
详细诊断：为什么 Agent 无法进食？

逐帧分析进食逻辑
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population
from core.eoe.environment import Environment


def diagnose_eating():
    """诊断进食问题"""
    
    print("=" * 70)
    print("  进食行为诊断")
    print("=" * 70)
    
    # 加载阶段一的大脑
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    # 创建测试种群
    pop = Population(
        population_size=5,
        elite_ratio=0.2,
        env_width=30.0,  # 更小的世界
        env_height=30.0,
        lifespan=100,
        n_food=10,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        seasonal_cycle=False,  # 先关闭季节
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop._init_population()
    test_agent = pop.agents[0]
    env = pop.environment
    
    print(f"\n[环境状态]")
    print(f"  巢穴模式: {env.nest_enabled}")
    print(f"  食物数量: {len(env.food_positions)}")
    print(f"  食物位置: {env.food_positions[:3]}...")
    
    print(f"\n[Agent 初始状态]")
    print(f"  位置: ({test_agent.x:.1f}, {test_agent.y:.1f})")
    print(f"  能量: {test_agent.internal_energy:.1f}")
    print(f"  携带: {test_agent.food_carried}")
    
    # 寻找最近的食物
    min_dist = float('inf')
    nearest_food = None
    for i, (fx, fy) in enumerate(env.food_positions):
        dist = ((test_agent.x - fx)**2 + (test_agent.y - fy)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_food = (fx, fy, dist, i)
    
    print(f"\n[最近食物]")
    print(f"  位置: ({nearest_food[0]:.1f}, {nearest_food[1]:.1f})")
    print(f"  距离: {nearest_food[2]:.1f}")
    print(f"  索引: {nearest_food[3]}")
    print(f"  进食阈值: 3.0")
    
    # 手动移动 Agent 到食物旁边
    print(f"\n[移动 Agent 到食物旁边]")
    fx, fy = nearest_food[0], nearest_food[1]
    test_agent.x = fx - 2.0  # 距离 2 小于阈值 3
    test_agent.y = fy
    
    print(f"  新位置: ({test_agent.x:.1f}, {test_agent.y:.1f})")
    print(f"  到食物距离: 2.0")
    
    # 尝试进食
    print(f"\n[尝试进食]")
    energy_before = test_agent.internal_energy
    carried_before = test_agent.food_carried
    eaten_before = getattr(test_agent, 'food_eaten', 0)
    
    # 手动调用进食逻辑
    result = env._check_food_collision(test_agent)
    
    energy_after = test_agent.internal_energy
    carried_after = test_agent.food_carried
    eaten_after = getattr(test_agent, 'food_eaten', 0)
    
    print(f"  进食结果: {result}")
    print(f"  能量变化: {energy_before:.1f} -> {energy_after:.1f} (差: {energy_after-energy_before:.1f})")
    print(f"  携带变化: {carried_before} -> {carried_after}")
    print(f"  进食计数: {eaten_before} -> {eaten_after}")
    
    if energy_after > energy_before:
        print(f"\n  ✅ 进食成功！能量增加了 {energy_after-energy_before:.1f}")
    else:
        print(f"\n  ❌ 进食失败！能量没有增加")
    
    # 现在测试巢穴模式
    print(f"\n" + "=" * 70)
    print("  对比测试: 巢穴模式 vs 非巢穴模式")
    print("=" * 70)
    
    # 重新创建 - 不带巢穴
    pop2 = Population(
        population_size=5,
        elite_ratio=0.2,
        env_width=30.0,
        env_height=30.0,
        lifespan=100,
        n_food=10,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        seasonal_cycle=False,
        nest_enabled=False,  # 显式关闭巢穴
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop2._init_population()
    test_agent2 = pop2.agents[0]
    env2 = pop2.environment
    
    print(f"\n[非巢穴模式]")
    print(f"  巢穴模式: {env2.nest_enabled}")
    
    # 移动到食物旁边
    for i, (fx, fy) in enumerate(env2.food_positions):
        dist = ((test_agent2.x - fx)**2 + (test_agent2.y - fy)**2)**0.5
        if dist < 10:
            test_agent2.x = fx - 2.0
            test_agent2.y = fy
            break
    
    energy_before2 = test_agent2.internal_energy
    result2 = env2._attempt_eat(test_agent2)
    energy_after2 = test_agent2.internal_energy
    
    print(f"  能量变化: {energy_before2:.1f} -> {energy_after2:.1f} (差: {energy_after2-energy_before2:.1f})")
    print(f"  进食计数: {getattr(test_agent2, 'food_eaten', 0)}")
    
    if energy_after2 > energy_before2:
        print(f"\n  ✅ 非巢穴模式进食成功！")
    else:
        print(f"\n  ❌ 非巢穴模式进食也失败！")


if __name__ == '__main__':
    diagnose_eating()