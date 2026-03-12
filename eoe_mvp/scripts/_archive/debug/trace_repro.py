#!/usr/bin/env python3
"""打印繁殖时的实际父母信息 - 更详细"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population

# 添加调试：在 reproduce 函数中打印信息
original_reproduce = Population.reproduce

def debug_reproduce(self, verbose: bool = False):
    # 打印调试信息
    print(f"\n[DEBUG reproduce 被调用]")
    print(f"  Agent 数量: {len(self.agents)}")
    
    alive_agents = [a for a in self.agents if a.is_alive]
    print(f"  活着的 Agent: {len(alive_agents)}")
    
    fitnesses = [a.fitness for a in self.agents]
    sorted_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
    
    print(f"  前3个按 fitness 排序: fitness={[a.fitness for a in sorted_agents[:3]]}")
    print(f"  前3个 alive: {[a.is_alive for a in sorted_agents[:3]]}")
    print(f"  前3个 energy: {[a.internal_energy for a in sorted_agents[:3]]}")
    print(f"  前3个 food_eaten: {[a.food_eaten for a in sorted_agents[:3]]}")
    
    survivors_with_food = [a for a in sorted_agents if a.is_alive and a.food_eaten > 0]
    print(f"  survivors_with_food: {len(survivors_with_food)}")
    
    if survivors_with_food:
        survivors_with_food = sorted(survivors_with_food, key=lambda a: a.internal_energy, reverse=True)
        print(f"  第一个精英: food_eaten={survivors_with_food[0].food_eaten}, energy={survivors_with_food[0].internal_energy:.0f}")
    else:
        print(f"  没有 survivors_with_food!")
    
    # 调用原始函数
    return original_reproduce(self, verbose)

Population.reproduce = debug_reproduce

# Now run test
brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
with open(brain_path, 'r') as f:
    brain_data = json.load(f)

print("=" * 60)
print("测试 - 追踪繁殖")
print("=" * 60)

pop = Population(
    population_size=10,
    elite_ratio=0.2,
    env_width=30.0,
    env_height=30.0,
    lifespan=40,
    n_food=10,
    food_energy=30.0,
    respawn_food=True,
    metabolic_alpha=0.003,
    metabolic_beta=0.003,
    seasonal_cycle=True,
    season_length=20,
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=1.1,
    immediate_eating=True,
    use_champion=True,
    champion_brain=brain_data,
    pure_survival_mode=True
)

pop._init_population()

# Gen 0
print("\n--- Gen 0 ---")
stats = pop.epoch(verbose=False)
pop.generation += 1
print(f"Gen 0 结束: alive={sum(1 for a in pop.agents if a.is_alive)}, eaten={sum(a.food_eaten for a in pop.agents)}")

# Gen 1 - 这里会触发繁殖
print("\n--- Gen 1 ---")
stats = pop.epoch(verbose=False)
pop.generation += 1

print(f"\n结果: 子代初始能量 = {[a.internal_energy for a in pop.agents[:3]]}")