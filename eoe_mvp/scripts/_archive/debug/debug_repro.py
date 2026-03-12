#!/usr/bin/env python3
"""调试繁殖逻辑"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population

brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
with open(brain_path, 'r') as f:
    brain_data = json.load(f)

pop = Population(
    population_size=10,
    elite_ratio=0.2,
    env_width=30.0,
    env_height=30.0,
    lifespan=50,
    n_food=10,
    food_energy=30.0,
    respawn_food=True,
    metabolic_alpha=0.003,
    metabolic_beta=0.003,
    seasonal_cycle=True,
    season_length=25,
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=1.1,
    immediate_eating=True,
    use_champion=True,
    champion_brain=brain_data,
    pure_survival_mode=True
)

pop._init_population()

# Gen 0
stats = pop.epoch(verbose=False)
pop.generation += 1

print("Gen 0 结束后的精英:")
# 打印精英的食物数和能量
sorted_agents = sorted(pop.agents, key=lambda a: a.fitness, reverse=True)
for i, a in enumerate(sorted_agents[:5]):
    print(f"  {i}: fitness={a.fitness:.0f}, food_eaten={a.food_eaten}, energy={a.internal_energy:.0f}, alive={a.is_alive}")

# 检查 reproduce 之前的逻辑
print("\n模拟繁殖选择:")
survivors_with_food = [a for a in sorted_agents if a.is_alive and a.food_eaten > 0]
print(f"  survivors_with_food: {len(survivors_with_food)}")
if survivors_with_food:
    print(f"  第一个: food_eaten={survivors_with_food[0].food_eaten}, energy={survivors_with_food[0].internal_energy}")