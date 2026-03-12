#!/usr/bin/env python3
"""详细调试 - 追踪每代的父母选择"""

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
    lifespan=40,  # 缩短到 40 帧
    n_food=10,
    food_energy=30.0,
    respawn_food=True,
    metabolic_alpha=0.003,
    metabolic_beta=0.003,
    seasonal_cycle=True,
    season_length=20,  # 20帧夏天 + 20帧冬天
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

# 打印 Gen 0 结束后的状态
print("=" * 50)
print("Gen 0 结束后的 Agent 状态")
print("=" * 50)

sorted_agents = sorted(pop.agents, key=lambda a: a.fitness, reverse=True)
for i, a in enumerate(sorted_agents):
    print(f"{i}: alive={a.is_alive}, energy={a.internal_energy:.0f}, food_eaten={a.food_eaten}")

# 手动触发繁殖查看
print("\n" + "=" * 50)
print("模拟繁殖")
print("=" * 50)

# 重建精英选择逻辑
n_elites = int(10 * 0.2)
survivors_with_food = [a for a in sorted_agents if a.is_alive and a.food_eaten > 0]
survivors_with_food = sorted(survivors_with_food, key=lambda a: a.internal_energy, reverse=True)

print(f"survivors_with_food: {len(survivors_with_food)}")
for i, a in enumerate(survivors_with_food[:5]):
    print(f"  {i}: energy={a.internal_energy:.0f}, food_eaten={a.food_eaten}")

if survivors_with_food:
    elite_pool = survivors_with_food[:min(n_elites * 2, len(survivors_with_food))]
    print(f"\nelite_pool (前{len(elite_pool)}个):")
    for i, a in enumerate(elite_pool):
        print(f"  {i}: energy={a.internal_energy:.0f}, food_eaten={a.food_eaten}")
    
    # 计算子代初始能量
    parent = elite_pool[0]
    parent_food = parent.food_eaten
    child_energy = 150.0 + parent_food * 30.0
    print(f"\n子代初始能量计算:")
    print(f"  父亲食物数: {parent_food}")
    print(f"  子代初始能量: 150 + {parent_food}*30 = {child_energy}")