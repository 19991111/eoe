#!/usr/bin/env python3
"""详细诊断每代的能量变化"""

import sys
import os
import json

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

print("=" * 60)
print("逐代能量诊断")
print("=" * 60)

for gen in range(5):
    # 记录初始能量
    initial_energy = [a.internal_energy for a in pop.agents]
    
    # 运行一代
    stats = pop.epoch(verbose=False)
    pop.generation += 1
    
    # 记录结束能量
    final_energy = [a.internal_energy for a in pop.agents]
    alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
    eaten = sum(a.food_eaten for a in pop.agents)
    
    print(f"\nGen {gen}:")
    print(f"  初始能量: min={min(initial_energy):.0f}, max={max(initial_energy):.0f}, avg={sum(initial_energy)/len(initial_energy):.0f}")
    print(f"  结束能量: min={min(final_energy):.0f}, max={max(final_energy):.0f}, avg={sum(final_energy)/len(final_energy):.0f}")
    print(f"  存活: {alive}/10, 进食: {eaten}")
    print(f"  季节: {pop.environment.current_season}, frame: {pop.environment.season_frame}")