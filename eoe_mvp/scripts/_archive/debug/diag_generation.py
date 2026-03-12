#!/usr/bin/env python3
"""检查代际转换时的状态"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population

brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
with open(brain_path, 'r') as f:
    brain_data = json.load(f)

print("=" * 50)
print("代际转换诊断")
print("=" * 50)

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

# 第一代
print(f"\n[Gen 0 初始]")
print(f"  season={pop.environment.current_season}, frame={pop.environment.season_frame}")
print(f"  is_day={pop.environment.is_day}")
print(f"  Agent能量: {[a.internal_energy for a in pop.agents[:3]]}")

stats = pop.epoch(verbose=False)
pop.generation += 1

print(f"\n[Gen 0 结束]")
print(f"  season={pop.environment.current_season}, frame={pop.environment.season_frame}")
print(f"  存活: {sum(1 for a in pop.agents if a.is_alive)}")

# 此时应该触发重置
print(f"\n[重置后检查 - 在 epoch return 之前]")
# 手动检查重置是否发生
print(f"  season={pop.environment.current_season}, frame={pop.environment.season_frame}")

# 运行一代
stats = pop.epoch(verbose=False)
pop.generation += 1

print(f"\n[Gen 1 结束]")
print(f"  season={pop.environment.current_season}")
print(f"  Agent能量: {[a.internal_energy for a in pop.agents[:3]]}")
print(f"  存活: {sum(1 for a in pop.agents if a.is_alive)}")