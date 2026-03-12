#!/usr/bin/env python3
"""检查 season_frame 是否在 step() 中更新"""

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
    population_size=5,
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

print(f"[初始] seasonal_cycle={pop.environment.seasonal_cycle}")
print(f"[初始] season_length={pop.environment.season_length}")
print(f"[初始] season_frame={pop.environment.season_frame}")

# 运行 30 帧
for i in range(30):
    pop.environment.step()
    if i % 10 == 9:
        print(f"Frame {i+1}: season={pop.environment.current_season}, frame={pop.environment.season_frame}")

print(f"\n[30帧后] season={pop.environment.current_season}, frame={pop.environment.season_frame}")