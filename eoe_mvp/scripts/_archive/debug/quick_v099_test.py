#!/usr/bin/env python3
"""
v0.99 快速验证测试 - 10代快速检查
"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population

# 加载大脑
brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
with open(brain_path, 'r') as f:
    brain_data = json.load(f)

print("=" * 50)
print("v0.99 快速验证 - 即时进食")
print("=" * 50)

# 最小配置测试
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
    season_length=25,  # 25帧夏天 + 25帧冬天 = 50帧寿命
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=1.1,
    immediate_eating=True,
    use_champion=True,
    champion_brain=brain_data,
    pure_survival_mode=True
)

pop._init_population()

print(f"\n[初始] season={pop.environment.current_season}, frame={pop.environment.season_frame}")

for gen in range(10):
    stats = pop.epoch(verbose=False)
    pop.generation += 1
    
    total_eaten = sum(getattr(a, 'food_eaten', 0) for a in pop.agents)
    alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
    
    print(f"Gen {gen}: eaten={total_eaten}, alive={alive}/10, season={pop.environment.current_season}")

print("\n✅ 完成")