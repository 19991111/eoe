#!/usr/bin/env python3
"""正确的方式：使用 run 函数"""

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

print("=" * 60)
print("v0.99 正确测试 - 使用 run 函数")
print("=" * 60)

pop = Population(
    population_size=20,
    elite_ratio=0.2,
    env_width=40.0,
    env_height=40.0,
    lifespan=60,
    n_food=15,
    food_energy=30.0,
    respawn_food=True,
    metabolic_alpha=0.003,
    metabolic_beta=0.003,
    seasonal_cycle=True,
    season_length=30,
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=1.1,
    immediate_eating=True,
    use_champion=True,
    champion_brain=brain_data,
    pure_survival_mode=True
)

pop._init_population()

# 使用 run 函数 - 这会正确调用 reproduce
history = pop.run(n_generations=20, verbose=False)

print("\n" + "=" * 60)
print("结果分析")
print("=" * 60)

# 分析结果
best_fitness = [h['best_fitness'] for h in history]
avg_fitness = [h['avg_fitness'] for h in history]

print(f"\n最佳适应度: {max(best_fitness):.0f} (Gen {np.argmax(best_fitness)})")
print(f"平均适应度趋势: {avg_fitness[0]:.0f} -> {avg_fitness[-1]:.0f}")