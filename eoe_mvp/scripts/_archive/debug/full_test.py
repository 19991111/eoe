#!/usr/bin/env python3
"""完整测试 - 验证即时进食 + 夏天存活"""

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
print("v0.99 完整测试 - 即时进食 + 跨代存活")
print("=" * 60)

pop = Population(
    population_size=20,
    elite_ratio=0.2,
    env_width=40.0,
    env_height=40.0,
    lifespan=60,  # 60帧 = 夏天30 + 冬天30
    n_food=15,
    food_energy=30.0,
    respawn_food=True,
    metabolic_alpha=0.003,
    metabolic_beta=0.003,
    seasonal_cycle=True,
    season_length=30,  # 30帧夏天 + 30帧冬天 = 60帧寿命
    winter_food_multiplier=0.0,
    winter_metabolic_multiplier=1.1,
    immediate_eating=True,
    use_champion=True,
    champion_brain=brain_data,
    pure_survival_mode=True
)

pop._init_population()

history = []

for gen in range(20):
    # 记录开始状态
    start_energy = [a.internal_energy for a in pop.agents]
    
    # 运行一代
    stats = pop.epoch(verbose=False)
    pop.generation += 1
    
    # 记录结束状态
    end_energy = [a.internal_energy for a in pop.agents]
    alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
    eaten = sum(a.food_eaten for a in pop.agents)
    
    history.append({
        'gen': gen,
        'start_energy_avg': np.mean(start_energy),
        'end_energy_avg': np.mean(end_energy),
        'alive': alive,
        'eaten': eaten
    })
    
    print(f"Gen {gen:2d}: start_energy={np.mean(start_energy):.0f}, "
          f"end_energy={np.mean(end_energy):.0f}, alive={alive}/20, eaten={eaten}")

# 分析结果
print("\n" + "=" * 60)
print("结果分析")
print("=" * 60)

# 检查有多少代有存活
generations_with_survivors = sum(1 for h in history if h['alive'] > 0)
print(f"\n有存活后代的代数: {generations_with_survivors}/20")

# 检查连续存活
max_consecutive = 0
current = 0
for h in history:
    if h['alive'] > 0:
        current += 1
        max_consecutive = max(max_consecutive, current)
    else:
        current = 0

print(f"最长连续存活代数: {max_consecutive}")

# 检查是否有进化（进食数增加）
eaten_history = [h['eaten'] for h in history]
print(f"进食数范围: {min(eaten_history)} - {max(eaten_history)}")

if max_consecutive >= 5:
    print("\n✅ 成功！Agent 可以跨代存活！")
elif generations_with_survivors > 0:
    print("\n⚠️ 部分成功 - 有存活但不稳定")
else:
    print("\n❌ 失败 - 仍无法存活")