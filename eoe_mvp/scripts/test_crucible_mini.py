#!/usr/bin/env python3
"""极简熔炉测试"""
import sys
sys.path.insert(0, '.')
import json
import os
from core.eoe.population import Population

os.makedirs('champions', exist_ok=True)

# 加载R10冠军
with open('champions/stage4_v110_r10.json') as f:
    brain = json.load(f)

print("v11.1 熔炉测试 (5代)")

pop = Population(
    population_size=10,
    elite_ratio=0.4,
    env_width=30.0,
    env_height=30.0,
    lifespan=15,
    n_food=5,
    food_energy=80.0,
    seasonal_cycle=True,
    season_length=10,
    energy_decay_k=0.00005,
    port_interference_gamma=1.5,
    season_jitter=0.05,
    nest_tax=0.05,
    use_champion=True,
    champion_brain=brain
)

pop.environment.enable_fatigue_system(
    enabled=True,
    max_fatigue=50.0,
    fatigue_build_rate=0.15,
    sleep_danger_prob=0.95,
    enable_wakeup_hunger=True,
    enable_sleep_drop=True
)

pop.environment.enable_thermal_sanctuary(
    enabled=True,
    summer_temp=28.0,
    winter_temp=-10.0,
    food_heat=15.0,
    nest_insulation=0.02
)

best_brain = None
best_fit = 0

for gen in range(5):
    pop.environment.apply_dynamic_pressure(pop.generation)
    stats = pop.epoch(verbose=False)
    
    alive = [a for a in pop.environment.agents if a.is_alive]
    if not alive:
        print(f"Gen{gen}: 全部死亡")
        break
    
    true_champ = pop.select_true_champion(alive)
    total_stored = sum(a.food_stored for a in alive)
    pressure = pop._calculate_environmental_pressure()
    complexity = pop._calculate_complexity_score(true_champ)
    
    gen_best = max(alive, key=lambda a: a.fitness)
    if gen_best.fitness > best_fit:
        best_fit = gen_best.fitness
        best_brain = gen_best.genome.to_dict()
    
    print(f"Gen{gen}: 存活{len(alive):2d} | 真冠军={true_champ.fitness:7.1f} | Stored={total_stored:3d} | 压力={pressure:.2f}")
    
    pop.update_hall_of_fame(true_champ)
    
    if len(alive) < 2:
        break
    
    pop.reproduce(verbose=False)
    pop.generation += 1

print(f"\n最高适应度: {best_fit:.1f}")
print(f"英雄冢: {len(pop.hall_of_fame)}人")

if best_brain:
    with open('champions/stage4_v111_mini.json', 'w') as f:
        json.dump(best_brain, f, indent=2)
    print("已保存: champions/stage4_v111_mini.json")