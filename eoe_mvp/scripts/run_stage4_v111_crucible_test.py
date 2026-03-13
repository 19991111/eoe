#!/usr/bin/env python3
"""
v0.0 压力梯度熔炉测试
使用文明溢价、蟑螂惩罚、逆境加成机制选择真冠军
"""
import sys
sys.path.insert(0, '.')
import json
import os
from core.eoe.population import Population

os.makedirs('champions', exist_ok=True)

# 加载R10冠军作为起点
with open('champions/stage4_v110_r10.json') as f:
    brain = json.load(f)

print("="*60)
print("v0.0 压力梯度熔炉测试")
print("="*60)

# 降低参数压力以加快测试
pop = Population(
    population_size=20,
    elite_ratio=0.3,
    env_width=40.0,
    env_height=40.0,
    lifespan=30,
    n_food=10,
    food_energy=80.0,
    seasonal_cycle=True,
    season_length=20,
    # v11.0 机制 - 降低压力
    energy_decay_k=0.00005,  # 减半
    port_interference_gamma=1.5,  # 降低
    season_jitter=0.08,  # 降低
    nest_tax=0.08,  # 降低
    use_champion=True,
    champion_brain=brain
)

# 启用阶段二机制
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

print(f"初始种群: {pop.population_size} Agent")

# 快速迭代5轮
best_overall = None
best_overall_fitness = 0

for round_num in range(1, 6):
    print(f"\n{'='*60}")
    print(f"第 {round_num} 轮演化")
    print('='*60)
    
    pop.generation = 0  # 重置代数
    
    # 运行10代
    best_fit = 0
    best_brain = None
    
    for gen in range(10):
        pop.environment.apply_dynamic_pressure(pop.generation)
        stats = pop.epoch(verbose=False)
        
        alive = [a for a in pop.environment.agents if a.is_alive]
        if not alive:
            print(f"R{round_num} Gen{gen}: 全部死亡")
            break
        
        # 使用压力梯度熔炉选择真冠军
        true_champion = pop.select_true_champion(alive)
        total_stored = sum(a.food_stored for a in alive)
        
        # 更新英雄冢
        pop.update_hall_of_fame(true_champion)
        
        # 使用真冠军的适应度
        gen_best = max(alive, key=lambda a: a.fitness)
        if gen_best.fitness > best_fit:
            best_fit = gen_best.fitness
            best_brain = gen_best.genome.to_dict()
        
        if gen % 5 == 0:
            pressure = pop._calculate_environmental_pressure()
            complexity = pop._calculate_complexity_score(true_champion)
            print(f"R{round_num} Gen{gen}: 存活{len(alive):2d} | 真冠军={true_champion.fitness:7.1f} | "
                  f"Stored={total_stored:3d} | 压力={pressure:.2f} | 复杂度={complexity:.2f}")
        
        if len(alive) < 2:
            break
        
        pop.reproduce(verbose=False)
        pop.generation += 1
    
    # 保存本轮冠军
    if best_brain:
        fname = f'champions/stage4_v111_r{round_num}.json'
        with open(fname, 'w') as f:
            json.dump(best_brain, f, indent=2)
        print(f"  → 保存冠军: {fname}")
        
        if best_fit > best_overall_fitness:
            best_overall_fitness = best_fit
            best_overall = best_brain

# 最终结果
print("\n" + "="*60)
print("v0.0 熔炉测试完成")
print("="*60)
print(f"最高适应度: {best_overall_fitness:.1f}")
print(f"英雄冢大小: {len(pop.hall_of_fame)}")

if pop.hall_of_fame:
    print("\n英雄冢成员:")
    for i, hof in enumerate(sorted(pop.hall_of_fame, key=lambda x: -x['fitness'])):
        print(f"  {i+1}. Gen{hou['generation']}: fitness={hou['fitness']:.1f}, pressure={hou['pressure']:.2f}")

# 保存最终冠军
if best_overall:
    with open('champions/stage4_v111_final.json', 'w') as f:
        json.dump(best_overall, f, indent=2)
    print(f"\n最终冠军: champions/stage4_v111_final.json")