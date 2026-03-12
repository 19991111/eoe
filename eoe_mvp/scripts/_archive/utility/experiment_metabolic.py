#!/usr/bin/env python3
"""
实验：降低代谢对冬天存活的影响

假设：代谢过高是冬天死亡的主因
验证：降低代谢后 Agent 能否存活过冬
"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population


def experiment_metabolic_impact():
    """实验：代谢对冬天存活的影响"""
    
    # 加载阶段一的大脑
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    print("=" * 70)
    print("  实验：不同代谢水平下的冬天存活测试")
    print("=" * 70)
    
    results = []
    
    for metabolic_level, (alpha, beta, name) in enumerate([
        (0.02, 0.02, "低代谢(0.02)"),
        (0.01, 0.01, "超低代谢(0.01)"),
        (0.005, 0.005, "极低代谢(0.005)"),
    ]):
        print(f"\n[{name}]")
        
        pop = Population(
            population_size=10,
            elite_ratio=0.2,
            env_width=40.0,
            env_height=40.0,
            lifespan=400,
            n_food=15,
            food_energy=30.0,
            respawn_food=True,
            metabolic_alpha=alpha,
            metabolic_beta=beta,
            seasonal_cycle=True,
            season_length=40,
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.2,  # 温和的冬天惩罚
            use_champion=True,
            champion_brain=brain_data,
            pure_survival_mode=True
        )
        
        pop._init_population()
        
        # 运行 100 帧观察
        survived_frames = []
        total_eaten = []
        total_stored = []
        
        for frame in range(100):
            # 执行一代（这里简化为单步）
            pop.environment.step()
            
            # 统计存活
            alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
            survived_frames.append(alive)
            total_eaten.append(sum(a.food_eaten for a in pop.agents))
            total_stored.append(sum(a.food_stored for a in pop.agents))
        
        avg_survived = np.mean(survived_frames[-10:])
        max_eaten = max(total_eaten)
        max_stored = max(total_stored)
        
        print(f"  存活数: {avg_survived:.1f}/10 (最后10帧平均)")
        print(f"  进食数: {max_eaten}")
        print(f"  贮粮数: {max_stored}")
        
        results.append({
            'name': name,
            'alpha': alpha,
            'beta': beta,
            'avg_survived': avg_survived,
            'max_eaten': max_eaten,
            'max_stored': max_stored
        })
    
    print("\n" + "=" * 70)
    print("  实验结论")
    print("=" * 70)
    
    best = max(results, key=lambda x: x['avg_survived'])
    print(f"\n  最佳代谢水平: {best['name']}")
    print(f"  存活率: {best['avg_survived']*10:.0f}%")
    
    if best['avg_survived'] > 0:
        print(f"\n  ✅ 降低代谢可以提高冬天存活率!")
        print(f"  建议: 使用 metabolic_alpha={best['alpha']}, beta={best['beta']}")
    else:
        print(f"\n  ❌ 即使降低代谢也无法存活")
        print(f"  需要其他机制: 增加初始能量、改进进食机制等")


def experiment_initial_energy():
    """实验：初始能量的影响"""
    
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    print("\n" + "=" * 70)
    print("  实验：初始能量对冬天存活的影响")
    print("=" * 70)
    
    # 使用低代谢参数
    for initial_energy in [150, 300, 500]:
        print(f"\n[初始能量: {initial_energy}]")
        
        pop = Population(
            population_size=5,
            elite_ratio=0.2,
            env_width=40.0,
            env_height=40.0,
            lifespan=500,
            n_food=15,
            food_energy=30.0,
            respawn_food=True,
            metabolic_alpha=0.01,
            metabolic_beta=0.01,
            seasonal_cycle=True,
            season_length=40,
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.2,
            initial_energy=initial_energy,  # 如果Population支持
            use_champion=True,
            champion_brain=brain_data,
            pure_survival_mode=True
        )
        
        pop._init_population()
        
        # 检查初始能量
        init_e = pop.agents[0].internal_energy
        print(f"  实际初始能量: {init_e}")
        
        # 运行观察
        for frame in range(100):
            pop.environment.step()
        
        alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
        print(f"  100帧后存活: {alive}/5")


if __name__ == '__main__':
    experiment_metabolic_impact()
    # experiment_initial_energy()  # 可能需要修改 Population 支持