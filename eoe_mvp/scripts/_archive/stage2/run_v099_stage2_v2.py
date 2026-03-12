#!/usr/bin/env python3
"""
v0.99 阶段二修复测试 - 第二轮调优

问题：能吃到食物但仍会死
解决：进一步降低代谢 + 增加初始能量
"""

import sys
import os
import json
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population


def run_v099_stage2_v2(
    n_generations: int = 100,
    population_size: int = 50,
    verbose: bool = True
) -> dict:
    """运行 v0.99 阶段二测试 - 第二轮"""
    
    print("=" * 70)
    print("  v0.99 阶段二测试 - 第二轮调优")
    print("=" * 70)
    
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    if not os.path.exists(brain_path):
        print(f"\n❌ 阶段一大脑不存在: {brain_path}")
        return {}
    
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    print(f"\n[配置 - 第二轮调优]")
    print(f"  代谢: α=0.003, β=0.003 (进一步降低)")
    print(f"  寿命: 100帧")
    print(f"  季节长度: 50帧 (夏天50帧 + 冬天50帧)")
    print(f"  冬天代谢: 1.1倍 (更温和)")
    print(f"  初始能量: 200 (增加)")
    print(f"  即时进食: True")
    
    pop = Population(
        population_size=population_size,
        elite_ratio=0.2,
        env_width=50.0,
        env_height=50.0,
        lifespan=100,
        n_food=20,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.003,  # 进一步降低
        metabolic_beta=0.003,
        seasonal_cycle=True,
        season_length=50,
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=1.1,  # 更温和
        immediate_eating=True,
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop._init_population()
    
    start_time = time.time()
    history = {
        'generation': [],
        'best_fitness': [],
        'total_food_eaten': [],
        'avg_nodes': [],
        'season': [],
        'alive': []
    }
    
    try:
        for gen in range(n_generations):
            stats = pop.epoch(verbose=False)
            pop.generation += 1
            
            total_eaten = sum(getattr(a, 'food_eaten', 0) for a in pop.agents)
            alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
            
            history['generation'].append(pop.generation)
            history['best_fitness'].append(stats['best_fitness'])
            history['total_food_eaten'].append(total_eaten)
            history['avg_nodes'].append(stats['avg_nodes'])
            history['season'].append(pop.environment.current_season)
            history['alive'].append(alive)
            
            if gen % 20 == 0:
                print(f"  Gen {gen:3d}: fit={stats['best_fitness']:8.1f}, "
                      f"eaten={total_eaten:3d}, season={pop.environment.current_season:6s}, "
                      f"alive={alive}/{population_size}")
                
    except KeyboardInterrupt:
        print("\n\n[中断]")
    
    elapsed = time.time() - start_time
    
    # 分析结果
    print("\n" + "=" * 70)
    print("  结果分析")
    print("=" * 70)
    
    avg_alive = np.mean(history['alive'][-20:])
    max_eaten = max(history['total_food_eaten'])
    final_season = history['season'][-1]
    
    print(f"\n  平均存活数: {avg_alive:.1f}/{population_size}")
    print(f"  最高进食数: {max_eaten}")
    print(f"  最终季节: {final_season}")
    
    # 按季节分析
    summer_alive = [a for a, s in zip(history['alive'], history['season']) if s == 'summer']
    winter_alive = [a for a, s in zip(history['alive'], history['season']) if s == 'winter']
    
    if summer_alive:
        print(f"  夏天平均存活: {np.mean(summer_alive[-10:]):.1f}")
    if winter_alive:
        print(f"  冬天平均存活: {np.mean(winter_alive[-10:]):.1f}")
    
    print(f"\n  运行时间: {elapsed:.1f}秒")
    
    return {'history': history, 'avg_alive': avg_alive}


if __name__ == '__main__':
    result = run_v099_stage2_v2(n_generations=100, population_size=50)
    sys.exit(0)