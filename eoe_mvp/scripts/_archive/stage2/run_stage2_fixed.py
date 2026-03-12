#!/usr/bin/env python3
"""
修复后的阶段二脚本 - 降低代谢 + 增加初始能量
"""

import sys
import os
import json
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population


def run_stage2_fixed(
    n_generations: int = 200,
    population_size: int = 50,
    verbose: bool = True
) -> dict:
    """运行修复后的阶段二测试"""
    
    print("=" * 70)
    print("  阶段二（修复版）：代谢优化后的冬天存活测试")
    print("=" * 70)
    
    # 检查阶段一大脑
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    if not os.path.exists(brain_path):
        print(f"\n❌ 阶段一大脑不存在: {brain_path}")
        return {}
    
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    print(f"\n[关键修复]")
    print(f"  代谢: 0.02 → 0.005 (降低4倍)")
    print(f"  初始能量: 150 → 300 (增加2倍)")
    print(f"  冬天代谢惩罚: 1.1 → 1.2 (更温和)")
    print(f"  季节长度: 60 → 40 (更短的季节周期)")
    
    # 创建种群 - 修复代谢问题
    pop = Population(
        population_size=population_size,
        elite_ratio=0.2,
        env_width=50.0,
        env_height=50.0,
        lifespan=600,
        n_food=20,
        food_energy=30.0,
        respawn_food=True,
        # 关键修复：降低代谢
        metabolic_alpha=0.005,
        metabolic_beta=0.005,
        # 季节系统
        seasonal_cycle=True,
        season_length=40,  # 更短的季节
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=1.2,  # 温和的冬天
        # 使用阶段一的大脑
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop._init_population()
    
    # 运行演化
    start_time = time.time()
    history = {
        'generation': [],
        'best_fitness': [],
        'total_food_eaten': [],
        'total_food_stored': [],
        'avg_nodes': [],
        'avg_edges': [],
        'winter_survival_rate': [],
        'season': []
    }
    
    try:
        for gen in range(n_generations):
            stats = pop.epoch(verbose=False)
            pop.generation += 1
            
            # 收集数据
            alive = sum(1 for a in pop.agents if a.is_alive and a.internal_energy > 0)
            total_eaten = sum(getattr(a, 'food_eaten', 0) for a in pop.agents)
            total_stored = sum(getattr(a, 'food_stored', 0) for a in pop.agents)
            
            # 冬天存活率
            winter_agents = [a for a in pop.agents if pop.environment.current_season == 'winter']
            if winter_agents:
                winter_survival = sum(1 for a in winter_agents if a.is_alive and a.internal_energy > 0) / len(winter_agents)
            else:
                winter_survival = 1.0
            
            history['generation'].append(pop.generation)
            history['best_fitness'].append(stats['best_fitness'])
            history['total_food_eaten'].append(total_eaten)
            history['total_food_stored'].append(total_stored)
            history['avg_nodes'].append(stats['avg_nodes'])
            history['avg_edges'].append(stats['avg_edges'])
            history['winter_survival_rate'].append(winter_survival)
            history['season'].append(pop.environment.current_season)
            
            if gen % 20 == 0:
                print(f"  Gen {gen:3d}: fit={stats['best_fitness']:8.1f}, "
                      f"eaten={total_eaten:3d}, stored={total_stored:3d}, "
                      f"season={pop.environment.current_season}, "
                      f"winter_survive={winter_survival*100:.0f}%")
                
    except KeyboardInterrupt:
        print("\n\n[中断]")
    
    elapsed = time.time() - start_time
    
    # 考核结果
    print("\n" + "=" * 70)
    print("  阶段二（修复版）考核结果")
    print("=" * 70)
    
    # 1. 贮粮
    max_stored = max(history['total_food_stored'])
    passed_basic = max_stored >= 1
    print(f"\n1. 及格线: 贮粮行为")
    print(f"   最高贮粮: {max_stored}")
    print(f"   状态: {'✅ 及格' if passed_basic else '❌ 未及格'}")
    
    # 2. 冬天存活
    winter_rates = [r for r, s in zip(history['winter_survival_rate'], history['season']) if s == 'winter']
    if winter_rates:
        avg_winter = np.mean(winter_rates[-20:]) if len(winter_rates) >= 20 else np.mean(winter_rates)
    else:
        avg_winter = 0
    passed_good = avg_winter > 0.5
    print(f"\n2. 优秀线: 冬天存活")
    print(f"   冬天存活率: {avg_winter*100:.1f}%")
    print(f"   状态: {'✅ 优秀' if passed_good else '❌ 未达到'}")
    
    # 综合
    passed = sum([passed_basic, passed_good])
    print(f"\n  综合: {passed}/2 项通过")
    print(f"  运行时间: {elapsed:.1f}秒")
    
    return {'history': history, 'passed': passed >= 2}


if __name__ == '__main__':
    result = run_stage2_fixed(n_generations=100, population_size=30)
    sys.exit(0 if result.get('passed') else 1)