#!/usr/bin/env python3
"""
阶段二：哺乳动物时期 - 跨越时间的演化

核心目标：逼迫种群演化出"贮粮（Hoarding）"行为

开启机制:
- 季节循环 (season_length=40)
- 冬季代谢惩罚 (winter_metabolic_multiplier=1.2)
- 巢穴系统 (物理存放点)

关闭机制:
- 疲劳系统 (无 enable_fatigue_system)
- 热力学庇护所 (无 enable_thermal_sanctuary)
- LLM Demiurge Loop

初始化: 使用阶段一大脑 (use_champion=True)

阶段二考核里程碑:
1. 及格线: 贮粮次数突破个位数 (1~5次)
2. 优秀线: 冬天死亡比例下降，寿命跨越两个季节
3. 满分线: 贮粮100+次，大脑复杂度质变
"""

import sys
import os
import json
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population
from core.eoe.environment import Environment


def run_stage2_test(
    n_generations: int = 200,
    population_size: int = 50,
    verbose: bool = True
) -> dict:
    """运行阶段二测试"""
    
    print("=" * 70)
    print("  阶段二：哺乳动物时期 - 季节与贮粮测试")
    print("=" * 70)
    print("\n[配置]")
    print(f"  种群大小: {population_size}")
    print(f"  演化代数: {n_generations}")
    print(f"  代谢: Alpha=0.02, Beta=0.02")
    print(f"  食物能量: 30.0")
    print(f"  寿命: 600帧 (跨越多个季节)")
    print(f"  季节长度: 60帧 (更长的夏天)")
    print(f"  冬季代谢: 1.1倍 (温和梯度)")
    print(f"  巢穴系统: 开启")
    print(f"  阶段一大脑初始化: champions/stage1_best_brain.json")
    print(f"  疲劳系统: 关闭")
    print(f"  热力学庇护所: 关闭")
    print(f"  LLM Demiurge: 关闭")
    print("=" * 70)
    
    # 检查阶段一大脑是否存在
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    if not os.path.exists(brain_path):
        print(f"\n❌ 阶段一大脑不存在: {brain_path}")
        print("   请先完成阶段一测试")
        return {}
    
    # 加载阶段一大脑
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    print(f"\n[已加载阶段一大脑]")
    print(f"  节点: {len(brain_data['nodes'])}")
    print(f"  边: {len(brain_data['edges'])}")
    
    # 创建种群 - 使用阶段一的大脑初始化
    pop = Population(
        population_size=population_size,
        elite_ratio=0.2,
        env_width=60.0,       # 较小世界，减少探索难度
        env_height=60.0,
        lifespan=600,         # 更长寿命，跨越多个季节
        n_food=25,            # 更多食物
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        # 阶段二关键: 季节系统 (梯度设置)
        seasonal_cycle=True,
        season_length=60,     # 更长的夏天 (60帧)
        winter_food_multiplier=0.0,  # 冬天无食物
        winter_metabolic_multiplier=1.1,  # 非常温和的冬天惩罚 (1.1倍)
        # 使用阶段一的大脑
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    # 初始化种群
    pop._init_population()
    
    print(f"\n[初始状态]")
    print(f"  Agent数量: {len(pop.agents)}")
    print(f"  当前季节: {pop.current_season}")
    
    # 运行演化
    start_time = time.time()
    history = {
        'generation': [],
        'best_fitness': [],
        'total_food_eaten': [],
        'total_food_stored': [],
        'avg_nodes': [],
        'avg_edges': [],
        'winter_death_rate': [],
        'season': []
    }
    
    try:
        for gen in range(n_generations):
            stats = pop.epoch(verbose=False)
            pop.generation += 1
            
            # 收集数据
            total_eaten = sum(getattr(a, 'food_eaten', 0) for a in pop.agents)
            total_stored = sum(getattr(a, 'food_stored', 0) for a in pop.agents)
            
            # 冬天死亡统计
            winter_deaths = sum(1 for a in pop.agents 
                              if not a.is_alive and a.internal_energy <= 0 
                              and pop.current_season == 'winter')
            total_dead = sum(1 for a in pop.agents if not a.is_alive)
            winter_death_rate = winter_deaths / max(1, total_dead)
            
            history['generation'].append(pop.generation)
            history['best_fitness'].append(stats['best_fitness'])
            history['total_food_eaten'].append(total_eaten)
            history['total_food_stored'].append(total_stored)
            history['avg_nodes'].append(stats['avg_nodes'])
            history['avg_edges'].append(stats['avg_edges'])
            history['winter_death_rate'].append(winter_death_rate)
            history['season'].append(pop.current_season)
            
            if gen % 20 == 0:
                print(f"  Gen {gen:3d}: fit={stats['best_fitness']:8.1f}, "
                      f"eaten={total_eaten:3d}, stored={total_stored:3d}, "
                      f"nodes={stats['avg_nodes']:.0f}, season={pop.current_season}")
                
    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动停止")
    
    elapsed = time.time() - start_time
    
    # ==================== 阶段二考核 ====================
    print("\n" + "=" * 70)
    print("  📊 阶段二考核结果")
    print("=" * 70)
    
    # 1. 及格线: 贮粮次数突破个位数
    last_20_stored = history['total_food_stored'][-20:]
    max_stored = max(last_20_stored) if last_20_stored else 0
    passed_basic = max_stored >= 1  # 至少1次贮粮
    print(f"\n1. 及格线: 贮粮行为出现")
    print(f"   最高贮粮次数: {max_stored}")
    print(f"   状态: {'✅ 及格' if passed_basic else '❌ 未及格'} (阈值≥1)")
    
    # 2. 优秀线: 冬天死亡比例下降
    last_10_winter_deaths = [history['winter_death_rate'][i] 
                             for i in range(-10, 0) 
                             if history['season'][i] == 'winter']
    if last_10_winter_deaths:
        avg_winter_death = np.mean(last_10_winter_deaths)
        passed_good = avg_winter_death < 0.8  # 低于80%
    else:
        avg_winter_death = 1.0
        passed_good = False
    print(f"\n2. 优秀线: 冬天幸存者")
    print(f"   近期冬天死亡率: {avg_winter_death*100:.1f}%")
    print(f"   状态: {'✅ 优秀' if passed_good else '❌ 未达到'} (阈值<80%)")
    
    # 3. 满分线: 贮粮100+次 + 大脑复杂度质变
    total_stored_final = sum(history['total_food_stored'])
    final_avg_nodes = history['avg_nodes'][-1]
    final_avg_edges = history['avg_edges'][-1]
    passed_perfect = (total_stored_final >= 100 and 
                     final_avg_nodes > 15)
    print(f"\n3. 满分线: 农耕文明")
    print(f"   累计贮粮次数: {total_stored_final}")
    print(f"   最终平均节点: {final_avg_nodes:.1f}")
    print(f"   最终平均边数: {final_avg_edges:.1f}")
    print(f"   状态: {'✅ 满分' if passed_perfect else '❌ 未达到'} (贮粮≥100 且 节点>15)")
    
    # 综合判定
    passed = sum([passed_basic, passed_good, passed_perfect])
    print("\n" + "=" * 70)
    print(f"  综合判定: {passed}/3 项通过")
    print("=" * 70)
    
    if passed >= 2:
        print("\n🎉 恭喜！种群已具备哺乳动物智能！")
        # 保存阶段二最佳大脑
        best_idx = np.argmax([a.fitness for a in pop.agents])
        best_agent = pop.agents[best_idx]
        
        os.makedirs('champions', exist_ok=True)
        brain_data = best_agent.genome.to_dict()
        with open('champions/stage2_best_brain.json', 'w') as f:
            json.dump(brain_data, f, indent=2)
        print(f"\n💾 阶段二最佳大脑已保存: champions/stage2_best_brain.json")
        print(f"   适应度: {best_agent.fitness:.1f}")
        print(f"   贮粮数: {getattr(best_agent, 'food_stored', 0)}")
    else:
        print("\n⚠️  需要继续训练或调整参数")
    
    print(f"\n  运行时间: {elapsed:.1f}秒")
    
    return {
        'history': history,
        'passed': passed >= 2,
        'max_stored': max_stored,
        'avg_winter_death': avg_winter_death,
        'total_stored': total_stored_final,
        'final_nodes': final_avg_nodes
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='阶段二：哺乳动物时期测试')
    parser.add_argument('--generations', '-g', type=int, default=200, help='演化代数')
    parser.add_argument('--population', '-p', type=int, default=50, help='种群大小')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    result = run_stage2_test(
        n_generations=args.generations,
        population_size=args.population,
        verbose=not args.quiet
    )
    
    sys.exit(0 if result.get('passed', False) else 1)