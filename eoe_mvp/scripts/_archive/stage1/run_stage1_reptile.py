#!/usr/bin/env python3
"""
阶段一：爬行脑时期测试

核心目标：让 0 初始边的无脑 Agent，演化出"感知食物 -> 移动 -> 进食"的本能

开启机制:
- 基础代谢 (Alpha/Beta)
- 食物生成
- 基础 Sensor 和 Actuator

关闭机制:
- 季节循环 (seasonal_cycle=False)
- 疲劳系统 (无 enable_fatigue_system)
- 热力学庇护所 (无 enable_thermal_sanctuary)
- 敌对 Agent (red_queen=False)
- LLM Demiurge Loop (不使用)

验收标准:
- 种群能在常温环境下稳定存活
- 主动寻找并吃掉食物
- 寿命受限于老死而不是饿死
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population
from core.eoe.environment import Environment


def run_stage1_test(
    n_generations: int = 100,
    population_size: int = 50,
    verbose: bool = True
) -> dict:
    """运行阶段一测试"""
    
    print("=" * 60)
    print("  阶段一：爬行脑时期 - 基础生存测试")
    print("=" * 60)
    print("\n[配置]")
    print(f"  种群大小: {population_size}")
    print(f"  演化代数: {n_generations}")
    print(f"  代谢: Alpha=0.02, Beta=0.02 (降低以支持长寿命)")
    print(f"  食物能量: 30.0")
    print(f"  寿命: 500帧")
    print(f"  季节循环: 关闭")
    print(f"  疲劳系统: 关闭")
    print(f"  热力学庇护所: 关闭")
    print(f"  敌对Agent: 关闭")
    print(f"  LLM Demiurge: 关闭")
    print("=" * 60)
    
    # 创建环境 - 纯基础设置
    env = Environment(
        width=100.0,
        height=100.0,
        n_food=15,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,  # v0.98: 降低代谢，让Agent能活更久
        metabolic_beta=0.02,
        n_walls=0,
        day_night_cycle=False
    )
    
    # 创建种群 - 关闭所有高级机制
    pop = Population(
        population_size=population_size,
        elite_ratio=0.2,
        env_width=100.0,
        env_height=100.0,
        lifespan=500,
        n_food=15,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,  # v0.98: 降低代谢
        metabolic_beta=0.02,
        # 关键：关闭所有高级机制
        seasonal_cycle=False,      # 无季节
        winter_food_multiplier=1.0,  # 冬天也有食物
        winter_metabolic_multiplier=1.0,  # 无额外代谢
        red_queen=False,           # 无敌对Agent
        use_champion=False,        # 纯随机初始化
        pure_survival_mode=True    # 纯生存适应度
    )
    
    # 初始化种群
    pop._init_population()
    
    print(f"\n[初始状态]")
    print(f"  Agent数量: {len(pop.agents)}")
    print(f"  初始大脑节点数: {len(pop.agents[0].genome.nodes)}")
    print(f"  初始大脑边数: {len(pop.agents[0].genome.edges)}")
    
    # 运行演化
    start_time = time.time()
    
    try:
        history = pop.run(n_generations=n_generations, verbose=verbose)
    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动停止")
        history = []
    
    elapsed = time.time() - start_time
    
    # 分析结果
    print("\n" + "=" * 60)
    print("  阶段一测试结果")
    print("=" * 60)
    
    if not history:
        print("  无历史数据（测试被中断）")
        return {}
    
    # 统计
    best_fitnesses = [h['best_fitness'] for h in history]
    avg_fitnesses = [h['avg_fitness'] for h in history]
    avg_nodes = [h['avg_nodes'] for h in history]
    avg_edges = [h['avg_edges'] for h in history]
    
    # 找到最佳的代数
    best_gen = np.argmax(best_fitnesses)
    best_fit = best_fitnesses[best_gen]
    
    print(f"\n  运行时间: {elapsed:.1f}秒")
    print(f"\n  [适应度]")
    print(f"    最高适应度: {max(best_fitnesses):.2f} (Gen {best_gen})")
    print(f"    最终适应度: {best_fitnesses[-1]:.2f}")
    print(f"    平均适应度: {np.mean(avg_fitnesses[-10:]):.2f}")
    
    print(f"\n  [大脑复杂度]")
    print(f"    最终平均节点数: {avg_nodes[-1]:.1f}")
    print(f"    最终平均边数: {avg_edges[-1]:.1f}")
    print(f"    最大节点数: {max([h['avg_nodes'] for h in history]):.1f}")
    
    # 检查是否有进食行为
    final_best = history[-1]['best_agent']
    if hasattr(final_best, 'food_eaten'):
        print(f"\n  [行为]")
        print(f"    最佳个体进食数: {final_best.food_eaten}")
    
    # 验收标准检查
    print("\n" + "=" * 60)
    print("  验收标准检查")
    print("=" * 60)
    
    checks = []
    
    # 1. 稳定存活
    survival_rate = len([a for a in pop.agents if a.is_alive]) / len(pop.agents)
    checks.append(("种群存活率 > 50%", survival_rate > 0.5))
    print(f"    种群存活率: {survival_rate*100:.1f}% {'✅' if survival_rate > 0.5 else '❌'}")
    
    # 2. 主动进食
    avg_food = np.mean([getattr(a, 'food_eaten', 0) for a in pop.agents])
    checks.append(("平均进食数 > 0", avg_food > 0))
    print(f"    平均进食数: {avg_food:.2f} {'✅' if avg_food > 0 else '❌'}")
    
    # 3. 脑复杂度增加
    complexity_grew = avg_nodes[-1] > avg_nodes[0]
    checks.append(("大脑复杂度增加", complexity_grew))
    print(f"    大脑复杂度增加: {'是' if complexity_grew else '否'} {'✅' if complexity_grew else '❌'}")
    
    # 4. 适应度提升
    fitness_improved = best_fitnesses[-1] > best_fitnesses[0]
    checks.append(("适应度提升", fitness_improved))
    print(f"    适应度提升: {'是' if fitness_improved else '否'} {'✅' if fitness_improved else '❌'}")
    
    passed = sum(1 for _, p in checks if p)
    print(f"\n  通过: {passed}/{len(checks)}")
    
    if passed == len(checks):
        print("\n  🎉 阶段一验收通过！")
    else:
        print("\n  ⚠️  需要进一步优化")
    
    return {
        'history': history,
        'best_fitness': max(best_fitnesses),
        'final_avg_nodes': avg_nodes[-1],
        'passed': passed == len(checks)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='阶段一：爬行脑时期测试')
    parser.add_argument('--generations', '-g', type=int, default=100, help='演化代数')
    parser.add_argument('--population', '-p', type=int, default=50, help='种群大小')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    result = run_stage1_test(
        n_generations=args.generations,
        population_size=args.population,
        verbose=not args.quiet
    )
    
    sys.exit(0 if result.get('passed', False) else 1)