#!/usr/bin/env python3
"""
Stage 4 v12.6 - 脑代谢压力测试
================================
使用Stage 3脑结构初始化,测试新的脑代谢压力机制

核心变化:
- 删除 complexity_premium (不再奖励"大")
- 启用 brain_metabolic_alpha/beta (大脑要付费)
- 目标是演化出更稀疏、更高效的脑结构
"""
import sys
sys.path.insert(0, '.')

import json
from core.eoe import Population


def main():
    print("="*70)
    print("Stage 4 v12.6 - 脑代谢压力测试")
    print("="*70)
    
    # 加载Stage 3大脑作为初始化
    with open('champions/stage3_champion.json') as f:
        champion_brain = json.load(f)
    
    print(f"\n使用Stage 3脑结构初始化:")
    print(f"  节点: {len(champion_brain['nodes'])}")
    print(f"  边: {len(champion_brain['edges'])}")
    
    print(f"\n脑代谢压力参数:")
    print(f"  brain_metabolic_alpha = 0.01 (节点能耗)")
    print(f"  brain_metabolic_beta = 0.005 (边能耗)")
    print(f"  complexity_premium = 0.0 (已删除)")
    
    # 创建种群
    pop = Population(
        population_size=30,         # 30个智能体
        elite_ratio=0.2,            # 20%精英
        env_width=100,
        env_height=100,
        target_pos=(80.0, 80.0),
        n_food=15,
        food_energy=80.0,
        seasonal_cycle=True,
        season_length=35,
        winter_food_multiplier=0.0,  # 冬天无食物
        winter_metabolic_multiplier=2.0,  # 冬天2倍代谢
        use_champion=True,
        champion_brain=champion_brain,
        
        # v12.6 脑代谢压力
        metabolic_alpha=0.003,
        metabolic_beta=0.05,
    )
    
    print(f"\n种群配置:")
    print(f"  种群大小: {pop.population_size}")
    print(f"  精英比例: {pop.elite_ratio}")
    print(f"  能量初始: 150.0")
    print(f"  寿命: {pop.lifespan}")
    
    # 运行演化
    n_generations = 50
    
    print(f"\n开始演化: {n_generations}代")
    print("-"*70)
    
    history = pop.run(n_generations=n_generations, verbose=True)
    
    # 结果分析
    print("\n" + "="*70)
    print("演化结果")
    print("="*70)
    
    # 找最佳
    best_gen = max(range(len(history)), key=lambda i: history[i]['best_fitness'])
    best = history[best_gen]
    
    print(f"\n最佳个体 (Gen {best_gen}):")
    print(f"  适应度: {best['best_fitness']:.2f}")
    print(f"  节点: {best['avg_nodes']:.1f}")
    print(f"  边: {best['avg_edges']:.1f}")
    if 'avg_brain_efficiency' in best:
        print(f"  大脑效率: {best['avg_brain_efficiency']:.3f}")
        print(f"  脑代谢成本: {best['brain_cost_per_agent']:.3f}")
    
    # 适应度变化
    print(f"\n适应度变化:")
    print(f"  第1代: {history[0]['best_fitness']:.2f}")
    print(f"  第10代: {history[9]['best_fitness']:.2f}")
    print(f"  第25代: {history[24]['best_fitness']:.2f}")
    print(f"  第50代: {history[49]['best_fitness']:.2f}")
    
    # 节点变化
    print(f"\n节点变化:")
    print(f"  第1代: {history[0]['avg_nodes']:.1f}")
    print(f"  第10代: {history[9]['avg_nodes']:.1f}")
    print(f"  第25代: {history[24]['avg_nodes']:.1f}")
    print(f"  第50代: {history[49]['avg_nodes']:.1f}")
    
    # 保存冠军
    best_agent = best['best_agent']
    champion_data = best_agent.genome.to_dict()
    
    save_path = 'champions/stage4_v126_brain.json'
    with open(save_path, 'w') as f:
        json.dump(champion_data, f, indent=2)
    
    print(f"\n冠军已保存: {save_path}")
    
    # 分析冠军脑结构
    nodes = len(champion_data['nodes'])
    edges = len(champion_data['edges'])
    meta = sum(1 for n in champion_data['nodes'] if n.get('node_type') == 'META_NODE')
    
    print(f"\n冠军脑结构分析:")
    print(f"  总节点: {nodes}")
    print(f"  总边: {edges}")
    print(f"  META节点: {meta} ({meta/nodes*100:.0f}%)")
    print(f"  密度: {edges/nodes:.2f}")
    
    print("\n✅ Stage 4 v12.6 完成!")
    return history


if __name__ == "__main__":
    main()