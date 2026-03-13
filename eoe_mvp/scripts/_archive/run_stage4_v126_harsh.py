#!/usr/bin/env python3
"""
Stage 4 v12.6 - 严苛环境测试
============================
核心思路: 让环境自己选择,不靠参数惩罚

严苛设定:
- 食物减少 (10个,60能量)
- 冬天无食物+3倍代谢
- 掠食者活跃
- 只有真正高效的个体才能存活
"""
import sys
sys.path.insert(0, '.')

import json
from core.eoe import Population


def main():
    print("="*70)
    print("Stage 4 v12.6 - 严苛环境测试")
    print("="*70)
    
    # 加载Stage 3大脑
    with open('champions/stage3_champion.json') as f:
        champion_brain = json.load(f)
    
    print(f"\n使用Stage 3脑结构初始化:")
    print(f"  节点: {len(champion_brain['nodes'])}, 边: {len(champion_brain['edges'])}")
    
    # 严苛环境配置
    print(f"\n【严苛环境设定】")
    print(f"  食物数量: 8 (原15)")
    print(f"  食物能量: 50 (原80)")
    print(f"  冬天代谢倍率: 3.0 (原2.0)")
    print(f"  冬天食物乘数: 0.0 (无食物)")
    print(f"  红皇后竞争: 启用 (3个敌对)")
    print(f"  寿命: 350 (原500)")
    
    pop = Population(
        population_size=30,
        elite_ratio=0.15,  # 更严格的精英比例
        env_width=70,      # 更小的环境
        env_height=70,
        target_pos=(60.0, 60.0),
        
        # 食物大幅减少
        n_food=8,
        food_energy=50.0,
        
        # 季节更严酷
        seasonal_cycle=True,
        season_length=25,  # 更短的季节
        winter_food_multiplier=0.0,  # 冬天无食物
        winter_metabolic_multiplier=3.0,  # 3倍代谢
        
        # 红皇后竞争 (敌对Agent)
        red_queen=True,
        n_rivals=3,
        rival_refresh_interval=30,
        
        # 脑代谢压力
        use_champion=True,
        champion_brain=champion_brain,
    )
    
    print(f"\n种群配置:")
    print(f"  初始数量: {pop.population_size}")
    print(f"  精英比例: {pop.elite_ratio}")
    print(f"  脑代谢: α={pop.brain_metabolic_alpha}, β={pop.brain_metabolic_beta}")
    
    # 运行演化
    n_generations = 30
    print(f"\n开始演化: {n_generations}代 (严苛环境)")
    print("-"*70)
    
    history = pop.run(n_generations=n_generations, verbose=True)
    
    # 分析结果
    print("\n" + "="*70)
    print("演化结果分析")
    print("="*70)
    
    # 找最佳
    best_gen = max(range(len(history)), key=lambda i: history[i]['best_fitness'])
    best = history[best_gen]
    
    print(f"\n最佳个体 (Gen {best_gen}):")
    print(f"  适应度: {best['best_fitness']:.2f}")
    print(f"  平均节点: {best['avg_nodes']:.1f}")
    print(f"  平均边: {best['avg_edges']:.1f}")
    
    # 存活率分析
    print(f"\n存活率变化:")
    for i in [0, 4, 9, 14, 19, 24, 29]:
        if i < len(history):
            # 近似存活率 (从avg_fitness推断)
            print(f"  Gen {i}: best={history[i]['best_fitness']:.1f}, nodes={history[i]['avg_nodes']:.0f}")
    
    # 节点变化趋势
    print(f"\n节点变化趋势:")
    print(f"  Gen 0:  {history[0]['avg_nodes']:.1f}")
    print(f"  Gen 10: {history[10]['avg_nodes']:.1f}")
    print(f"  Gen 20: {history[20]['avg_nodes']:.1f}")
    print(f"  Gen 29: {history[29]['avg_nodes']:.1f}")
    
    delta = history[29]['avg_nodes'] - history[0]['avg_nodes']
    if delta < 0:
        print(f"  → 节点减少 {-delta:.1f}! 稀疏化成功!")
    else:
        print(f"  → 节点增加 {delta:.1f}")
    
    # 保存冠军
    best_agent = best['best_agent']
    champion_data = best_agent.genome.to_dict()
    
    save_path = 'champions/stage4_v126_harsh.json'
    with open(save_path, 'w') as f:
        json.dump(champion_data, f, indent=2)
    
    print(f"\n冠军已保存: {save_path}")
    
    # 分析脑结构
    nodes = len(champion_data['nodes'])
    edges = len(champion_data['edges'])
    meta = sum(1 for n in champion_data['nodes'] if n.get('node_type') == 'META_NODE')
    sensor = sum(1 for n in champion_data['nodes'] if n.get('node_type') == 'SENSOR')
    
    print(f"\n冠军脑结构:")
    print(f"  节点: {nodes}, 边: {edges}")
    print(f"  META: {meta} ({meta/nodes*100:.0f}%)")
    print(f"  传感器: {sensor}")
    print(f"  密度: {edges/nodes:.2f}")
    
    print("\n✅ 严苛环境测试完成!")
    return history


if __name__ == "__main__":
    main()