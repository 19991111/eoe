#!/usr/bin/env python3
"""
冬天死亡诊断脚本

分析为什么 Agent 在冬天死亡:
1. 能量消耗速度 vs 补充速度
2. 贮粮行为是否正确触发
3. 代谢惩罚是否过重
"""

import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population
from core.eoe.environment import Environment


def diagnose_winter_survival():
    """诊断冬天生存问题"""
    
    print("=" * 70)
    print("  冬天生存诊断")
    print("=" * 70)
    
    # 加载阶段一的大脑
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    if not os.path.exists(brain_path):
        print(f"❌ 阶段一大脑不存在: {brain_path}")
        return
    
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    print(f"\n[已加载大脑]")
    print(f"  节点: {len(brain_data['nodes'])}")
    print(f"  边: {len(brain_data['edges'])}")
    
    # 创建测试种群 - 小规模快速测试
    pop = Population(
        population_size=10,
        elite_ratio=0.2,
        env_width=50.0,
        env_height=50.0,
        lifespan=300,         # 缩短测试
        n_food=20,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        seasonal_cycle=True,
        season_length=40,     # 短季节以便快速观察
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=1.5,  # 较重的惩罚
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop._init_population()
    
    # 找一个有代表性的 Agent
    test_agent = pop.agents[0]
    
    print(f"\n[环境参数]")
    print(f"  季节长度: {pop.environment.season_length} 帧")
    print(f"  冬天食物倍率: {pop.environment.winter_food_multiplier}")
    print(f"  冬天代谢倍率: {pop.environment.winter_metabolic_multiplier}")
    print(f"  巢穴系统: {pop.environment.nest_enabled}")
    print(f"  初始能量: {test_agent.initial_energy}")
    
    # 逐帧模拟，观察能量变化
    print(f"\n[逐帧诊断 - 观察单个 Agent]")
    
    print(f"\n初始状态:")
    print(f"  能量: {test_agent.internal_energy:.1f}")
    print(f"  食物携带: {test_agent.food_carried}")
    print(f"  食物存储: {test_agent.food_stored}")
    print(f"  当前位置: ({test_agent.x:.1f}, {test_agent.y:.1f})")
    
    season_length = pop.environment.season_length
    
    # 运行 100 帧，观察季节变化和能量变化
    energy_history = []
    season_history = []
    food_eaten_history = []
    
    for frame in range(100):
        # 单步更新
        pop.environment.step()
        
        # 记录状态
        energy_history.append(test_agent.internal_energy)
        season_history.append(pop.environment.current_season)
        food_eaten_history.append(getattr(test_agent, 'food_eaten', 0))
        
        if frame % 10 == 0:
            print(f"  Frame {frame:3d}: energy={test_agent.internal_energy:7.2f}, "
                  f"season={pop.environment.current_season:6s}, "
                  f"carried={test_agent.food_carried}, stored={test_agent.food_stored}")
    
    # 分析结果
    print(f"\n[诊断结果]")
    
    # 1. 能量消耗分析
    initial_energy = energy_history[0]
    final_energy = energy_history[-1]
    energy_spent = initial_energy - final_energy
    
    summer_frames = sum(1 for s in season_history if s == 'summer')
    winter_frames = sum(1 for s in season_history if s == 'winter')
    
    print(f"\n1. 能量消耗:")
    print(f"   初始能量: {initial_energy:.1f}")
    print(f"   最终能量: {final_energy:.1f}")
    print(f"   消耗总量: {energy_spent:.1f}")
    print(f"   平均消耗/帧: {energy_spent/100:.3f}")
    
    print(f"\n2. 季节分布:")
    print(f"   夏天帧数: {summer_frames}")
    print(f"   冬天帧数: {winter_frames}")
    
    # 2. 代谢分析
    print(f"\n3. 代谢计算:")
    n_nodes = len(test_agent.genome.nodes)
    n_edges = len(test_agent.genome.edges)
    base_metabolic = n_nodes * 0.02 + n_edges * 0.02
    winter_metabolic = base_metabolic * 1.5 * 1.5  # 黑夜1.5x * 冬天1.5x
    
    print(f"   节点数: {n_nodes}, 边数: {n_edges}")
    print(f"   基础代谢/帧: {base_metabolic:.3f}")
    print(f"   冬天夜晚代谢/帧: {winter_metabolic:.3f}")
    print(f"   100帧冬天消耗: {winter_metabolic * 40:.1f}")
    
    # 3. 食物来源分析
    print(f"\n4. 食物来源:")
    print(f"   进食次数: {test_agent.food_eaten}")
    print(f"   携带食物: {test_agent.food_carried}")
    print(f"   存储食物: {test_agent.food_stored}")
    
    if test_agent.food_stored > 0:
        winter_bonus = test_agent.food_stored * 30.0 * 0.5
        print(f"   存储食物可补充能量: {winter_bonus:.1f}")
    
    # 4. 问题诊断
    print(f"\n[问题诊断]")
    
    if winter_frames > 0:
        # 计算冬天期间的能量变化
        winter_energy_start = None
        winter_energy_end = None
        
        for i, s in enumerate(season_history):
            if s == 'winter' and winter_energy_start is None:
                winter_energy_start = energy_history[i]
            if s == 'winter':
                winter_energy_end = energy_history[i]
        
        if winter_energy_start is not None and winter_energy_end is not None:
            winter_spent = winter_energy_start - winter_energy_end
            print(f"   冬天期间能量消耗: {winter_spent:.1f}")
            
            # 检查是否有能量补充
            energy_gaps = []
            for i in range(1, len(energy_history)):
                if energy_history[i] > energy_history[i-1]:
                    energy_gaps.append(energy_history[i] - energy_history[i-1])
            
            if energy_gaps:
                print(f"   能量补充次数: {len(energy_gaps)}")
                print(f"   最大单次补充: {max(energy_gaps):.1f}")
            else:
                print(f"   ❌ 零能量补充！")
    
    # 5. 关键发现
    print(f"\n[关键发现]")
    
    if energy_spent > initial_energy * 0.8:
        print(f"   ⚠️ 能量消耗过快 (消耗了 {energy_spent/initial_energy*100:.0f}% 的初始能量)")
    
    if test_agent.food_stored == 0:
        print(f"   ⚠️ 无贮粮行为 - Agent 不知道要在夏天贮粮")
    else:
        print(f"   ✓ 有贮粮: {test_agent.food_stored} 个")
    
    if final_energy <= 0:
        print(f"   ❌ 能量耗尽死亡")
    else:
        print(f"   状态: 存活 (能量 {final_energy:.1f})")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    diagnose_winter_survival()