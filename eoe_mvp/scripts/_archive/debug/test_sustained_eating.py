#!/usr/bin/env python3
"""
测试：在理想条件下 Agent 能否通过进食维持能量？

让 Agent 持续站在食物旁边，观察能量变化
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.eoe.population import Population


def test_sustained_eating():
    """测试持续进食"""
    
    print("=" * 70)
    print("  持续进食测试")
    print("=" * 70)
    
    # 加载阶段一的大脑
    brain_path = os.path.join(PROJECT_ROOT, 'champions', 'stage1_best_brain.json')
    with open(brain_path, 'r') as f:
        brain_data = json.load(f)
    
    # 创建测试种群 - 关闭季节，关闭巢穴（直接吃）
    pop = Population(
        population_size=1,
        elite_ratio=0.2,
        env_width=20.0,
        env_height=20.0,
        lifespan=500,
        n_food=5,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        seasonal_cycle=False,  # 关闭季节 -> nest_enabled=False
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop._init_population()
    agent = pop.agents[0]
    env = pop.environment
    
    print(f"\n[初始状态]")
    print(f"  能量: {agent.internal_energy:.1f}")
    print(f"  节点: {len(agent.genome.nodes)}, 边: {len(agent.genome.edges)}")
    
    # 把 Agent 和食物放在理想位置 - 每次 step 后重新放置
    # 这样 Agent 永远都能吃到食物
    food_idx = 0
    fx, fy = env.food_positions[food_idx]
    agent.x = fx - 2.0  # 距离 2 < 阈值 3
    agent.y = fy
    
    print(f"\n[运行 50 帧，食物位置固定]")
    print(f"  位置: ({agent.x:.1f}, {agent.y:.1f})")
    print(f"  食物: ({fx:.1f}, {fy:.1f})")
    
    energy_history = [agent.internal_energy]
    food_eaten_history = [0]
    
    for frame in range(50):
        # 重新放置 Agent 到食物旁边
        agent.x = fx - 2.0
        agent.y = fy
        
        # 执行一步
        env.step()
        
        energy_history.append(agent.internal_energy)
        food_eaten_history.append(agent.food_eaten)
        
        if frame % 10 == 9:
            print(f"  Frame {frame+1}: energy={agent.internal_energy:7.2f}, "
                  f"eaten={agent.food_eaten}, carried={agent.food_carried}")
    
    # 分析结果
    print(f"\n[结果分析]")
    initial = energy_history[0]
    final = energy_history[-1]
    net_change = final - initial
    total_eaten = food_eaten_history[-1]
    
    print(f"  初始能量: {initial:.1f}")
    print(f"  最终能量: {final:.1f}")
    print(f"  净变化: {net_change:.1f}")
    print(f"  进食次数: {total_eaten}")
    print(f"  平均每帧代谢消耗: {(initial - final - total_eaten * 30 * 0.8) / 50:.3f}")
    
    if final > initial:
        print(f"\n  ✅ 进食可以维持能量！Agent 可以通过不断进食存活")
    else:
        print(f"\n  ❌ 即使不断进食，能量仍在减少！")
        print(f"  问题：代谢消耗 > 进食收益")
    
    # 测试巢穴模式
    print(f"\n" + "=" * 70)
    print("  巢穴模式测试")
    print("=" * 70)
    
    pop2 = Population(
        population_size=1,
        elite_ratio=0.2,
        env_width=20.0,
        env_height=20.0,
        lifespan=500,
        n_food=5,
        food_energy=30.0,
        respawn_food=True,
        metabolic_alpha=0.02,
        metabolic_beta=0.02,
        seasonal_cycle=True,  # 开启季节
        season_length=40,
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=1.5,
        nest_enabled=True,  # 巢穴模式
        use_champion=True,
        champion_brain=brain_data,
        pure_survival_mode=True
    )
    
    pop2._init_population()
    agent2 = pop2.agents[0]
    env2 = pop2.environment
    
    print(f"\n[初始状态]")
    print(f"  能量: {agent2.internal_energy:.1f}")
    print(f"  巢穴: {env2.nest_position}")
    
    # 放置食物和巢穴
    env2.nest_position = (10.0, 10.0)
    fx2, fy2 = env2.food_positions[0]
    
    print(f"\n[运行 80 帧 (跨越夏天和冬天)]")
    
    energy_history2 = [agent2.internal_energy]
    season_history = []
    
    for frame in range(80):
        # 把 Agent 放在食物旁边
        agent2.x = fx2 - 2.0
        agent2.y = fy2
        
        # 执行一步
        env2.step()
        
        energy_history2.append(agent2.internal_energy)
        season_history.append(env2.current_season)
        
        if frame % 20 == 19:
            winter_frames = sum(1 for s in season_history[-20:] if s == 'winter')
            print(f"  Frame {frame+1}: energy={agent2.internal_energy:7.2f}, "
                  f"season={env2.current_season}, stored={agent2.food_stored}")
    
    print(f"\n[结果分析]")
    initial2 = energy_history2[0]
    final2 = energy_history2[-1]
    
    print(f"  初始能量: {initial2:.1f}")
    print(f"  最终能量: {final2:.1f}")
    print(f"  净变化: {final2 - initial2:.1f}")
    print(f"  进食次数: {agent2.food_eaten}")
    print(f"  携带次数: {agent2.food_carried}")
    print(f"  贮粮次数: {agent2.food_stored}")


if __name__ == '__main__':
    test_sustained_eating()