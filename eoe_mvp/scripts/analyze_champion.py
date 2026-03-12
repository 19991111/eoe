#!/usr/bin/env python3
"""
冠军行为策略分析
================
分析严苛环境下冠军大脑的行为模式
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
from core.eoe import Population


def analyze_champion_behavior(brain_path, n_steps=200):
    """分析冠军的行为策略"""
    
    with open(brain_path) as f:
        brain_data = json.load(f)
    
    print("="*70)
    print("冠军行为策略分析")
    print("="*70)
    
    # 创建环境
    pop = Population(
        population_size=1,
        env_width=70, env_height=70,
        n_food=8,
        food_energy=50.0,
        seasonal_cycle=True,
        season_length=25,
        winter_food_multiplier=0.0,
        winter_metabolic_multiplier=3.0,
        use_champion=True,
        champion_brain=brain_data,
    )
    
    agent = pop.agents[0]
    env = pop.environment
    
    print(f"\n大脑结构:")
    print(f"  节点: {len(agent.genome.nodes)}")
    print(f"  边: {len(agent.genome.edges)}")
    
    # 运行agent并记录行为
    history = {
        'x': [], 'y': [],           # 位置
        'energy': [],               # 能量
        'food_eaten': 0,            # 吃的食物数
        'distance_to_food': [],     # 到最近食物的距离
        'season': [],               # 季节
        'action': [],               # 行为
        'velocity': [],             # 速度
    }
    
    for step in range(n_steps):
        # 记录状态
        history['x'].append(agent.x)
        history['y'].append(agent.y)
        history['energy'].append(agent.internal_energy)
        
        # 计算到最近食物的距离
        if env.food_positions:
            min_dist = min([
                ((agent.x - fx)**2 + (agent.y - fy)**2)**0.5 
                for fx, fy in env.food_positions
            ])
            history['distance_to_food'].append(min_dist)
        else:
            history['distance_to_food'].append(999)
        
        history['season'].append(env.current_season)
        
        # 记录速度
        vx = getattr(agent, 'vx', 0) or 0
        vy = getattr(agent, 'vy', 0) or 0
        velocity = (vx**2 + vy**2)**0.5
        history['velocity'].append(velocity)
        
        # 行为分类
        if velocity > 0.5:
            history['action'].append('moving')
        elif env.food_positions:
            # 检查是否在吃食物
            eating = False
            for fx, fy in env.food_positions:
                if ((agent.x - fx)**2 + (agent.y - fy)**2)**0.5 < 3:
                    eating = True
                    break
            history['action'].append('eating' if eating else 'idle')
        else:
            history['action'].append('idle')
        
        # 执行一步
        agent.step(env)
        
        # 检查食物被吃
        initial_food = len(env.food_positions)
        # ... 处理会在step中自动发生
    
    # 统计结果
    print(f"\n=== {n_steps}步行为统计 ===")
    
    # 位置分析
    x_range = max(history['x']) - min(history['x'])
    y_range = max(history['y']) - min(history['y'])
    print(f"\n位置活动范围:")
    print(f"  X: {min(history['x']):.1f} ~ {max(history['x']):.1f} (范围: {x_range:.1f})")
    print(f"  Y: {min(history['y']):.1f} ~ {max(history['y']):.1f} (范围: {y_range:.1f})")
    
    # 能量分析
    print(f"\n能量变化:")
    print(f"  初始: {history['energy'][0]:.1f}")
    print(f"  最终: {history['energy'][-1]:.1f}")
    print(f"  最大: {max(history['energy']):.1f}")
    print(f"  最小: {min(history['energy']):.1f}")
    
    # 食物统计
    food_count = len([a for a in history['action'] if a == 'eating'])
    move_count = len([a for a in history['action'] if a == 'moving'])
    idle_count = len([a for a in history['action'] if a == 'idle'])
    
    print(f"\n行为统计:")
    print(f"  移动: {move_count}步 ({move_count/n_steps*100:.0f}%)")
    print(f"  进食: {food_count}步 ({food_count/n_steps*100:.0f}%)")
    print(f"  空闲: {idle_count}步 ({idle_count/n_steps*100:.0f}%)")
    
    # 速度分析
    avg_velocity = np.mean(history['velocity'])
    print(f"\n移动模式:")
    print(f"  平均速度: {avg_velocity:.2f}")
    print(f"  最大速度: {max(history['velocity']):.2f}")
    
    # 季节行为
    summer_actions = [history['action'][i] for i in range(len(history['season'])) if history['season'][i] == 'summer']
    winter_actions = [history['action'][i] for i in range(len(history['season'])) if history['season'][i] == 'winter']
    
    print(f"\n季节行为差异:")
    if summer_actions:
        summer_move = sum(1 for a in summer_actions if a == 'moving') / len(summer_actions) * 100
        print(f"  夏天移动: {summer_move:.0f}%")
    if winter_actions:
        winter_move = sum(1 for a in winter_actions if a == 'moving') / len(winter_actions) * 100
        print(f"  冬天移动: {winter_move:.0f}%")
    
    # 食物寻找策略
    print(f"\n食物寻找策略:")
    avg_dist = np.mean(history['distance_to_food'][:50])
    print(f"  初始平均距离: {avg_dist:.1f}")
    
    # 检查是否有探索模式
    if x_range > 30 or y_range > 30:
        print(f"  策略: 广泛探索 (活动范围大)")
    elif avg_dist < 10:
        print(f"  策略: 就近觅食 (紧跟食物)")
    else:
        print(f"  策略: 中等探索")
    
    # 生存分析
    survived = sum(1 for e in history['energy'] if e > 0)
    print(f"\n生存分析:")
    print(f"  存活步数: {survived}/{n_steps}")
    print(f"  生存率: {survived/n_steps*100:.0f}%")
    
    return history


def analyze_brain_structure(brain_path):
    """分析大脑结构"""
    
    with open(brain_path) as f:
        brain = json.load(f)
    
    print("\n" + "="*70)
    print("大脑结构分析")
    print("="*70)
    
    nodes = len(brain['nodes'])
    edges = len(brain['edges'])
    
    # 节点类型
    types = {}
    for n in brain['nodes']:
        t = n.get('node_type', 'unknown')
        types[t] = types.get(t, 0) + 1
    
    print(f"\n节点类型分布:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        pct = c/nodes*100
        bar = "█" * int(pct/5)
        print(f"  {t:15s}: {c:2d} ({pct:5.1f}%) {bar}")
    
    # 边分析
    print(f"\n连接分析:")
    print(f"  总边数: {edges}")
    print(f"  密度: {edges/nodes:.2f}")
    
    # 计算节点的连接度
    in_degree = {}
    out_degree = {}
    for e in brain['edges']:
        src = e.get('source')
        dst = e.get('target')
        if src is not None:
            out_degree[src] = out_degree.get(src, 0) + 1
        if dst is not None:
            in_degree[dst] = in_degree.get(dst, 0) + 1
    
    # 找出高度节点
    all_degrees = {}
    for n in brain['nodes']:
        nid = n.get('node_id')
        deg = in_degree.get(nid, 0) + out_degree.get(nid, 0)
        all_degrees[nid] = deg
    
    top_nodes = sorted(all_degrees.items(), key=lambda x: -x[1])[:5]
    print(f"\n高度连接节点 (Hub):")
    for nid, deg in top_nodes:
        n_type = next((n['node_type'] for n in brain['nodes'] if n['node_id'] == nid), '?')
        print(f"  Node {nid}: degree={deg}, type={n_type}")
    
    # 识别脑区
    print(f"\n脑区识别:")
    
    # 传感器连接
    sensor_nodes = [n['node_id'] for n in brain['nodes'] if n.get('node_type') == 'SENSOR']
    if sensor_nodes:
        sensor_outs = set()
        for e in brain['edges']:
            if e.get('source') in sensor_nodes:
                sensor_outs.add(e.get('target'))
        print(f"  感知层 → 第一层: {len(sensor_outs)} 节点")
    
    # 运动输出
    actuator_nodes = [n['node_id'] for n in brain['nodes'] if n.get('node_type') == 'ACTUATOR']
    print(f"  运动输出: {len(actuator_nodes)} 节点")
    
    # 计算层
    compute_nodes = [n['node_id'] for n in brain['nodes'] if n.get('node_type') in ['ADD', 'MULTIPLY', 'DELAY']]
    print(f"  计算层: {len(compute_nodes)} 节点")
    
    # 预测层
    predictor_nodes = [n['node_id'] for n in brain['nodes'] if n.get('node_type') == 'PREDICTOR']
    print(f"  预测层: {len(predictor_nodes)} 节点")
    
    # 元认知
    meta_nodes = [n['node_id'] for n in brain['nodes'] if n.get('node_type') == 'META_NODE']
    print(f"  元认知层: {len(meta_nodes)} 节点 (极低!)")
    
    return types


if __name__ == "__main__":
    brain_path = "champions/stage4_v126_harsh.json"
    
    # 结构分析
    types = analyze_brain_structure(brain_path)
    
    # 行为分析
    history = analyze_champion_behavior(brain_path, n_steps=200)
    
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    # 策略推断
    print("\n推断的行为策略:")
    
    # 基于结构
    if types.get('META_NODE', 0) <= 2:
        print("  ✓ 极简元认知: 不依赖自我监控,直接基于环境行动")
    
    if types.get('DELAY', 0) >= 5:
        print("  ✓ 时序记忆: 具备短期记忆能力,可能记住食物位置")
    
    if types.get('PREDICTOR', 0) >= 3:
        print("  ✓ 预测能力: 能预测环境变化(季节/食物)")
    
    if types.get('ADD', 0) + types.get('MULTIPLY', 0) >= 15:
        print("  ✓ 复杂计算: 大脑主要是计算单元,处理传感器信息")
    
    # 基于行为
    move_ratio = sum(1 for a in history['action'] if a == 'moving') / len(history['action'])
    if move_ratio > 0.7:
        print("  ✓ 活跃探索: 持续移动寻找资源")
    elif move_ratio < 0.3:
        print("  ✓ 机会主义: 等待时机,减少能量消耗")
    
    print("\n✅ 分析完成!")