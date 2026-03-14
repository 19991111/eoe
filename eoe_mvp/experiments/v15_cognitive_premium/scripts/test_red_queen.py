#!/usr/bin/env python3
"""
v15 Red Queen 测试
===================
验证智能猎物(Z字逃跑)机制
"""

import torch
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

from core.eoe.intelligent_prey import IntelligentPreySystem, IntelligentPreyConfig

# 配置
config = IntelligentPreyConfig()
config.detection_range = 25.0
config.escape_trigger_distance = 15.0
config.escape_speed = 2.0
config.zigzag_period = 8
config.zigzag_amplitude = 0.5
config.fatigue_duration = 30

print("=" * 60)
print("v15 Red Queen 智能猎物测试")
print("=" * 60)

# 创建3个能量源(猎物)
n_sources = 3
source_positions = torch.tensor([
    [30.0, 50.0],
    [50.0, 30.0],
    [70.0, 50.0]
])
prey_system = IntelligentPreySystem(config, source_positions, 'cpu')

bounds = (0.0, 100.0, 0.0, 100.0)

# 模拟2个Agent
agent_positions = torch.tensor([
    [35.0, 52.0],  # Agent 1: 靠近第一个猎物
    [55.0, 32.0],  # Agent 2: 靠近第二个猎物
])
agent_velocities = torch.tensor([
    [1.0, 0.0],   # 向右移动
    [0.0, 1.0],   # 向上移动
])

print(f"\n初始猎物位置:")
for i, pos in enumerate(source_positions):
    print(f"  猎物{i}: {pos.tolist()}")

print(f"\nAgent位置: {agent_positions.tolist()}")
print(f"Agent速度: {agent_velocities.tolist()}")

print("\n" + "-" * 60)
print(f"{'步':>3} | {'猎物0位置':>20} | {'逃跑':>6} | {'能耗':>8}")
print("-" * 60)

for step in range(20):
    result = prey_system.update(
        agent_positions,
        agent_velocities,
        bounds,
        dt=1.0
    )
    
    # 模拟Agent继续移动
    agent_positions += agent_velocities * 1.0
    
    # 边界处理
    agent_positions = torch.clamp(agent_positions, 0, 100)
    
    pos = prey_system.get_positions()
    escaped = result['results'][0]['escaped']
    cost = result['results'][0]['energy_cost']
    
    if step < 10 or step % 5 == 0:
        print(f"{step+1:>3} | ({pos[0,0]:>6.1f}, {pos[0,1]:>6.1f}) | {'是' if escaped else '否':>6} | {cost:>8.2f}")

print("-" * 60)
print(f"\n最终位置: {prey_system.get_positions().tolist()}")
print(f"逃跑次数: {sum(1 for r in result['results'] if r['escaped'])}")

# 测试直线靠近 vs Z字靠近
print("\n关键机制:")
print("  1. 检测范围: 25单位")
print("  2. 触发距离: 15单位") 
print("  3. Z字形周期: 8步")
print("  4. 逃跑后疲劳: 30步")
print("  5. 能量成本: 0.5/步")

print("\n✅ Red Queen 测试完成!")