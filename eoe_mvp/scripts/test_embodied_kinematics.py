#!/usr/bin/env python3
"""
v13.1 具身运动学集成测试
验证阻尼和能量代谢是否正确工作
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from core.eoe.environment import Environment


def test_embodied_physics():
    """测试具身运动学物理模型"""
    
    print("="*60)
    print("🧪 v13.1 具身运动学集成测试")
    print("="*60)
    
    # 创建环境 (启用具身运动学)
    env = Environment(
        width=100,
        height=100,
        n_food=0,  # 无食物
        metabolic_alpha=0.001,  # 低基础代谢
        metabolic_beta=0.001
    )
    
    # 手动启用具身运动学
    env.use_embodied_kinematics = True
    env.linear_damping = 0.1
    env.angular_damping = 0.2
    env.actuator_cost_rate = 0.01
    env.max_linear_speed = 10.0
    
    print(f"\n📋 环境参数:")
    print(f"  use_embodied_kinematics: {env.use_embodied_kinematics}")
    print(f"  linear_damping: {env.linear_damping}")
    print(f"  angular_damping: {env.angular_damping}")
    print(f"  actuator_cost_rate: {env.actuator_cost_rate}")
    
    # 创建一个简单的 Agent 对象 (用于测试物理)
    class MockAgent:
        def __init__(self):
            self.x = 50.0
            self.y = 50.0
            self.theta = 0.0
            self.internal_energy = 100.0
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.speed = 0.0
    
    agent = MockAgent()
    
    print(f"\n📋 Agent 初始状态:")
    print(f"  位置: ({agent.x}, {agent.y})")
    print(f"  朝向: {agent.theta:.2f}")
    print(f"  能量: {agent.internal_energy:.2f}")
    print(f"  线速度: {agent.linear_velocity}")
    print(f"  角速度: {agent.angular_velocity}")
    
    # 测试用例 1: 两侧同向发力 -> 直线前进
    print(f"\n🚀 测试1: 两侧同向发力 (1.0, 1.0) -> 直线前进")
    actuator_outputs = np.array([1.0, 1.0])
    env._update_agent_physics(agent, actuator_outputs)
    
    print(f"  位置: ({agent.x:.2f}, {agent.y:.2f})")
    print(f"  线速度: {agent.linear_velocity:.4f}")
    print(f"  角速度: {agent.angular_velocity:.4f}")
    print(f"  能量: {agent.internal_energy:.4f}")
    
    # 验证
    test1_pass = (
        agent.linear_velocity > 0.5 and
        agent.internal_energy < 100.0  # 能量已扣除
    )
    print(f"  {'✅ 通过' if test1_pass else '❌ 失败'}")
    
    # 测试用例 2: 差速转向 -> 旋转
    print(f"\n🔄 测试2: 差速 (1.0, -1.0) -> 右转")
    # 重置 Agent
    agent.x = 50.0
    agent.y = 50.0
    agent.theta = 0.0
    agent.linear_velocity = 0.0
    agent.angular_velocity = 0.0
    agent.internal_energy = 100.0
    
    actuator_outputs = np.array([1.0, -1.0])
    env._update_agent_physics(agent, actuator_outputs)
    
    print(f"  位置: ({agent.x:.2f}, {agent.y:.2f})")
    print(f"  线速度: {agent.linear_velocity:.4f}")
    print(f"  角速度: {agent.angular_velocity:.4f}")
    
    # 验证: 净推力为 0，应该原地转向
    test2_pass = (
        abs(agent.linear_velocity) < 0.1 and
        agent.angular_velocity < -0.5  # 右转 (负角速度)
    )
    print(f"  {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    # 测试用例 3: 倒车 (负推力)
    print(f"\n🔙 测试3: 倒车 (-1.0, -1.0) -> 后退")
    # 重置 Agent
    agent.x = 50.0
    agent.y = 50.0
    agent.theta = 0.0
    agent.linear_velocity = 0.0
    agent.angular_velocity = 0.0
    agent.internal_energy = 100.0
    
    actuator_outputs = np.array([-1.0, -1.0])
    env._update_agent_physics(agent, actuator_outputs)
    
    print(f"  位置: ({agent.x:.2f}, {agent.y:.2f})")
    print(f"  线速度: {agent.linear_velocity:.4f}")
    print(f"  能量: {agent.internal_energy:.4f}")
    
    # 验证: 线速度应该为负 (倒车)
    test3_pass = agent.linear_velocity < -0.5
    print(f"  {'✅ 通过' if test3_pass else '❌ 失败'}")
    
    # 测试用例 4: 阻尼效果 (无推力时速度衰减)
    print(f"\n📉 测试4: 阻尼效果 (无推力)")
    # 重置 Agent
    agent.x = 50.0
    agent.y = 50.0
    agent.theta = 0.0
    agent.linear_velocity = 2.0  # 初始速度
    agent.angular_velocity = 0.5
    agent.internal_energy = 100.0
    
    # 无推力更新
    actuator_outputs = np.array([0.0, 0.0])
    
    # 多步更新
    for _ in range(10):
        env._update_agent_physics(agent, actuator_outputs)
    
    print(f"  初始线速度: 2.0")
    print(f"  最终线速度: {agent.linear_velocity:.4f}")
    print(f"  初始角速度: 0.5")
    print(f"  最终角速度: {agent.angular_velocity:.4f}")
    
    # 验证: 速度应该衰减
    test4_pass = (
        agent.linear_velocity < 2.0 and
        agent.angular_velocity < 0.5
    )
    print(f"  {'✅ 通过' if test4_pass else '❌ 失败'}")
    
    # 总结
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    
    print(f"\n{'='*60}")
    print(f"{'✅ 所有测试通过' if all_pass else '❌ 部分测试失败'}")
    print(f"{'='*60}")
    
    return all_pass


if __name__ == "__main__":
    success = test_embodied_physics()
    sys.exit(0 if success else 1)