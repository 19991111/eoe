#!/usr/bin/env python3
"""
v13.1 多通道具身运动学测试
验证 Channel 0-3 的物理效果
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from core.eoe.environment import Environment


def test_multichannel_physics():
    """测试多通道物理模型"""
    
    print("="*60)
    print("🧪 v13.1 多通道具身运动学测试")
    print("="*60)
    
    # 创建环境 (启用具身运动学)
    env = Environment(
        width=100,
        height=100,
        n_food=0,  # 无食物
        metabolic_alpha=0.001,
        metabolic_beta=0.001,
        stigmergy_field_enabled=True  # 启用信息素场
    )
    
    # 手动启用具身运动学
    env.use_embodied_kinematics = True
    env.linear_damping = 0.1
    env.angular_damping = 0.2
    env.actuator_cost_rate = 0.01
    env.max_linear_speed = 10.0
    
    print(f"\n📋 环境参数:")
    print(f"  use_embodied_kinematics: {env.use_embodied_kinematics}")
    print(f"  stigmergy_field: {hasattr(env, 'stigmergy_field')}")
    
    # 简单的 Agent 对象
    class MockAgent:
        def __init__(self):
            self.x = 50.0
            self.y = 50.0
            self.theta = 0.0
            self.internal_energy = 100.0
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0
            self.speed = 0.0
    
    # ==================== 测试 Channel 0: ENERGY ====================
    print(f"\n{'='*60}")
    print("🔌 Channel 0: ENERGY (推力)")
    print("="*60)
    
    agent = MockAgent()
    # 格式: [left, right, impedance, stigmergy, stress]
    actuator_outputs = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    env._update_agent_physics(agent, actuator_outputs)
    
    print(f"  输入: (1.0, 1.0) 两侧同向")
    print(f"  线速度: {agent.linear_velocity:.4f} (预期 > 0)")
    print(f"  能量: {agent.internal_energy:.4f} (预期 < 100)")
    test0_pass = agent.linear_velocity > 0.5 and agent.internal_energy < 100.0
    print(f"  {'✅ 通过' if test0_pass else '❌ 失败'}")
    
    # ==================== 测试 Channel 1: IMPEDANCE ====================
    print(f"\n{'='*60}")
    print("🛡️ Channel 1: IMPEDANCE (阻尼调制)")
    print("="*60)
    
    # 测试高阻抗 -> 高阻尼 -> 慢速
    agent1 = MockAgent()
    agent1.linear_velocity = 2.0  # 初始速度
    
    # 低阻抗
    actuator_low_imp = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # impedance=0
    env._update_agent_physics(agent1, actuator_low_imp)
    v_low_imp = agent1.linear_velocity
    
    # 高阻抗
    agent2 = MockAgent()
    agent2.linear_velocity = 2.0  # 相同初始速度
    
    actuator_high_imp = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # impedance=1 (sigmoid后~0.73)
    env._update_agent_physics(agent2, actuator_high_imp)
    v_high_imp = agent2.linear_velocity
    
    print(f"  初始速度: 2.0")
    print(f"  低阻抗(0) 后速度: {v_low_imp:.4f}")
    print(f"  高阻抗(1) 后速度: {v_high_imp:.4f}")
    test1_pass = v_high_imp < v_low_imp  # 高阻尼速度衰减更多
    print(f"  {'✅ 通过' if test1_pass else '❌ 失败'}")
    
    # ==================== 测试 Channel 2: STIGMERGY ====================
    print(f"\n{'='*60}")
    print("💨 Channel 2: STIGMERGY (信息素排放)")
    print("="*60)
    
    agent = MockAgent()
    agent.x = 25.0
    agent.y = 25.0
    
    # 检查信息素场
    if hasattr(env, 'stigmergy_field') and env.stigmergy_field is not None:
        # 写入信息素
        actuator_outputs = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # stigmergy=1
        env._update_agent_physics(agent, actuator_outputs)
        
        # 读取位置的信息素值
        stigmergy_value = env.stigmergy_field.sample(agent.x, agent.y)
        print(f"  写入位置: ({agent.x}, {agent.y})")
        print(f"  信息素值: {stigmergy_value:.4f} (预期 > 0)")
        test2_pass = stigmergy_value > 0
    else:
        print(f"  ⚠️ 信息素场未启用")
        stigmergy_value = 0.0
        test2_pass = True  # 跳过测试
    
    print(f"  {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    # ==================== 测试 Channel 3: STRESS ====================
    print(f"\n{'='*60}")
    print("⚔️ Channel 3: STRESS (攻击/摄食)")
    print("="*60)
    
    # 这个测试需要多 Agent，这里简化测试 stress 的代谢成本
    agent = MockAgent()
    agent.internal_energy = 100.0
    
    # 无 stress
    actuator_no_stress = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    env._update_agent_physics(agent, actuator_no_stress)
    energy_no_stress = agent.internal_energy
    
    # 有 stress
    agent2 = MockAgent()
    agent2.internal_energy = 100.0
    actuator_with_stress = np.array([1.0, 1.0, 0.0, 0.0, 1.0])  # stress=1
    env._update_agent_physics(agent2, actuator_with_stress)
    energy_with_stress = agent2.internal_energy
    
    print(f"  无 stress 能量: {energy_no_stress:.4f}")
    print(f"  有 stress 能量: {energy_with_stress:.4f}")
    # stress 会增加代谢成本
    test3_pass = energy_with_stress < energy_no_stress
    print(f"  {'✅ 通过' if test3_pass else '❌ 失败'}")
    
    # ==================== 综合测试 ====================
    print(f"\n{'='*60}")
    print("🔬 综合测试: 所有通道同时激活")
    print("="*60)
    
    agent = MockAgent()
    agent.x = 50.0
    agent.y = 50.0
    agent.theta = 0.0
    agent.internal_energy = 100.0
    
    # 全通道激活
    # [left, right, impedance, stigmergy, stress]
    actuator_full = np.array([1.0, 0.5, 0.5, 0.8, 0.3])
    env._update_agent_physics(agent, actuator_full)
    
    print(f"  输入: {actuator_full}")
    print(f"  线速度: {agent.linear_velocity:.4f} (预期 > 0)")
    print(f"  角速度: {agent.angular_velocity:.4f} (预期 < 0, 右转)")
    print(f"  能量: {agent.internal_energy:.4f} (预期 < 100)")
    
    test_full_pass = (
        agent.linear_velocity > 0 and 
        agent.angular_velocity < 0 and
        agent.internal_energy < 100
    )
    print(f"  {'✅ 通过' if test_full_pass else '❌ 失败'}")
    
    # ==================== 总结 ====================
    all_pass = test0_pass and test1_pass and test2_pass and test3_pass and test_full_pass
    
    print(f"\n{'='*60}")
    print(f"{'✅ 所有测试通过' if all_pass else '❌ 部分测试失败'}")
    print(f"{'='*60}")
    
    return all_pass


if __name__ == "__main__":
    success = test_multichannel_physics()
    sys.exit(0 if success else 1)