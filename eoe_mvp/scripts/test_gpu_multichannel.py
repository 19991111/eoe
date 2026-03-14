#!/usr/bin/env python3
"""
v13.2 GPU 批量多通道测试 (5 通道模型)
======================================
测试新的 5 通道语义:
- Channel 0: THRUST (推力)
- Channel 1: ARMOR (装甲)
- Channel 2: PHEROMONE (信息素)
- Channel 3: FEED (摄食)
- Channel 4: ATTACK (攻击/危险场)
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from core.eoe.batched_agents import BatchedAgents


def test_gpu_multichannel():
    """测试 GPU 批量 Agent 的 5 通道物理"""
    
    print("="*60)
    print("🧪 v13.2 GPU 批量 5 通道测试")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建批量 Agent
    n_agents = 8
    agents = BatchedAgents(
        n_agents=n_agents,
        env_width=100,
        env_height=100,
        device=device,
        init_energy=100.0
    )
    
    # 设置初始位置 (围成一圈)
    angles = torch.linspace(0, 2*np.pi, n_agents + 1)[:-1].to(device)
    agents.state.positions[:, 0] = 50 + 10 * torch.cos(angles)
    agents.state.positions[:, 1] = 50 + 10 * torch.sin(angles)
    agents.state.thetas = angles
    
    # 准备基础数据 - 5 通道
    # 基础推力数据 (用于测试 thrust 和 angular)
    base_actuator = torch.tensor([
        [1.0, 1.0],   # 前进
        [1.0, -1.0],  # 右转
        [-1.0, -1.0], # 倒车
        [1.0, 0.5],   # 左转
        [0.5, 1.0],   # 右转
        [-1.0, 1.0],  # 原地左转
        [0.0, 0.0],   # 静止
        [0.5, 0.5],   # 慢速前进
    ], device=device)
    
    # 对称推力数据 (用于测试 armor)
    symmetric_actuator = torch.tensor([
        [1.0, 1.0],   # 前进
        [1.0, 1.0],   # 前进 (对称)
        [-1.0, -1.0], # 倒车
        [1.0, 0.5],   # 左转
        [0.5, 1.0],   # 右转
        [-1.0, 1.0],  # 原地左转
        [0.0, 0.0],   # 静止
        [0.5, 0.5],   # 慢速前进
    ], device=device)
    
    emitter_keys = torch.zeros(n_agents, 2, 5, device=device)  # 5 channels
    emitter_keys[:, :, 0] = 1.0  # Channel 0: THRUST
    
    offsets = torch.tensor([
        [[0.0, 1.0], [0.0, -1.0]],
    ] * n_agents, device=device)
    
    # ==================== 测试 1: Channel 0 THRUST ====================
    print(f"\n{'='*60}")
    print("🔌 测试1: Channel 0 THRUST (推力)")
    print("="*60)
    
    linear_accel, angular_accel, channel_outputs = agents.decode_actuators(
        base_actuator, emitter_keys, offsets
    )
    
    print(f"  线加速度: {linear_accel.cpu().numpy()}")
    print(f"  角加速度: {angular_accel.cpu().numpy()}")
    
    test1_pass = (
        linear_accel[0].item() > 0 and   # 前进
        linear_accel[2].item() < 0 and   # 倒车
        angular_accel[1].item() < 0 and  # 右转
        angular_accel[5].item() > 0      # 左转
    )
    print(f"  {'✅ 通过' if test1_pass else '❌ 失败'}")
    
    # ==================== 测试 2: Channel 1 ARMOR ====================
    print(f"\n{'='*60}")
    print("🛡️ 测试2: Channel 1 ARMOR (装甲)")
    print("="*60)
    
    # 不同的 emitter_keys 用于装甲测试
    emitter_keys_armor = torch.zeros(n_agents, 2, 5, device=device)
    emitter_keys_armor[:, :, 0] = 1.0  # Channel 0
    emitter_keys_armor[0, :, 1] = 0.0  # 无装甲
    emitter_keys_armor[1, :, 1] = 1.0  # 高装甲
    
    linear_accel1, angular_accel1, channel_outputs1 = agents.decode_actuators(
        symmetric_actuator, emitter_keys_armor, offsets
    )
    
    # 重置速度
    agents.state.linear_velocity = torch.zeros(n_agents, device=device)
    agents.state.linear_velocity[0] = 2.0
    agents.state.linear_velocity[1] = 2.0
    
    armor = channel_outputs1.get('armor')
    agents.apply_actuator_physics(linear_accel1, angular_accel1, 0.1, impedance=armor)
    
    print(f"  无装甲 Agent 0 armor: {armor[0].item():.4f}")
    print(f"  高装甲 Agent 1 armor: {armor[1].item():.4f}")
    
    test2_pass = armor[1].item() > armor[0].item()
    print(f"  {'✅ 通过' if test2_pass else '❌ 失败'}")
    
    # ==================== 测试 3: Channel 2 PHEROMONE ====================
    print(f"\n{'='*60}")
    print("💨 测试3: Channel 2 PHEROMONE (信息素)")
    print("="*60)
    
    class MockStigmergyField:
        def __init__(self):
            self.data = {}
        
        def deposit_batch(self, positions, amounts):
            for i in range(len(positions)):
                key = (positions[i, 0].item(), positions[i, 1].item())
                self.data[key] = amounts[i].item()
    
    mock_env = MockStigmergyField()
    
    emitter_keys_phero = torch.zeros(n_agents, 2, 5, device=device)
    emitter_keys_phero[:, :, 0] = 1.0
    emitter_keys_phero[:, :, 2] = 1.0  # Channel 2
    
    actuator_outputs_phero = torch.zeros(n_agents, 2, device=device)
    actuator_outputs_phero[:, 0] = 1.0
    
    _, _, channel_outputs2 = agents.decode_actuators(
        actuator_outputs_phero, emitter_keys_phero, offsets
    )
    
    # 应用通道环境
    class MockEnv:
        stigmergy_field = MockStigmergyField()
        danger_field = None
        energy_field = None
    
    agents.apply_channel_environment(channel_outputs2, MockEnv())
    
    print(f"  写入的信息素点: {len(MockEnv.stigmergy_field.data)}")
    test3_pass = len(MockEnv.stigmergy_field.data) > 0
    print(f"  {'✅ 通过' if test3_pass else '❌ 失败'}")
    
    # ==================== 测试 4: Channel 3 FEED + Channel 4 ATTACK ====================
    print(f"\n{'='*60}")
    print("🍽️ 测试4: Channel 3 FEED (摄食) + Channel 4 ATTACK (攻击)")
    print("="*60)
    
    # 重置位置和能量
    agents.state.positions[:, 0] = 50.0
    agents.state.positions[:, 1] = 50.0
    agents.state.energies = torch.tensor([100.0] * n_agents, device=device)
    
    emitter_keys_feed = torch.zeros(n_agents, 2, 5, device=device)
    emitter_keys_feed[:, :, 0] = 1.0  # THRUST
    emitter_keys_feed[:, :, 3] = 1.0  # FEED
    
    actuator_outputs_feed = torch.zeros(n_agents, 2, device=device)
    actuator_outputs_feed[0, 0] = 1.0  # Agent 0 进食
    
    _, _, channel_outputs3 = agents.decode_actuators(
        actuator_outputs_feed, emitter_keys_feed, offsets
    )
    
    print(f"  进食者 feed: {channel_outputs3['feed'][0].item():.4f}")
    print(f"  攻击者 attack: {channel_outputs3['attack'][0].item():.4f}")
    
    # 初始能量
    initial_energy = agents.state.energies.clone()
    
    # 创建带能量场的环境
    class MockEnergyField:
        def sample_batch(self, positions):
            return torch.ones(len(positions), device=positions.device) * 50.0  # 高能量场
    
    class MockEnvWithEnergy:
        stigmergy_field = None
        danger_field = None
        energy_field = MockEnergyField()
    
    agents.apply_channel_environment(channel_outputs3, MockEnvWithEnergy())
    
    energy_change = agents.state.energies - initial_energy
    
    print(f"  进食者能量变化: {energy_change[0].item():.4f}")
    
    test4_pass = energy_change[0] > 0  # 应该获取能量
    print(f"  {'✅ 通过' if test4_pass else '❌ 失败'}")
    
    # ==================== 测试 5: 完整 step_embodied ====================
    print(f"\n{'='*60}")
    print("🔬 测试5: 完整 step_embodied (5 通道)")
    print("="*60)
    
    # 重置
    agents.state.positions[:, 0] = 50.0
    agents.state.positions[:, 1] = 50.0
    agents.state.thetas = torch.zeros(n_agents, device=device)
    agents.state.linear_velocity = torch.zeros(n_agents, device=device)
    agents.state.angular_velocity = torch.zeros(n_agents, device=device)
    agents.state.energies = torch.tensor([100.0] * n_agents, device=device)
    agents.state.is_alive = torch.ones(n_agents, dtype=torch.bool, device=device)
    
    # 全通道激活
    emitter_keys_full = torch.zeros(n_agents, 2, 5, device=device)
    emitter_keys_full[:, :, 0] = 1.0  # THRUST
    emitter_keys_full[:, :, 1] = 0.5  # ARMOR
    emitter_keys_full[:, :, 2] = 0.3  # PHEROMONE
    emitter_keys_full[:, :, 3] = 0.2  # FEED
    emitter_keys_full[:, :, 4] = 0.1  # ATTACK
    
    actuator_full = torch.ones(n_agents, 2, device=device) * 0.5
    
    result = agents.step_embodied(
        actuator_full, 
        emitter_keys_full, 
        offsets, 
        dt=0.1
    )
    
    print(f"  推力: {result['linear_accel'].cpu().numpy()[:3]}")
    print(f"  装甲: {result.get('channel_armor', torch.zeros(3)).cpu().numpy()}")
    print(f"  信息素: {result.get('channel_pheromone', torch.zeros(3)).cpu().numpy()}")
    print(f"  摄食: {result.get('channel_feed', torch.zeros(3)).cpu().numpy()}")
    print(f"  攻击: {result.get('channel_attack', torch.zeros(3)).cpu().numpy()}")
    
    test5_pass = (
        result['linear_accel'].abs().max().item() > 0
    )
    print(f"  {'✅ 通过' if test5_pass else '❌ 失败'}")
    
    # ==================== 总结 ====================
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass
    
    print(f"\n{'='*60}")
    print(f"{'✅ 所有测试通过' if all_pass else '❌ 部分测试失败'}")
    print(f"{'='*60}")
    
    return all_pass


if __name__ == "__main__":
    success = test_gpu_multichannel()
    sys.exit(0 if success else 1)