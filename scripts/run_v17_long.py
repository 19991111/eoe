#!/usr/bin/env python3
"""
v17 Long Experiment - 冰河世纪协议
===================================
运行长时间演化实验，测试冰河世纪协议效果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'scripts')

import torch
import numpy as np
from configs import PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.batched_agents import BatchedAgents
from core.eoe.manifest import PhysicsManifest


class V17LongConfig(PoolConfig):
    """v17.5 长实验配置 - 冰河世纪协议"""
    
    # ========== 基础配置 ==========
    DEVICE = 'cuda'
    MAX_AGENTS = 500
    
    # 关闭所有复杂机制（最小测试）
    SOFT_CARRYING_CAP = False
    CROWDING_PENALTY_ENABLED = False
    SEASONS_ENABLED = False
    
    # ========== 代谢 (极低) ==========
    BASE_METABOLISM = 0.001  # 极低代谢
    MOVEMENT_PENALTY = 0.0   # 无移动惩罚
    
    # ========== 能量源 (增加数量) ==========
    ENERGY_SOURCES = 30      # 从5增加到30
    
    # ========== 繁殖 (保守) ==========
    REPRODUCTION_THRESHOLD = 45.0  # 降低阈值
    
    # ========== 启用机制 ==========
    CAMBRIAN_INIT = True
    METABOLIC_GRACE = True
    
    # ========== 捕食 (关闭) ==========
    PREDATION_ENABLED = False
    
    # ========== v17.5 冰河世纪协议 (正式版) ==========
    ICE_AGE_ENABLED = False               # 暂时关闭
    ICE_AGE_START_STEP = 2000
    ENERGY_DYNAMIC_ENABLED = False        # 暂时关闭
    ENERGY_MOVE_INTERVAL = 3
    ENERGY_JUMP_PROB = 0.4
    ENERGY_JUMP_DIST = 15.0
    KIF_STORM_ENABLED = False             # 暂时关闭
    KIF_STORM_COUNT = 5
    KIF_STORM_INTENSITY = 800.0
    KIF_STORM_MOVE_SPEED = 1.0
    KIF_STORM_RADIUS = 15.0
    
    # 禁用SuperNode
    SUPERNODE_ENABLED = False
    
    # 关闭季节（测试用）
    SEASONS_ENABLED = False
    
    # 关闭捕食（测试用）
    PREDATION_ENABLED = False
    
    # 关闭Stigmergy（测试用）
    STIGMERGY_ENABLED = False


def run_experiment(steps=5000, save_interval=500):
    """运行v17.5长实验 - 冰河世纪协议"""
    
    print("=" * 60)
    print("v17.5 Long Experiment - Ice Age Protocol")
    print("=" * 60)
    
    # 初始化配置
    config = V17LongConfig()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n=== 关键配置 ===")
    print(f"DEVICE: {config.DEVICE}")
    print(f"BASE_METABOLISM: {config.BASE_METABOLISM}")
    print(f"MOVEMENT_PENALTY: {config.MOVEMENT_PENALTY}")
    print(f"ICE_AGE_ENABLED: {config.ICE_AGE_ENABLED}")
    print(f"ICE_AGE_START_STEP: {config.ICE_AGE_START_STEP}")
    print(f"ENERGY_DYNAMIC_ENABLED: {config.ENERGY_DYNAMIC_ENABLED}")
    print(f"KIF_STORM_ENABLED: {config.KIF_STORM_ENABLED}")
    
    # 初始化环境
    print("\n=== 初始化环境 ===")
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=config.DEVICE,
        # v17.5: 冰河世纪协议原生支持
        ice_age_enabled=config.ICE_AGE_ENABLED,
        ice_age_start_step=config.ICE_AGE_START_STEP,
        energy_dynamic_enabled=config.ENERGY_DYNAMIC_ENABLED,
        energy_move_interval=config.ENERGY_MOVE_INTERVAL,
        energy_jump_prob=config.ENERGY_JUMP_PROB,
        energy_jump_dist=config.ENERGY_JUMP_DIST,
        kif_storm_enabled=config.KIF_STORM_ENABLED,
        kif_storm_count=config.KIF_STORM_COUNT,
        kif_storm_intensity=config.KIF_STORM_INTENSITY,
        kif_storm_move_speed=config.KIF_STORM_MOVE_SPEED,
        kif_storm_radius=config.KIF_STORM_RADIUS,
    )
    
    # v17.6: 增加能量源数量
    if hasattr(config, 'ENERGY_SOURCES'):
        n_sources = config.ENERGY_SOURCES
        env.energy_field.n_sources = n_sources
        env.energy_field.sources = torch.zeros(n_sources, 6, device=env.energy_field.device)
        env.energy_field.sources[:, 0] = torch.rand(n_sources, device=env.energy_field.device) * 100  # x
        env.energy_field.sources[:, 1] = torch.rand(n_sources, device=env.energy_field.device) * 100  # y
        env.energy_field.sources[:, 2] = torch.rand(n_sources, device=env.energy_field.device) * 100 + 100  # strength
        env.energy_field.sources[:, 3] = 1.0  # active
        env.energy_field.sources[:, 4] = 500.0  # capacity
        env.energy_field.sources[:, 5] = 500.0  # max_capacity
        print(f"  ✅ 能量源数量: {n_sources}")
    
    # 预热环境
    print("预热环境...")
    for _ in range(10):
        env.step()
    
    # 调整能量场
    env.energy_field.source_capacity = 5000.0
    env.energy_field.decay_rate = 1.0
    
    # ============================================================
    # v17.5: 冰河世纪协议 - 原生支持
    # ============================================================
    if config.ICE_AGE_ENABLED:
        print("\n=== 使用原生冰河世纪协议 ===")
    else:
        print("\n=== 静态能量源模式 ===")
    
    # 初始化智能体池
    print("初始化智能体池...")
    agents = BatchedAgents(
        initial_population=40,
        max_agents=config.MAX_AGENTS,
        env_width=100.0,
        env_height=100.0,
        device=config.DEVICE,
        config=config,
        env=env
    )
    
    # 预热
    for _ in range(10):
        agents.step(env)
    
    # 获取存活数
    alive_count = agents.alive_mask.sum().item()
    print(f"初始种群: {alive_count}")
    
    # 记录指标
    history = {
        'steps': [],
        'population': [],
        'avg_nodes': [],
        'complex_structures': [],
        'supernodes': []
    }
    
    # 记录初始状态
    history['steps'].append(0)
    history['population'].append(alive_count)
    batch = agents.get_active_batch()
    # 计算平均节点数 (从brain_masks)
    if agents.brain_masks is not None and alive_count > 0:
        nodes_per_agent = (agents.brain_masks[agents.alive_mask].abs().sum(dim=2) > 0).sum(dim=1).float()
        history['avg_nodes'].append(nodes_per_agent.mean().item())
        history['complex_structures'].append((nodes_per_agent > 4).sum().item())
    else:
        history['avg_nodes'].append(0)
        history['complex_structures'].append(0)
    
    
    # 开始训练
    print(f"\n=== 开始训练 ({steps}步) ===")
    import time
    start_time = time.time()
    
    for step in range(steps):
        agents.step(env)
        
        # 每100步打印状态
        if (step + 1) % 100 == 0:
            pop = agents.alive_mask.sum().item()
            avg_nodes = (agents.brain_masks[agents.alive_mask].abs().sum(dim=2) > 0).sum(dim=1).float().mean().item()
            complex_count = ((agents.brain_masks[agents.alive_mask].abs().sum(dim=2) > 0).sum(dim=1) > 4).sum().item()
            
            print(f"Step {step+1:5d} | 种群: {pop:3d} | 节点: {avg_nodes:.2f} | 复杂: {complex_count:3d} | {time.time()-start_time:.1f}s")
            
            # 记录历史
            history['steps'].append(step + 1)
            history['population'].append(pop)
            history['avg_nodes'].append(avg_nodes)
            history['complex_structures'].append(complex_count)
            
    
    # 实验结束
    total_time = time.time() - start_time
    final_pop = agents.alive_mask.sum().item()
    final_nodes = (agents.brain_masks[agents.alive_mask].abs().sum(dim=2) > 0).sum(dim=1).float().mean().item()
    final_complex = ((agents.brain_masks[agents.alive_mask].abs().sum(dim=2) > 0).sum(dim=1) > 4).sum().item()
    
    print(f"\n=== 实验完成 ===")
    print(f"总步数: {steps}")
    print(f"总时间: {total_time:.1f}s")
    print(f"最终种群: {final_pop}")
    print(f"最终平均节点: {final_nodes:.2f}")
    print(f"复杂结构数: {final_complex}")
    
    return history, config


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--save-interval', type=int, default=500)
    args = parser.parse_args()
    
    run_experiment(steps=args.steps, save_interval=args.save_interval)