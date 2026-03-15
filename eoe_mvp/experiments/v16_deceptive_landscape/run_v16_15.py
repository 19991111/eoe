#!/usr/bin/env python
"""
v16.15 Deceptive Landscape - 巩固增强版
============================================

核心策略: 延长巩固期 + 提高能量预算 + 更平滑曲线

v16.14发现: 种群稳定但复杂度下降，"降级求生"局部最优
解决方案: 
1. 五阶段更平滑曲线，50%阶段延长至1200步
2. 提高初始能量和能量上限
3. 降低基础代谢，让复杂大脑不那么"昂贵"

课程学习阶段 (v16.15):
- 阶段I (0-1000):   80% 可见 - 伊甸园
- 阶段II (1000-1800): 60% 可见 - 过渡
- 阶段III (1800-3000): 50% 可见 - 巩固期延长！
- 阶段IV (3000-4500): 40% 可见 - 进阶
- 阶段V (4500-6000): 30% 可见 - 终极

目标: 稳定保持6+节点种群，涌现SuperNode
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import numpy as np
import time
import json

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.energy_audit import create_energy_audit_hook


def run_v16_15_experiment(
    steps: int = 6000,
    initial_pop: int = 50,
    device: str = 'cuda:0'
):
    print("=" * 70)
    print("🔬 v16.15 欺骗性景观实验 - 巩固增强版")
    print("=" * 70)
    print("  目标: 稳定6+节点种群，涌现SuperNode")
    print("  策略: 五阶段平滑曲线 + 高能量预算 + 低代谢")
    print("  80%→60%→50%(延长)→40%→30%")
    print("  核心理念: 巩固期延长，让模式固化")
    print("=" * 70)
    
    # ===== 1. 环境配置 =====
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
        flickering_energy_enabled=True,
        flickering_period=30,         # 30步可见 (80% of 37.5)
        flickering_invisible_moves=7,  # 7步隐身
        flickering_speed=0.5,         # 稍微放慢，让轨迹更可预测
        # 关闭其他场
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
        seasons_enabled=False,
    )
    
    # 启用可预测圆形运动
    if hasattr(env.flickering_energy_field, 'set_circular_motion'):
        env.flickering_energy_field.set_circular_motion(True)
        print("  ✅ 能量源圆形轨迹已启用")
    
    # 限制游走范围 (稍微扩大，让预测更有价值)
    if hasattr(env.flickering_energy_field, 'max_displacement'):
        env.flickering_energy_field.max_displacement = 30.0
        print("  ✅ 能量源游走范围: 30单位")
    
    print("  环境就绪")
    
    # ===== 2. Agent配置 =====
    config = PoolConfig()
    
    # 寒武纪随机大脑 (稍微提高最大节点数)
    config.PRETRAINED_INIT = False
    config.CAMBRIAN_INIT = True
    config.CAMBRIAN_MIN_NODES = 4
    config.CAMBRIAN_MAX_NODES = 12  # 从10提高到12
    
    # 认知溢价 (增强版 - 给高复杂度更多奖励)
    config.VISIBLE_REWARD_MULTIPLIER = 2.0   # 可见2x
    config.INVISIBLE_REWARD_MULTIPLIER = 6.0 # 隐身6x (从5x提高)
    config.COGNITIVE_PREMIUM_ONLY_INVISIBLE = True
    config.ENABLE_INVISIBLE_SENSING = True
    
    # 主动感知
    config.ACTIVE_SENSING_ENABLED = True
    config.ACTIVE_SENSING_THRESHOLD = 0.04   # 更敏感
    config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.25
    
    # 代谢 (降低 - 让复杂大脑不那么昂贵)
    config.BASE_METABOLISM = 0.012  # 从0.015降到0.012
    config.HEBBIAN_ENABLED = True
    config.NONLINEAR_METABOLISM = True
    
    # 繁衍 (稍微提高阈值，让优质基因更易传播)
    config.REPRODUCE_ENABLED = True
    config.REPRODUCE_THRESHOLD = 50.0  # 从45提高到50
    
    # 运动惩罚 (保持轻惩罚)
    config.MOVEMENT_PENALTY = 0.005
    
    # 探索噪声 (稍微降低，让行为更稳定)
    config.EXPLORATION_NOISE = 0.35  # 从0.4降到0.35
    
    print("  配置就绪")
    
    # ===== 3. 课程学习配置 (v16.15 - 五阶段巩固版) =====
    CURRICULUM_PHASES = [
        (0, 1000, 0.80, "🌱 伊甸园: 80%可见"),        # 宽松环境，让种群繁荣
        (1000, 1800, 0.60, "🌿 过渡: 60%可见"),      # 轻度过渡
        (1800, 3000, 0.50, "🌊 巩固: 50%可见"),      # 关键！延长至1200步
        (3000, 4500, 0.40, "🎓 进阶: 40%可见"),      # 中度筛选
        (4500, 6000, 0.30, "🔥 终极: 30%可见"),      # 终极挑战
    ]
    current_phase = 0
    
    # 打印初始阶段
    _, _, init_ratio, init_name = CURRICULUM_PHASES[0]
    print(f"\n  📚 初始: {init_name}")
    
    # ===== 4. 创建Agent池 =====
    max_agents = min(initial_pop * 4, 100)
    
    # 提高初始能量 (从70提高到90)
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=90.0,  # 从70提高到90
        config=config,
        env=env
    )
    
    # 创建大脑
    alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    agents.set_brains(alive_genomes)
    
    # ===== 5. 统计记录 =====
    stats = {
        'steps': [],
        'population': [],
        'avg_nodes': [],
        'complex_structures': [],
        'super_structures': [],  # 7+节点
        'avg_velocity': [],
        'visible_ratio': [],
    }
    
    print("\n" + "-" * 70)
    print("🚀 开始模拟...")
    print("-" * 70)
    
    start_time = time.time()
    
    for step in range(steps):
        # 课程学习: 动态调整可见性
        if current_phase < len(CURRICULUM_PHASES):
            phase_start, phase_end, target_ratio, phase_name = CURRICULUM_PHASES[current_phase]
            if step >= phase_end and current_phase + 1 < len(CURRICULUM_PHASES):
                current_phase += 1
                _, _, new_ratio, new_name = CURRICULUM_PHASES[current_phase]
                fe = env.flickering_energy_field
                total = fe.flicker_period + fe.invisible_moves
                fe.flicker_period = max(5, int(total * new_ratio))
                fe.invisible_moves = total - fe.flicker_period
                print(f"\n  📢 [课程升级] Step {step}: {new_name}")
        
        # 大脑前向 + 探索噪声
        def brain_fn(batch):
            sensors = agents.get_sensors(env)
            outputs = agents.forward_brains(sensors)
            noise = torch.randn_like(outputs) * config.EXPLORATION_NOISE
            return outputs + noise
        
        env.step()
        step_stats = agents.step(env=env, dt=0.1, brain_fn=brain_fn)
        
        # 定期记录
        if (step + 1) % 200 == 0:
            n_alive = agents.alive_mask.sum().item()
            stats['steps'].append(step + 1)
            stats['population'].append(n_alive)
            
            if n_alive > 0:
                alive_indices = agents.alive_mask.nonzero(as_tuple=True)[0]
                avg_nodes = agents.state.node_counts[alive_indices].float().mean().item()
                stats['avg_nodes'].append(avg_nodes)
                complex_count = (agents.state.node_counts[alive_indices] > 4).sum().item()
                super_count = (agents.state.node_counts[alive_indices] > 6).sum().item()
                stats['complex_structures'].append(complex_count)
                stats['super_structures'].append(super_count)
                
                alive_velocities = agents.state.linear_velocity[alive_indices]
                avg_vel = alive_velocities.norm(dim=-1).mean().item()
                stats['avg_velocity'].append(avg_vel)
            
            # 记录当前可见性
            fe = env.flickering_energy_field
            current_ratio = fe.flicker_period / (fe.flicker_period + fe.invisible_moves)
            stats['visible_ratio'].append(current_ratio)
            
            is_visible = (fe.step_count % (fe.flicker_period + fe.invisible_moves)) < fe.flicker_period
            
            elapsed = time.time() - start_time
            print(f"  Step {step+1:5d} | 存活: {n_alive:3d} | "
                  f"节点: {avg_nodes:.1f} | 复杂: {complex_count:2d} | "
                  f"超级: {super_count:2d} | 可见:{current_ratio*100:.0f}% | {elapsed:.1f}s")
        
        if agents.alive_mask.sum().item() == 0:
            print(f"\n⚠️ 种群在 Step {step+1} 灭绝!")
            break
    
    elapsed = time.time() - start_time
    
    # ===== 6. 结果报告 =====
    print("\n" + "=" * 70)
    print("📊 实验完成")
    print("=" * 70)
    print(f"  运行步数:   {step + 1}")
    print(f"  最终种群:   {agents.alive_mask.sum().item()}")
    print(f"  平均节点:   {np.mean(stats['avg_nodes'][-5:]):.2f}" if stats['avg_nodes'] else "  N/A")
    print(f"  复杂结构:   {stats['complex_structures'][-1] if stats['complex_structures'] else 0}")
    print(f"  超级结构:   {stats['super_structures'][-1] if stats['super_structures'] else 0}")
    print(f"  平均速度:   {np.mean(stats['avg_velocity'][-5:]):.3f}" if stats['avg_velocity'] else "N/A")
    print(f"  模拟耗时:   {elapsed:.1f}秒")
    print(f"  步速:       {(step+1)/elapsed:.1f}步/秒")
    
    # 保存结果
    results = {
        'version': 'v16.15',
        'config': {
            'curriculum_learning': True,
            'phase_1_ratio': 0.80,
            'phase_2_ratio': 0.60,
            'phase_3_ratio': 0.50,
            'phase_4_ratio': 0.40,
            'phase_5_ratio': 0.30,
            'visible_reward': config.VISIBLE_REWARD_MULTIPLIER,
            'invisible_reward': config.INVISIBLE_REWARD_MULTIPLIER,
            'base_metabolism': config.BASE_METABOLISM,
            'init_energy': 90.0,
            'exploration_noise': config.EXPLORATION_NOISE,
        },
        'stats': stats,
        'final': {
            'step': step + 1,
            'population': agents.alive_mask.sum().item(),
            'avg_nodes': stats['avg_nodes'][-1] if stats['avg_nodes'] else 0,
        }
    }
    
    output_path = "experiments/v16_deceptive_landscape/v16_15_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    run_v16_15_experiment(
        steps=args.steps,
        initial_pop=args.pop,
        device=args.device
    )