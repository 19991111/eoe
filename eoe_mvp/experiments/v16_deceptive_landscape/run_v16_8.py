#!/usr/bin/env python
"""
v16.8 Deceptive Landscape Evolution - 认知溢价强化版
=====================================================

核心策略: 彻底打破"伏地魔"策略

1. 认知溢价: 隐身进食 5x 暴击奖励
2. 拒绝天降馅饼: 能量源曲线运动 + 限制游走范围
3. 适度代谢压力: 迫使Agent主动捕食

目标: 逼迫Agent演化出预测记忆网络
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


def run_v16_8_experiment(
    steps: int = 5000,
    initial_pop: int = 50,
    device: str = 'cuda:0'
):
    print("=" * 70)
    print("🔬 v16.8 欺骗性景观实验 - 认知溢价强化版")
    print("=" * 70)
    print("  目标: 打破伏地魔策略，逼迫预测记忆涌现")
    print("  策略: 隐身5x暴击 + 曲线运动 + 范围限制")
    print("=" * 70)
    
    # ===== 1. 环境配置 =====
    # 同时启用基础EPF和闪烁场
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        # 基础EPF场
        energy_field_enabled=True,
        # 闪烁能量场 (与EPF叠加)
        flickering_energy_enabled=True,
        flickering_period=50,      # 50步可见
        flickering_invisible_moves=100,  # 100步隐身
        flickering_speed=0.8,
        # 其他场关闭
        impedance_field_enabled=False,
        stigmergy_field_enabled=False,
        danger_field_enabled=False,
        matter_grid_enabled=False,
        wind_field_enabled=False,
        seasons_enabled=False,
    )
    
    # 启用隐身期曲线运动
    if hasattr(env.flickering_energy_field, '_invisible_curved'):
        env.flickering_energy_field._invisible_curved = True
        print("  ✅ 隐身期曲线运动已启用")
    
    # 限制能量源游走范围 (新机制!)
    if hasattr(env.flickering_energy_field, 'max_displacement'):
        env.flickering_energy_field.max_displacement = 30.0
        print("  ✅ 能量源游走范围限制: 30单位")
    
    print("  环境就绪")
    
    # ===== 2. Agent配置 =====
    config = PoolConfig()
    
    # 使用随机寒武纪大脑 (不从预训练加载)
    config.PRETRAINED_INIT = False
    config.PRETRAINED_STRUCTURES_FILE = None
    config.CAMBRIAN_INIT = True
    config.CAMBRIAN_MIN_NODES = 4
    config.CAMBRIAN_MAX_NODES = 8
    
    # 认知溢价 (核心!) - v16.9调整
    config.VISIBLE_REWARD_MULTIPLIER = 3.0   # 可见3x (丰水期)
    config.INVISIBLE_REWARD_MULTIPLIER = 5.0 # 隐身5x (旱季奖励)
    config.COGNITIVE_PREMIUM_ONLY_INVISIBLE = True
    config.ENABLE_INVISIBLE_SENSING = True
    
    # 感知效率 (降低门槛)
    config.ACTIVE_SENSING_ENABLED = True
    config.ACTIVE_SENSING_THRESHOLD = 0.1   # 降低到0.1
    config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.3  # 静止也有30%感知
    
    # 代谢配置 - 进一步调低
    config.BASE_METABOLISM = 0.01           # 较低基础代谢
    config.HEBBIAN_ENABLED = True
    config.NONLINEAR_METABOLISM = True
    
    # 繁衍配置
    config.REPRODUCE_ENABLED = True
    config.REPRODUCE_THRESHOLD = 50.0
    
    # 运动惩罚 (禁用以鼓励移动)
    config.MOVEMENT_PENALTY = 0.0
    
    # 随机探索噪声 (帮助发现能量)
    config.EXPLORATION_NOISE = 0.3
    
    print("  配置就绪")
    
    # ===== 3. 创建Agent池 =====
    max_agents = min(initial_pop * 4, 100)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=120.0,  # 更高初始能量
        config=config,
        env=env
    )
    
    # 加载预训练脑结构
    if config.PRETRAINED_INIT and config.PRETRAINED_STRUCTURES_FILE:
        alive_genomes = agents._load_pretrained_genomes(agents.alive_mask.sum().item())
        if alive_genomes:
            print(f"  ✅ 预加载脑结构: {len(alive_genomes)} 种")
        else:
            alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    else:
        alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    
    agents.set_brains(alive_genomes)
    
    # ===== 4. 统计记录 =====
    stats = {
        'steps': [],
        'population': [],
        'avg_nodes': [],
        'complex_structures': [],
        'avg_velocity': [],
        'invisible_feeds': [],  # 隐身进食次数
    }
    
    # ===== 5. 主循环 =====
    print("\n" + "-" * 70)
    print("🚀 开始模拟... (课程学习模式)")
    print("-" * 70)
    
    # ===== 课程学习配置 =====
    CURRICULUM_PHASES = [
        (0, 2500, 1.0, "🏫 基础: 100%可见,学会移动"),      # 更长100%可见
        (2500, 5000, 0.5, "📚 进阶: 50%可见,筛选记忆"),    # 50%可见
        (5000, 8000, 0.3, "🎓 终极: 30%可见,预测挑战"),    # 30%可见
    ]
    current_phase = 0
    
    start_time = time.time()
    invisible_feed_count = 0
    
    for step in range(steps):
        # ===== 课程学习: 动态调整可见性 =====
        if current_phase < len(CURRICULUM_PHASES):
            phase_start, phase_end, target_ratio, phase_name = CURRICULUM_PHASES[current_phase]
            if step >= phase_end and current_phase + 1 < len(CURRICULUM_PHASES):
                current_phase += 1
                _, _, new_ratio, new_name = CURRICULUM_PHASES[current_phase]
                # 调整闪烁周期比例
                fe = env.flickering_energy_field
                total = fe.flicker_period + fe.invisible_moves
                fe.flicker_period = max(5, int(total * new_ratio))
                fe.invisible_moves = total - fe.flicker_period
                print(f"\n  📢 [课程升级] Step {step}: {new_name}")
        
        # 大脑前向函数 (每步手动调用) + 探索噪声
        def brain_fn(batch):
            sensors = agents.get_sensors(env)
            outputs = agents.forward_brains(sensors)
            # 添加探索噪声帮助发现能量
            noise = torch.randn_like(outputs) * 0.3
            return outputs + noise
        
        env.step()
        step_stats = agents.step(env=env, dt=0.1, brain_fn=brain_fn)
        
        # 追踪隐身进食 (简化)
        if 'invisible_feeds' in step_stats:
            invisible_feed_count += step_stats['invisible_feeds']
        
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
                stats['complex_structures'].append(complex_count)
                
                # 速度
                alive_velocities = agents.state.linear_velocity[alive_indices]
                avg_vel = alive_velocities.norm(dim=-1).mean().item()
                stats['avg_velocity'].append(avg_vel)
            
            # 打印进度
            energy_field = env.flickering_energy_field
            is_visible = (energy_field.step_count % (energy_field.flicker_period + energy_field.invisible_moves)) < energy_field.flicker_period
            
            elapsed = time.time() - start_time
            print(f"  Step {step+1:5d} | 存活: {n_alive:3d} | "
                  f"节点: {avg_nodes:.1f} | 复杂: {complex_count:2d} | "
                  f"速度: {avg_vel:.3f} | {'可见' if is_visible else '隐身'} | "
                  f"Time: {elapsed:.1f}s")
        
        # 检查种群灭绝
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
    print(f"  平均速度:   {np.mean(stats['avg_velocity'][-5:]):.3f}" if stats['avg_velocity'] else "  N/A")
    print(f"  模拟耗时:   {elapsed:.1f}秒")
    print(f"  平均速度:   {(step+1)/elapsed:.1f}步/秒")
    
    # 保存结果
    results = {
        'version': 'v16.8',
        'config': {
            'visible_reward': config.VISIBLE_REWARD_MULTIPLIER,
            'invisible_reward': config.INVISIBLE_REWARD_MULTIPLIER,
            'base_metabolism': config.BASE_METABOLISM,
            'active_sensing_threshold': config.ACTIVE_SENSING_THRESHOLD,
        },
        'stats': stats,
        'final': {
            'step': step + 1,
            'population': agents.alive_mask.sum().item(),
            'avg_nodes': stats['avg_nodes'][-1] if stats['avg_nodes'] else 0,
        }
    }
    
    output_path = "experiments/v16_deceptive_landscape/v16_8_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    run_v16_8_experiment(args.steps, args.pop, args.device)