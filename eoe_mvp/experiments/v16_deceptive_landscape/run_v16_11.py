#!/usr/bin/env python
"""
v16.9 Deceptive Landscape - 定居者税版
======================================

核心策略: 打破"伏地魔"策略

1. 定居者税: 静止超过50步 = 额外代谢惩罚
2. 可预测轨迹: 能量源做圆形运动 (预测有价值)
3. 课程学习: 逐步增加难度

目标: 逼迫Agent演化出持续运动能力 + 预测记忆
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


class StationaryTaxModifier:
    """定居者税: 静止Agent额外扣能量
    
    使用位置变化来检测是否静止（更准确）
    """
    
    def __init__(
        self,
        threshold_dist: float = 0.05,  # 移动距离低于此值视为"静止"
        tax_steps: int = 20,            # 连续静止多少步后开始征税
        tax_rate: float = 0.01,         # 每步额外消耗
    ):
        self.threshold_dist = threshold_dist
        self.tax_steps = tax_steps
        self.tax_rate = tax_rate
        
        # 跟踪每个Agent的静止步数
        self.stationary_counter = None
        self.prev_positions = None
        self.max_agents = 0
        
    def reset(self, max_agents: int, device: str):
        self.max_agents = max_agents
        self.stationary_counter = torch.zeros(max_agents, device=device)
        self.prev_positions = torch.zeros(max_agents, 2, device=device)
        
    def apply(self, agents):
        """基于位置变化征税
        
        对连续静止>tax_steps步的Agent额外征税
        """
        if self.stationary_counter is None or self.max_agents == 0:
            return
            
        # 获取当前位置和存活状态
        positions = agents.state.positions  # [max_agents, 2]
        alive_mask = agents.alive_mask  # [max_agents]
        
        # 计算位置变化
        if self.prev_positions is None:
            self.prev_positions = positions.clone()
            return
        
        displacement = positions - self.prev_positions
        dist_moved = torch.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2 + 1e-8)
        
        # 低于阈值 = 静止
        is_stationary = dist_moved < self.threshold_dist
        
        # 更新静止计数
        self.stationary_counter = torch.where(
            is_stationary,
            self.stationary_counter + 1,
            torch.zeros_like(self.stationary_counter)
        )
        
        # 超过阈值步数 -> 征税
        should_tax = (self.stationary_counter >= self.tax_steps) & alive_mask.bool()
        
        # 应用税收
        tax_mask = should_tax.float()
        agents.state.energies -= tax_mask * self.tax_rate
        
        # 更新上一帧位置
        self.prev_positions = positions.clone()


def run_v16_11_experiment(
    steps: int = 6000,
    initial_pop: int = 50,
    device: str = 'cuda:0'
):
    print("=" * 70)
    print("🔬 v16.11 欺骗性景观实验 - 认知压力增强版")
    print("=" * 70)
    print("  目标: 逼出更复杂脑结构 (>8节点)")
    print("  策略: 高认知溢价 + 更高代谢 + 课程学习")
    print("  关键: invisible 6x奖励 + 更高代谢迫使其频繁进食")
    print("=" * 70)
    
    # ===== 1. 环境配置 =====
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=device,
        energy_field_enabled=True,
        flickering_energy_enabled=True,
        flickering_period=30,        # 30步可见
        flickering_invisible_moves=70,  # 70步隐身
        flickering_speed=0.6,        # 较慢速度
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
    
    # 限制游走范围
    if hasattr(env.flickering_energy_field, 'max_displacement'):
        env.flickering_energy_field.max_displacement = 25.0
        print("  ✅ 能量源游走范围: 25单位")
    
    print("  环境就绪")
    
    # ===== 2. Agent配置 =====
    config = PoolConfig()
    
    # 寒武纪随机大脑
    config.PRETRAINED_INIT = False
    config.CAMBRIAN_INIT = True
    config.CAMBRIAN_MIN_NODES = 4
    config.CAMBRIAN_MAX_NODES = 10
    
    # 认知溢价 (平衡版)
    config.VISIBLE_REWARD_MULTIPLIER = 2.0   # 可见2x
    config.INVISIBLE_REWARD_MULTIPLIER = 5.0 # 隐身5x (从4x提高)
    config.COGNITIVE_PREMIUM_ONLY_INVISIBLE = True
    config.ENABLE_INVISIBLE_SENSING = True
    
    # 主动感知
    config.ACTIVE_SENSING_ENABLED = True
    config.ACTIVE_SENSING_THRESHOLD = 0.05   # 更敏感
    config.ACTIVE_SENSING_MIN_EFFICIENCY = 0.2
    
    # 代谢 (适度提高)
    config.BASE_METABOLISM = 0.015  # 从0.012提高到0.015 (温和)
    config.HEBBIAN_ENABLED = True
    config.NONLINEAR_METABOLISM = True
    
    # 繁衍
    config.REPRODUCE_ENABLED = True
    config.REPRODUCE_THRESHOLD = 45.0
    
    # 运动 (轻惩罚，课程学习后逐渐增加)
    config.MOVEMENT_PENALTY = 0.005
    
    # 探索噪声
    config.EXPLORATION_NOISE = 0.4
    
    print("  配置就绪")
    
    # ===== 3. 初始化定居者税 (v16.12微调版) =====
    stationary_tax = StationaryTaxModifier(
        threshold_dist=0.03,  # 更宽容 - 允许微小移动
        tax_steps=25,         # 连续25步不动开始征税
        tax_rate=0.003,       # 大幅降低到0.003 (从0.01)
    )
    
    # ===== 4. 创建Agent池 =====
    max_agents = min(initial_pop * 4, 100)
    
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100.0,
        env_height=100.0,
        device=device,
        init_energy=70.0,   # 适度降低
        config=config,
        env=env
    )
    
    # 初始化定居者税跟踪
    stationary_tax.reset(max_agents, device)
    
    # 创建大脑
    alive_genomes = agents._create_cambrian_genomes(agents.alive_mask.sum().item())
    agents.set_brains(alive_genomes)
    
    # ===== 5. 统计记录 =====
    stats = {
        'steps': [],
        'population': [],
        'avg_nodes': [],
        'complex_structures': [],
        'avg_velocity': [],
        'stationary_tax_paid': [],  # 被征税次数
    }
    
    # ===== 6. 课程学习 =====
    CURRICULUM_PHASES = [
        (0, 2000, 1.0, "🏫 基础: 100%可见"),
        (2000, 4000, 0.6, "📚 进阶: 60%可见"),
        (4000, 6000, 0.4, "🎓 终极: 40%可见"),
    ]
    current_phase = 0
    
    print("\n" + "-" * 70)
    print("🚀 开始模拟...")
    print("-" * 70)
    
    start_time = time.time()
    total_tax_paid = 0
    
    for step in range(steps):
        # 课程学习: 调整可见性
        if current_phase < len(CURRICULUM_PHASES):
            phase_start, phase_end, target_ratio, phase_name = CURRICULUM_PHASES[current_phase]
            if step >= phase_end and current_phase + 1 < len(CURRICULUM_PHASES):
                current_phase += 1
                _, _, new_ratio, _ = CURRICULUM_PHASES[current_phase]
                fe = env.flickering_energy_field
                total = fe.flicker_period + fe.invisible_moves
                fe.flicker_period = max(5, int(total * new_ratio))
                fe.invisible_moves = total - fe.flicker_period
                print(f"\n  📢 [课程升级] Step {step}: {phase_name}")
        
        # 大脑前向 + 探索噪声
        def brain_fn(batch):
            sensors = agents.get_sensors(env)
            outputs = agents.forward_brains(sensors)
            noise = torch.randn_like(outputs) * config.EXPLORATION_NOISE
            return outputs + noise
        
        env.step()
        
        # 应用定居者税
        stationary_tax.apply(agents)
        
        step_stats = agents.step(env=env, dt=0.1, brain_fn=brain_fn)
        
        # 记录税收情况
        if stationary_tax.stationary_counter is not None:
            taxed_count = (stationary_tax.stationary_counter >= stationary_tax.tax_steps).sum().item()
            total_tax_paid += taxed_count
        
        # 定期记录
        if (step + 1) % 200 == 0:
            n_alive = agents.alive_mask.sum().item()
            stats['steps'].append(step + 1)
            stats['population'].append(n_alive)
            stats['stationary_tax_paid'].append(total_tax_paid)
            
            if n_alive > 0:
                alive_indices = agents.alive_mask.nonzero(as_tuple=True)[0]
                avg_nodes = agents.state.node_counts[alive_indices].float().mean().item()
                stats['avg_nodes'].append(avg_nodes)
                complex_count = (agents.state.node_counts[alive_indices] > 4).sum().item()
                stats['complex_structures'].append(complex_count)
                
                alive_velocities = agents.state.linear_velocity[alive_indices]
                avg_vel = alive_velocities.norm(dim=-1).mean().item()
                stats['avg_velocity'].append(avg_vel)
            
            fe = env.flickering_energy_field
            is_visible = (fe.step_count % (fe.flicker_period + fe.invisible_moves)) < fe.flicker_period
            
            elapsed = time.time() - start_time
            print(f"  Step {step+1:5d} | 存活: {n_alive:3d} | "
                  f"节点: {avg_nodes:.1f} | 复杂: {complex_count:2d} | "
                  f"速度: {avg_vel:.3f} | {'可见' if is_visible else '隐身'} | "
                  f"税: {total_tax_paid} | {elapsed:.1f}s")
        
        if agents.alive_mask.sum().item() == 0:
            print(f"\n⚠️ 种群在 Step {step+1} 灭绝!")
            break
    
    elapsed = time.time() - start_time
    
    # ===== 7. 结果报告 =====
    print("\n" + "=" * 70)
    print("📊 实验完成")
    print("=" * 70)
    print(f"  运行步数:   {step + 1}")
    print(f"  最终种群:   {agents.alive_mask.sum().item()}")
    print(f"  平均节点:   {np.mean(stats['avg_nodes'][-5:]):.2f}" if stats['avg_nodes'] else "  N/A")
    print(f"  复杂结构:   {stats['complex_structures'][-1] if stats['complex_structures'] else 0}")
    print(f"  平均速度:   {np.mean(stats['avg_velocity'][-5:]):.3f}" if stats['avg_velocity'] else "N/A")
    print(f"  定居者税:   {total_tax_paid} 次")
    print(f"  模拟耗时:   {elapsed:.1f}秒")
    print(f"  步速:       {(step+1)/elapsed:.1f}步/秒")
    
    # 保存结果
    results = {
        'version': 'v16.10',
        'config': {
            'stationary_tax_threshold': getattr(stationary_tax, 'threshold_dist', 0.05),
            'stationary_tax_steps': stationary_tax.tax_steps,
            'stationary_tax_rate': stationary_tax.tax_rate,
            'visible_reward': config.VISIBLE_REWARD_MULTIPLIER,
            'invisible_reward': config.INVISIBLE_REWARD_MULTIPLIER,
            'base_metabolism': config.BASE_METABOLISM,
            'init_energy': 80.0,
        },
        'stats': stats,
        'final': {
            'step': step + 1,
            'population': agents.alive_mask.sum().item(),
            'avg_nodes': stats['avg_nodes'][-1] if stats['avg_nodes'] else 0,
            'stationary_tax_total': total_tax_paid,
        }
    }
    
    output_path = "experiments/v16_deceptive_landscape/v16_9_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  结果已保存: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    run_v16_11_experiment(args.steps, args.pop, args.device)