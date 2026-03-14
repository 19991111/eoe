#!/usr/bin/env python3
"""
v14.1 诊断实验脚本
==================
用于排查:
1. 鲍德温效应反噬 (学习成本过高)
2. 演化棘轮锁死 (陷入局部最优)  
3. 共生场应力反馈 (链式崩溃)
4. 基因组坍缩

运行: python scripts/run_diagnostics.py [steps]
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.node import NodeType, Node
from core.eoe.genome import OperatorGenome


def create_cambrian_genome():
    """寒武纪初始化基因组"""
    g = OperatorGenome()
    n_nodes = np.random.randint(3, 8)
    
    types = [NodeType.SENSOR]
    for _ in range(n_nodes - 2):
        types.append(np.random.choice([
            NodeType.ADD, NodeType.MULTIPLY,
            NodeType.THRESHOLD, NodeType.DELAY
        ]))
    types.append(NodeType.ACTUATOR)
    
    for j, t in enumerate(types):
        g.add_node(Node(j, t))
    
    for src in range(len(types) - 1):
        if np.random.random() < 0.7:
            g.add_edge(src, src+1, 0.001)
    
    # 确保连通
    if not any(e['source_id'] == 0 for e in g.edges):
        g.add_edge(0, np.random.randint(1, len(types)), 0.001)
    if not any(e['target_id'] == len(types)-1 for e in g.edges):
        g.add_edge(np.random.randint(0, len(types)-1), len(types)-1, 0.001)
    
    g.energy = 100.0
    return g


def run_diagnostic_experiment(
    n_steps: int = 20000,
    initial_population: int = 500,
    device: str = 'cuda:0'
):
    """运行诊断实验"""
    
    print("="*70)
    print("🔬 v14.1 诊断实验 - 排查潜在问题")
    print("="*70)
    
    # 配置
    config = PoolConfig()
    config.HEBBIAN_ENABLED = True
    config.HEBBIAN_REWARD_MODULATION = True
    config.SUPERNODE_ENABLED = True
    config.SUPERNODE_DETECTION_FREQUENCY = 500
    config.AGE_ENABLED = True
    config.PREDATION_ENABLED = True
    config.DIAGNOSTICS_ENABLED = True
    
    # 创建环境
    env = EnvironmentGPU(
        width=100.0, height=100.0, resolution=1.0,
        device=device,
        energy_field_enabled=True,
        seasons_enabled=True,
        season_length=3000,
        winter_multiplier=0.15,
        summer_multiplier=1.8,
        drought_intensity=0.08
    )
    
    # 注入能量
    env.energy_field.field[0, 0, 50, 50] = 200.0
    env.energy_field.field[0, 0, 25, 25] = 100.0
    env.energy_field.field[0, 0, 75, 75] = 100.0
    
    print(f"  🍎 能量源已注入")
    
    # 创建Agent池
    agents = BatchedAgents(
        initial_population=initial_population,
        max_agents=10000,
        env_width=100.0, env_height=100.0,
        device=device,
        init_energy=100.0,
        config=config,
        env=env
    )
    
    # 寒武纪初始化
    for i in range(initial_population):
        g = create_cambrian_genome()
        agents.genomes[i] = g
        agents.state.node_counts[i] = len(g.nodes)
    
    print(f"  🧬 寒武纪初始化完成: {initial_population} 个Agent")
    
    # 检查诊断系统
    if agents.diagnostics:
        print(f"  ✅ 诊断系统已启用")
    else:
        print(f"  ⚠️ 诊断系统未启用")
    
    # 运行参数
    energy_refill_step = 100
    last_refill = 0
    last_report = 0
    report_interval = 500
    
    # 统计
    total_energy_gained = 0.0
    total_hebbian_cost = 0.0
    
    print(f"\n开始运行 {n_steps} 步...")
    
    for step in range(n_steps):
        # 能量补充
        if step - last_refill >= energy_refill_step:
            env.energy_field.field[0, 0, 50, 50] = 200.0
            env.energy_field.field[0, 0, 25, 25] = 100.0
            env.energy_field.field[0, 0, 75, 75] = 100.0
            last_refill = step
        
        # Step
        result = agents.step(env=env, dt=0.1)
        
        # 能量奖励 (触发Hebbian学习)
        if step % 10 == 0:
            batch = agents.get_active_batch()
            if batch.n > 0:
                reward_mask = torch.rand(batch.n, device=device) < 0.1
                reward_indices = batch.indices[reward_mask]
                agents.state.energies[reward_indices] += 15.0
                total_energy_gained += reward_indices.shape[0] * 15.0
        
        # 估算Hebbian学习成本
        if hasattr(agents, '_hebbian_progress') and agents._hebbian_progress is not None:
            n_learning = (agents._hebbian_progress.abs() > 0.01).sum().item()
            total_hebbian_cost += n_learning * 0.01  # 每次学习成本
        
        # 记录诊断
        if agents.diagnostics and step % report_interval == 0:
            batch = agents.get_active_batch()
            
            n_learning = 0
            if hasattr(agents, '_hebbian_progress') and agents._hebbian_progress is not None:
                n_learning = (agents._hebbian_progress.abs() > 0.01).sum().item()
            
            n_supernodes = 0
            if hasattr(agents, 'supernode_registry'):
                stats = agents.supernode_registry.get_stats()
                n_supernodes = stats.get('n_supernodes', 0)
            
            agents.diagnostics.record_step(
                step=step,
                n_alive=result['n_alive'],
                energies=batch.energies if batch.n > 0 else torch.tensor([0.0], device=device),
                node_counts=agents.state.node_counts[agents.alive_mask] if result['n_alive'] > 0 else torch.tensor([0], device=device),
                hebbian_active=n_learning,
                hebbian_cost=n_learning * 0.01,
                energy_gained=reward_indices.shape[0] * 15.0 if step % 10 == 0 else 0,
                n_supernodes=n_supernodes,
                genome_lengths=agents.state.node_counts[agents.alive_mask] if result['n_alive'] > 0 else None
            )
        
        # 报告
        if step - last_report >= report_interval:
            elapsed = (step - last_report) / report_interval * 0.5  # 估算
            
            batch = agents.get_active_batch()
            n_alive = result['n_alive']
            avg_energy = batch.energies.mean().item() if n_alive > 0 else 0
            
            n_learning = 0
            if hasattr(agents, '_hebbian_progress') and agents._hebbian_progress is not None:
                n_learning = (agents._hebbian_progress.abs() > 0.01).sum().item()
            
            n_supernodes = 0
            if hasattr(agents, 'supernode_registry'):
                stats = agents.supernode_registry.get_stats()
                n_supernodes = stats.get('n_supernodes', 0)
            
            avg_nodes = agents.state.node_counts[agents.alive_mask].float().mean().item() if n_alive > 0 else 0
            
            print(f"Step {step:5d} | 存活: {n_alive:4d} | "
                  f"能量: {avg_energy:6.1f} | "
                  f"节点: {avg_nodes:.1f} | "
                  f"Hebbian: {n_learning:4d} | "
                  f"SuperNode: {n_supernodes:2d}")
            
            last_report = step
    
    # 最终统计
    print("\n" + "="*70)
    print("📊 最终统计")
    print("="*70)
    
    # 打印诊断报告
    if agents.diagnostics:
        agents.diagnostics.print_report()
        agents.diagnostics.save_to_file("diagnostics_final.json")
    
    # 手动计算关键指标
    print(f"\n🔍 关键指标:")
    print(f"  总能量获取: {total_energy_gained:.1f}")
    print(f"  总学习成本: {total_hebbian_cost:.1f}")
    print(f"  学习成本比: {total_hebbian_cost/max(total_energy_gained, 0.1):.1%}")
    
    # 检查警告
    learning_ratio = total_hebbian_cost / max(total_energy_gained, 0.1)
    if learning_ratio > 0.3:
        print(f"\n⚠️ 警告: 学习成本过高 ({learning_ratio:.1%})")
        print(f"   建议: 降低 HEBBIAN_BASE_LR 或增加能量获取")
    
    print("\n" + "="*70)
    print("✅ 诊断实验完成")
    print("="*70)


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    run_diagnostic_experiment(n_steps=n_steps, device=device)