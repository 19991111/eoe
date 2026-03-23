#!/usr/bin/env python3
"""
v17 小演化实验 - 保存最佳brain
==============================
运行短实验，保存演化过程中最好的brain供benchmark测试
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import time
import json
import numpy as np
from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU
from configs.presets.stable import StableConfig


def run_experiment(steps=1000):
    """运行短实验并保存最佳brain"""
    
    print("=" * 60)
    print("v17 Short Evolution - Save Best Brain")
    print("=" * 60)
    
    # 初始化配置 - 用稳定版
    config = StableConfig()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.MAX_AGENTS = 300
    
    print(f"\n=== 配置 ===")
    print(f"DEVICE: {config.DEVICE}")
    print(f"MAX_AGENTS: {config.MAX_AGENTS}")
    
    # 初始化环境
    print("\n=== 初始化环境 ===")
    env = EnvironmentGPU(
        width=100.0,
        height=100.0,
        resolution=1.0,
        device=config.DEVICE,
    )
    
    # 预热
    for _ in range(10):
        env.step()
    
    # 调整能量场
    env.energy_field.source_capacity = 5000.0
    env.energy_field.decay_rate = 1.0
    
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
    
    print(f"初始种群: {agents.alive_mask.sum().item()}")
    
    # 记录最佳brain
    best_fitness = -float('inf')
    best_genome = None
    best_step = 0
    
    # 主循环
    print(f"\n=== 开始训练 ({steps}步) ===")
    start_time = time.time()
    
    for step in range(steps):
        agents.step(env)
        
        # 定期记录
        if step % 100 == 0:
            pop = agents.alive_mask.sum().item()
            if pop > 0:
                alive_indices = torch.where(agents.alive_mask)[0]
                node_counts = agents.state.node_counts[alive_indices].float()
                nodes = node_counts.mean().item()
            else:
                nodes = 0
            
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | 种群: {pop:3d} | 节点: {nodes:.2f} | {elapsed:.1f}s")
            
            # 检查最佳个体 - 只用能量作为适应度！
            # 复杂的脑结构应该是为了"更好地获取能量"而被迫演化出来的
            if len(agents.genomes) > 0:
                for idx, genome in agents.genomes.items():
                    if genome is not None and idx < len(agents.state.energies):
                        energy = agents.state.energies[idx].item()
                        # fitness = energy（只奖励能量，让结构自然涌现）
                        if energy > best_fitness:
                            best_fitness = energy
                            best_genome = genome
                            best_step = step
        
        # 检查种群灭绝
        if agents.alive_mask.sum().item() == 0:
            print(f"\n⚠️ 种群灭绝!")
            break
    
    total_time = time.time() - start_time
    print(f"\n=== 实验完成 ===")
    print(f"总步数: {step + 1}")
    print(f"总时间: {total_time:.1f}s")
    print(f"最终种群: {agents.alive_mask.sum().item()}")
    print(f"最佳个体: Step {best_step}, Fitness={best_fitness:.2f}")
    
    # 保存最佳brain
    if best_genome:
        # 转换为可序列化格式
        brain_data = {
            'step': best_step,
            'fitness': best_fitness,
            'nodes': [],
            'edges': []
        }
        
        for node_id, node in best_genome.nodes.items():
            brain_data['nodes'].append({
                'id': node_id,
                'type': node.node_type.name if hasattr(node.node_type, 'name') else str(node.node_type),
            })
        
        for edge in best_genome.edges:
            brain_data['edges'].append({
                'source_id': edge.get('source_id'),
                'target_id': edge.get('target_id'),
                'weight': edge.get('weight', 0.0),
                'enabled': edge.get('enabled', True),
                'learning_rate': edge.get('learning_rate', 0.0),
            })
        
        # 保存
        save_path = '/home/node/.openclaw/workspace/eoe_mvp/benchmarks/results/trained_brain.json'
        with open(save_path, 'w') as f:
            json.dump(brain_data, f, indent=2)
        
        print(f"\n✅ 最佳brain已保存: {save_path}")
        return brain_data
    
    return None


if __name__ == "__main__":
    run_experiment(5000)