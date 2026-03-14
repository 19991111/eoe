"""
100代测试 - 简化版
==================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU


def run():
    device = 'cuda:0'
    print(f"设备: {device}")
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=50, max_agents=500, device=device, init_energy=80.0)
    
    # 随机节点数
    agents.state.node_counts[:50] = torch.randint(3, 8, (50,), device=device)
    
    print(f"\nGen | 存活 | 最高能 | 平均能")
    print("-" * 40)
    
    for gen in range(100):
        # 每代开始时补充能量（模拟上一代的后代）
        # 或者让系统自己运行
        
        for step in range(30):
            # 随机运动 + 攻击
            batch = agents.get_active_batch()
            if batch.n > 0:
                outputs = torch.randn(batch.n, 5, device=device)
                outputs[:, 0] = 1.0  # 前进
                # 随机攻击 (10% 概率高攻击)
                attack_mask = torch.rand(batch.n, device=device) < 0.1
                outputs[attack_mask, 3] = torch.rand(attack_mask.sum(), device=device) * 2.0  # 强攻击
                agents._apply_physics(batch, outputs, 0.1)
                agents._apply_metabolism(batch, 0.1)
                agents._apply_environment_interaction(batch, env)
                agents._process_deaths(batch, env)
                births = agents._process_reproduction(batch)
                agents._apply_boundaries(batch)
            
            env.step()
        
        s = agents.get_population_stats()
        
        if gen % 10 == 0:
            print(f"{gen:3d} | {s['n_alive']:4d} | {s['max_energy']:7.2f} | {s['mean_energy']:7.2f}")
        
        # 补充死亡
        if s['n_alive'] < 30:
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                agents.alive_mask[dead[:30-s['n_alive']]] = True
                agents.state.energies[dead[:30-s['n_alive']]] = 80.0
                agents._indices_dirty = True
    
    # 最终Top 5
    print("\n" + "="*50)
    print("🏆 Top 5 Agent (100代后)")
    print("="*50)
    
    # 分析年龄
    batch = agents.get_active_batch()
    if batch.n > 0:
        ages = agents.state.ages[batch.indices]
        print(f"\n【年龄分析】")
        print(f"  平均年龄: {ages.mean():.1f}")
        print(f"  最大年龄: {ages.max():.1f}")
        print(f"  最小年龄: {ages.min():.1f}")
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(5).indices]
        
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            print(f"#{i+1}: 索引={idx}, 能量={e:.2f}, 节点数={n}")
    else:
        print("全部灭亡!")


if __name__ == "__main__":
    run()