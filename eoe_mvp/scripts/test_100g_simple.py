"""
100代快速测试 - 无基因组版
==========================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU


def run_test():
    device = 'cuda:0'
    print(f"设备: {device}")
    
    n_gen = 100
    steps = 30
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=init_pop, max_agents=max_agents, device=device, init_energy=80.0)
    
    # 随机节点数
    agents.state.node_counts[:init_pop] = torch.randint(2, 8, (init_pop,), device=device)
    
    print(f"\nGen | 存活 | 出生 | 死亡 | 最高能 | 平均能")
    print("-" * 55)
    
    for gen in range(n_gen):
        births = 0
        deaths = 0
        
        for _ in range(steps):
            # 随机推力让Agent移动
            batch = agents.get_active_batch()
            if batch.n > 0:
                # 随机推力
                random_outputs = torch.randn(batch.n, 5, device=device)
                # 简单前进
                random_outputs[:, 0] = 1.0  # thrust
                
                # 手动应用物理
                agents._apply_physics(batch, random_outputs, 0.1)
                agents._apply_metabolism(batch, 0.1)
                agents._apply_environment_interaction(batch, env)
                deaths += agents._process_deaths(batch, env)
                births += agents._process_reproduction(batch)
                agents._apply_boundaries(batch)
            
            env.step()
        
        s = agents.get_population_stats()
        
        if gen % 10 == 0:
            print(f"{gen:3d} | {s['n_alive']:4d} | {births:4d} | {deaths:4d} | {s['max_energy']:7.2f} | {s['mean_energy']:7.2f}")
        
        # 补充
        if s['n_alive'] < init_pop:
            needed = init_pop - s['n_alive']
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                respawn = dead[:needed]
                agents.alive_mask[respawn] = True
                agents.state.energies[respawn] = 80.0
                agents._indices_dirty = True
    
    # Top 5
    print("\n" + "="*50)
    print("🏆 Top 5 Agent")
    print("="*50)
    
    batch = agents.get_active_batch()
    energies = batch.energies
    top_idx = batch.indices[energies.topk(5).indices]
    
    for i, idx in enumerate(top_idx.tolist()):
        e = agents.state.energies[idx].item()
        n = agents.state.node_counts[idx].item()
        v = agents.state.linear_velocity[idx].item()
        print(f"#{i+1}: 索引={idx}, 能量={e:.2f}, 节点={n}, 速度={v:.3f}")


if __name__ == "__main__":
    run_test()