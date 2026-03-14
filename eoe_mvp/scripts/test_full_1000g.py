"""
1000代 - 完整版 (捕食 + 年龄惩罚)
=================================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU


def run():
    device = 'cuda:0'
    print(f"设备: {device}")
    print(f"{'='*60}")
    print(f"🧪 1000代 - 黑暗森林 + 代谢衰老")
    print(f"{'='*60}")
    
    n_gen = 1000
    steps = 30
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=init_pop, max_agents=max_agents, device=device, init_energy=80.0)
    
    agents.state.node_counts[:init_pop] = torch.randint(3, 6, (init_pop,), device=device)
    
    print(f"\n{'Gen':>5} | {'存活':>5} | {'最高能':>8} | {'平均能':>8} | {'平均年龄':>8}")
    print("-" * 60)
    
    for gen in range(n_gen):
        for step in range(steps):
            batch = agents.get_active_batch()
            if batch.n > 0:
                outputs = torch.randn(batch.n, 5, device=device)
                outputs[:, 0] = 1.0
                # 10% 概率高攻击
                attack_mask = torch.rand(batch.n, device=device) < 0.1
                outputs[attack_mask, 3] = torch.rand(attack_mask.sum(), device=device) * 2.0
                
                agents._apply_physics(batch, outputs, 0.1)
                agents._apply_metabolism(batch, 0.1)
                agents._apply_environment_interaction(batch, env)
                agents._apply_predation(batch, outputs)
                agents._process_deaths(batch, env)
                agents._process_reproduction(batch)
                agents._apply_boundaries(batch)
            
            env.step()
        
        s = agents.get_population_stats()
        
        if gen % 100 == 0:
            batch = agents.get_active_batch()
            avg_age = agents.state.ages[batch.indices].mean().item() if batch.n > 0 else 0
            print(f"{gen:>5} | {s['n_alive']:>5} | {s['max_energy']:>8.2f} | {s['mean_energy']:>8.2f} | {avg_age:>8.1f}")
            import sys
            sys.stdout.flush()
        
        # 补充
        if s['n_alive'] < 30:
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                agents.alive_mask[dead[:30-s['n_alive']]] = True
                agents.state.energies[dead[:30-s['n_alive']]] = 80.0
                agents.state.ages[dead[:30-s['n_alive']]] = 0.0  # 新生儿
                agents._indices_dirty = True
    
    # 最终分析
    print(f"\n{'='*60}")
    print("🏆 Top 10 Agent")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(10).indices]
        
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            age = agents.state.ages[idx].item()
            print(f"  #{i+1}: 索引={idx:>3}, 能量={e:>7.2f}, 节点={n:>2}, 年龄={age:>6.1f}")
    
    # 年龄分布
    print(f"\n【年龄分布】")
    if batch.n > 0:
        ages = agents.state.ages[batch.indices]
        print(f"  平均: {ages.mean():.1f}")
        print(f"  最大: {ages.max():.1f}")
        print(f"  最小: {ages.min():.1f}")
    
    # 节点 vs 能量
    print(f"\n【节点数 vs 能量】")
    if batch.n > 0:
        for n in range(3, 10):
            mask = agents.state.node_counts[batch.indices] == n
            if mask.sum() > 0:
                energies_n = batch.energies[mask]
                print(f"  节点{n}: 数量={mask.sum()}, 平均能量={energies_n.mean():.2f}")


if __name__ == "__main__":
    run()