"""
1000代深度分析测试
==================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU


def run_1000_generations():
    device = 'cuda:0'
    print(f"设备: {device}")
    print(f"{'='*60}")
    print(f"🧪 1000代深度演化分析")
    print(f"{'='*60}")
    
    # 配置
    n_gen = 1000
    steps = 30
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=init_pop, max_agents=max_agents, device=device, init_energy=80.0)
    
    # 随机初始节点数
    agents.state.node_counts[:init_pop] = torch.randint(3, 6, (init_pop,), device=device)
    
    # 统计历史
    history = {
        'population': [],
        'max_energy': [],
        'mean_energy': [],
        'node_distribution': [],
        'births': [],
        'deaths': []
    }
    
    print(f"\n{'Gen':>5} | {'存活':>5} | {'出生':>5} | {'死亡':>5} | {'最高能':>8} | {'平均能':>8}")
    print("-" * 60)
    
    total_births = 0
    total_deaths = 0
    
    for gen in range(n_gen):
        births = 0
        deaths = 0
        
        for step in range(steps):
            batch = agents.get_active_batch()
            if batch.n > 0:
                outputs = torch.randn(batch.n, 5, device=device)
                outputs[:, 0] = 1.0
                agents._apply_physics(batch, outputs, 0.1)
                agents._apply_metabolism(batch, 0.1)
                agents._apply_environment_interaction(batch, env)
                deaths += agents._process_deaths(batch, env)
                births += agents._process_reproduction(batch)
                agents._apply_boundaries(batch)
            
            env.step()
        
        s = agents.get_population_stats()
        
        # 节点分布
        batch = agents.get_active_batch()
        if batch.n > 0:
            node_dist = torch.bincount(
                agents.state.node_counts[batch.indices].clamp(1, 20).long(),
                minlength=21
            ).tolist()
        else:
            node_dist = [0] * 21
        
        history['population'].append(s['n_alive'])
        history['max_energy'].append(s['max_energy'])
        history['mean_energy'].append(s['mean_energy'])
        history['node_distribution'].append(node_dist)
        history['births'].append(births)
        history['deaths'].append(deaths)
        
        total_births += births
        total_deaths += deaths
        
        if gen % 100 == 0:
            print(f"{gen:>5} | {s['n_alive']:>5} | {births:>5} | {deaths:>5} | {s['max_energy']:>8.2f} | {s['mean_energy']:>8.2f}")
        
        # 补充
        if s['n_alive'] < 30:
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                agents.alive_mask[dead[:30-s['n_alive']]] = True
                agents.state.energies[dead[:30-s['n_alive']]] = 80.0
                agents._indices_dirty = True
    
    # 最终统计
    print("\n" + "="*60)
    print("📊 1000代演化统计报告")
    print("="*60)
    
    print(f"\n【基本统计】")
    print(f"  总代数: {n_gen}")
    print(f"  总出生: {total_births}")
    print(f"  总死亡: {total_deaths}")
    print(f"  最终存活: {history['population'][-1]}")
    
    print(f"\n【能量演化】")
    print(f"  初始最高能量: {history['max_energy'][0]:.2f}")
    print(f"  最终最高能量: {history['max_energy'][-1]:.2f}")
    print(f"  能量峰值代: {max(range(n_gen), key=lambda i: history['max_energy'][i])} (能量={max(history['max_energy']):.2f})")
    print(f"  能量谷底代: {min(range(n_gen), key=lambda i: history['max_energy'][i])} (能量={min(history['max_energy']):.2f})")
    
    print(f"\n【种群动态】")
    print(f"  初始人口: {history['population'][0]}")
    print(f"  最终人口: {history['population'][-1]}")
    print(f"  人口峰值: {max(history['population'])} (代 {history['population'].index(max(history['population']))})")
    
    print(f"\n【节点数演化】")
    # 统计节点分布变化
    early_dist = history['node_distribution'][0]
    mid_dist = history['node_distribution'][n_gen//2]
    final_dist = history['node_distribution'][-1]
    
    print(f"  初始代节点分布: ", end="")
    for i, c in enumerate(early_dist):
        if c > 0:
            print(f"节点{i}={c} ", end="")
    print()
    
    print(f"  中间代(500)节点分布: ", end="")
    for i, c in enumerate(mid_dist):
        if c > 0:
            print(f"节点{i}={c} ", end="")
    print()
    
    print(f"  最终代节点分布: ", end="")
    for i, c in enumerate(final_dist):
        if c > 0:
            print(f"节点{i}={c} ", end="")
    print()
    
    # 计算节点多样性指数 (Shannon Entropy)
    def shannon_entropy(dist):
        total = sum(dist)
        if total == 0:
            return 0
        entropy = 0
        for c in dist:
            if c > 0:
                p = c / total
                entropy -= p * torch.log2(torch.tensor(p)).item()
        return entropy
    
    early_ent = shannon_entropy(torch.tensor(early_dist))
    final_ent = shannon_entropy(torch.tensor(final_dist))
    
    print(f"\n  多样性指数 (Shannon Entropy):")
    print(f"    初始: {early_ent:.3f}")
    print(f"    最终: {final_ent:.3f}")
    print(f"    变化: {'+' if final_ent > early_ent else ''}{final_ent - early_ent:.3f}")
    
    # Top 10 Agent 分析
    print(f"\n{'='*60}")
    print("🏆 Top 10 Agent 分析")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(10).indices]
        
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            v = agents.state.linear_velocity[idx].item()
            print(f"  #{i+1:>2}: 索引={idx:>3}, 能量={e:>7.2f}, 节点={n:>2}, 速度={v:>5.2f}")
    
    # 分析节点数与能量的关系
    print(f"\n【节点数 vs 能量分析】")
    batch = agents.get_active_batch()
    if batch.n > 0:
        for n in range(1, 10):
            mask = agents.state.node_counts[batch.indices] == n
            if mask.any():
                energies_n = batch.energies[mask]
                print(f"  节点数={n}: 数量={mask.sum()}, 平均能量={energies_n.mean():.2f}, 最大能量={energies_n.max():.2f}")
    
    return agents, env, history


if __name__ == "__main__":
    run_1000_generations()