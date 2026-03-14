"""
10000步 - 黑暗森林军备竞赛 (6节点起点)
=====================================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.manifest import PhysicsManifest
import matplotlib.pyplot as plt


def run():
    device = 'cuda:0'
    
    manifest = PhysicsManifest.from_yaml('dark_forest')
    print(f"🔧 黑暗森林配置:")
    print(f"  捕食变异: {manifest.predation_mutation}")
    print(f"{'='*60}")
    print(f"🧪 10000步 - 军备竞赛冲刺 (6节点起点)")
    print(f"{'='*60}")
    
    n_steps = 10000
    init_pop = 60  # 稍多初始人口
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    
    agents = BatchedAgents(
        initial_population=init_pop, 
        max_agents=max_agents, 
        device=device, 
        init_energy=80.0,
        env=env
    )
    
    # 初始分布: 偏向6节点 (50%概率6节点, 30%5节点, 20%4节点)
    initial_nodes = torch.where(
        torch.rand(init_pop, device=device) < 0.5,
        torch.tensor(6, device=device),
        torch.where(
            torch.rand(init_pop, device=device) < 0.6,
            torch.tensor(5, device=device),
            torch.tensor(4, device=device)
        )
    )
    agents.state.node_counts[:init_pop] = initial_nodes
    print(f"  初始节点分布: 6节点={(initial_nodes==6).sum()}, 5节点={(initial_nodes==5).sum()}, 4节点={(initial_nodes==4).sum()}")
    
    # 时间序列
    history = {
        'step': [],
        'population': [],
        'mean_energy': [],
        'max_energy': [],
        'mean_nodes': [],
        'mean_age': [],
        'node_dist': {n: [] for n in range(1, 20)}
    }
    
    print(f"\n进度: ", end="", flush=True)
    
    for step in range(n_steps):
        batch = agents.get_active_batch()
        if batch.n > 0:
            outputs = torch.randn(batch.n, 5, device=device)
            outputs[:, 0] = 1.0
            
            attack_mask = torch.rand(batch.n, device=device) < manifest.predation_mutation
            outputs[attack_mask, 3] = torch.rand(attack_mask.sum(), device=device) * 3.0
            
            agents._apply_physics(batch, outputs, 0.1)
            agents._apply_metabolism(batch, 0.1)
            agents._apply_environment_interaction(batch, env)
            
            if manifest.red_queen_enabled:
                agents._apply_predation(batch, outputs)
            
            agents._process_deaths(batch, env)
            agents._process_reproduction(batch)
            agents._apply_boundaries(batch)
        
        env.step()
        
        if step % 200 == 0:
            s = agents.get_population_stats()
            batch = agents.get_active_batch()
            
            history['step'].append(step)
            history['population'].append(s['n_alive'])
            history['mean_energy'].append(s['mean_energy'])
            history['max_energy'].append(s['max_energy'])
            
            if batch.n > 0:
                history['mean_nodes'].append(agents.state.node_counts[batch.indices].float().mean().item())
                history['mean_age'].append(agents.state.ages[batch.indices].mean().item())
                
                node_counts = agents.state.node_counts[batch.indices]
                for n in range(1, 20):
                    count = (node_counts == n).sum().item()
                    history['node_dist'][n].append(count)
            else:
                history['mean_nodes'].append(0)
                history['mean_age'].append(0)
            
            print(f"{step//200}", end=" ", flush=True)
        
        # 补充
        s = agents.get_population_stats()
        if s['n_alive'] < 30:
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                agents.alive_mask[dead[:30-s['n_alive']]] = True
                agents.state.energies[dead[:30-s['n_alive']]] = 80.0
                agents.state.ages[dead[:30-s['n_alive']]] = 0.0
                agents._indices_dirty = True
    
    print(f"\n完成!")
    
    # 绘图
    print(f"\n{'='*60}")
    print("📈 绘制时间序列图...")
    print("="*60)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    ax.plot(history['step'], history['population'], 'b-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('Population (Boom & Bust)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history['step'], history['max_energy'], 'r-', label='Max', linewidth=1)
    ax.plot(history['step'], history['mean_energy'], 'orange', label='Mean', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history['step'], history['mean_nodes'], 'g-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Nodes')
    ax.set_title('Brain Complexity (Evolution to Higher Intelligence)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=6, color='purple', linestyle='--', alpha=0.5, label='6 nodes')
    ax.axhline(y=7, color='red', linestyle='--', alpha=0.5, label='7 nodes')
    ax.legend()
    
    ax = axes[1, 1]
    ax.plot(history['step'], history['mean_age'], 'purple', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Age')
    ax.set_title('Metabolic Senescence')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    for n in range(4, 12):
        if any(history['node_dist'][n]):
            ax.plot(history['step'], history['node_dist'][n], label=f'n{n}', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Node Distribution (Arms Race)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    node_matrix = []
    for n in range(4, 12):
        node_matrix.append(history['node_dist'][n])
    node_matrix = torch.tensor(node_matrix).float()
    im = ax.imshow(node_matrix.numpy(), aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Step (x200)')
    ax.set_ylabel('Node Count')
    ax.set_title('Heatmap')
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(4, 12))
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('evolution_dark_forest_10k.png', dpi=150)
    print(f"✅ 图表已保存: evolution_dark_forest_10k.png")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("🏆 最终结果 - 10000步军备竞赛")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(10).indices]
        
        print("\nTop 10 Agent:")
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            age = agents.state.ages[idx].item()
            print(f"  #{i+1}: Energy={e:.2f}, Nodes={n}, Age={age:.0f}")
        
        print(f"\n【Final Stats】")
        print(f"  Population: {batch.n}")
        print(f"  Mean Energy: {batch.energies.mean():.2f}")
        print(f"  Mean Nodes: {agents.state.node_counts[batch.indices].float().mean():.2f}")
        print(f"  Mean Age: {agents.state.ages[batch.indices].mean():.1f}")
        
        print(f"\n【Node Distribution】")
        for n in range(3, 15):
            count = (agents.state.node_counts[batch.indices] == n).sum().item()
            if count > 0:
                print(f"  Nodes {n}: {count} ({100*count/batch.n:.1f}%)")


if __name__ == "__main__":
    run()