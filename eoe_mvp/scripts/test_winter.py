"""
2000代 - 凛冬将至 (季节系统)
===========================
能量随正弦波变化，冬天食物稀缺
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
    
    # 加载配置
    manifest = PhysicsManifest.from_yaml('winter')
    print(f"🔧 季节配置: enabled={manifest.seasons_enabled}, length={manifest.season_length}")
    print(f"  冬季倍率: {manifest.winter_multiplier}, 夏季倍率: {manifest.summer_multiplier}")
    print(f"🔧 红皇后: enabled={manifest.red_queen_enabled}")
    print(f"{'='*60}")
    print(f"🧪 2000步 - 凛冬将至")
    print(f"{'='*60}")
    
    n_steps = 2000
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    
    # 启用季节系统
    env.set_seasons(
        enabled=manifest.seasons_enabled,
        length=manifest.season_length,
        winter=manifest.winter_multiplier,
        summer=manifest.summer_multiplier
    )
    print(f"  ✅ 季节系统已启用")
    
    agents = BatchedAgents(
        initial_population=init_pop, 
        max_agents=max_agents, 
        device=device, 
        init_energy=80.0,
        env=env
    )
    
    agents.state.node_counts[:init_pop] = torch.randint(3, 6, (init_pop,), device=device)
    
    # 时间序列
    history = {
        'step': [],
        'population': [],
        'mean_energy': [],
        'max_energy': [],
        'mean_nodes': [],
        'mean_age': [],
        'seasonal_multiplier': [],
        'node_dist': {n: [] for n in range(1, 15)}
    }
    
    print(f"\n进度: ", end="", flush=True)
    
    for step in range(n_steps):
        batch = agents.get_active_batch()
        if batch.n > 0:
            outputs = torch.randn(batch.n, 5, device=device)
            outputs[:, 0] = 1.0
            
            # 红皇后捕食
            attack_mask = torch.rand(batch.n, device=device) < 0.15
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
        
        if step % 100 == 0:
            s = agents.get_population_stats()
            batch = agents.get_active_batch()
            
            # 记录季节倍率
            season_mult = env.get_seasonal_multiplier()
            
            history['step'].append(step)
            history['population'].append(s['n_alive'])
            history['mean_energy'].append(s['mean_energy'])
            history['max_energy'].append(s['max_energy'])
            history['seasonal_multiplier'].append(season_mult)
            
            if batch.n > 0:
                history['mean_nodes'].append(agents.state.node_counts[batch.indices].float().mean().item())
                history['mean_age'].append(agents.state.ages[batch.indices].mean().item())
                
                node_counts = agents.state.node_counts[batch.indices]
                for n in range(1, 15):
                    count = (node_counts == n).sum().item()
                    history['node_dist'][n].append(count)
            else:
                history['mean_nodes'].append(0)
                history['mean_age'].append(0)
            
            season_label = "冬" if season_mult < 0.5 else "夏" if season_mult > 1.0 else "春/秋"
            print(f"{step//100}", end=f"({season_label}:{season_mult:.1f}) ", flush=True)
        
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
    ax.set_title('Population (Seasonal Cycles)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history['step'], history['max_energy'], 'r-', label='Max', linewidth=1)
    ax.plot(history['step'], history['mean_energy'], 'orange', label='Mean', linewidth=1)
    ax_twin = ax.twinx()
    ax_twin.plot(history['step'], history['seasonal_multiplier'], 'g--', label='Season', alpha=0.5, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs Season')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history['step'], history['mean_nodes'], 'g-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Nodes')
    ax.set_title('Brain Complexity')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(history['step'], history['mean_age'], 'purple', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Age')
    ax.set_title('Metabolic Senescence')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    for n in range(1, 10):
        if any(history['node_dist'][n]):
            ax.plot(history['step'], history['node_dist'][n], label=f'n{n}', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Node Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    node_matrix = []
    for n in range(1, 10):
        node_matrix.append(history['node_dist'][n])
    node_matrix = torch.tensor(node_matrix).float()
    im = ax.imshow(node_matrix.numpy(), aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Step (x100)')
    ax.set_ylabel('Node Count')
    ax.set_title('Heatmap')
    ax.set_yticks(range(9))
    ax.set_yticklabels(range(1, 10))
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('evolution_winter.png', dpi=150)
    print(f"✅ 图表已保存: evolution_winter.png")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("🏆 最终结果 - 凛冬将至")
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
        for n in range(3, 12):
            count = (agents.state.node_counts[batch.indices] == n).sum().item()
            if count > 0:
                print(f"  Nodes {n}: {count} ({100*count/batch.n:.1f}%)")


if __name__ == "__main__":
    run()