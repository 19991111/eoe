"""
10000步 - 演化棘轮 + 迷宫 + 凛冬将至 (终极环境)
================================================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.manifest import PhysicsManifest
import matplotlib.pyplot as plt


def generate_maze(width, height, density, device):
    """生成迷宫墙壁"""
    field = torch.zeros(height, width, device=device)
    n_walls = int(width * height * density / 30)
    for _ in range(n_walls):
        start_x = torch.randint(5, width-5, (1,)).item()
        start_y = torch.randint(5, height-5, (1,)).item()
        direction = torch.randint(0, 2, (1,)).item()
        length = torch.randint(8, 20, (1,)).item()
        
        if direction == 0:
            end_x = min(start_x + length, width - 3)
            field[start_y, start_x:end_x] = 50.0
        else:
            end_y = min(start_y + length, height - 3)
            field[start_y:end_y, start_x] = 50.0
    
    return field


def run():
    device = 'cuda:0'
    
    manifest = PhysicsManifest.from_yaml('labyrinth')
    print(f"🔧 配置:")
    print(f"  迷宫: density={manifest.maze_density}")
    print(f"  绿洲: count={manifest.oasis_count}")
    print(f"  超级节点: 启用")
    print(f"  冬季: HARS winter_multiplier=0.1 (极寒!)")
    print(f"{'='*60}")
    print(f"🧪 10000步 - 凛冬将至 (终极环境)")
    print(f"{'='*60}")
    
    n_steps = 10000
    init_pop = 60
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    
    # 迷宫
    if hasattr(env, 'impedance_field'):
        maze_field = generate_maze(100, 100, manifest.maze_density, device)
        env.impedance_field.field[0, 0] = maze_field
        print(f"  ✅ 迷宫已生成")
    
    # 绿洲 - 只有绿洲有能量!
    if manifest.oases_enabled and hasattr(env, 'energy_field'):
        env.energy_field.n_sources = manifest.oasis_count
        env.energy_field.source_capacity = manifest.oasis_strength
        for i in range(manifest.oasis_count):
            env.energy_field.sources[i, 0] = torch.randint(10, 90, (1,), device=device).float()
            env.energy_field.sources[i, 1] = torch.randint(10, 90, (1,), device=device).float()
            env.energy_field.sources[i, 2] = manifest.oasis_strength / 5
            env.energy_field.sources[i, 3] = 1.0
            env.energy_field.sources[i, 4] = manifest.oasis_strength
            env.energy_field.sources[i, 5] = manifest.oasis_strength
        print(f"  ✅ 绿洲已配置 (唯能量来源)")
    
    # 极寒冬季 - 能量生成降至10%!
    if hasattr(env, 'energy_field'):
        env.energy_field.seasonal_multiplier = 0.1  # 冬季只有10%!
        env.energy_field.season_length = 400  # 冬季400步
        print(f"  ✅ 凛冬将至 (冬×10能量!)")
    
    agents = BatchedAgents(
        initial_population=init_pop, 
        max_agents=max_agents, 
        device=device, 
        init_energy=100.0,  # 更高初始能量
        env=env
    )
    
    # 初始: 7-9节点, 1-3超级节点
    agents.state.node_counts[:init_pop] = torch.randint(7, 10, (init_pop,), device=device)
    agents.state.supernodes[:init_pop] = torch.randint(1, 4, (init_pop,), device=device)
    
    history = {
        'step': [],
        'population': [],
        'mean_energy': [],
        'max_energy': [],
        'mean_nodes': [],
        'mean_effective_nodes': [],
        'mean_supernodes': [],
        'mean_age': [],
        'season': [],  # 0=夏, 1=冬
        'node_dist': {n: [] for n in range(1, 30)}
    }
    
    print(f"\n进度: ", end="", flush=True)
    
    for step in range(n_steps):
        batch = agents.get_active_batch()
        if batch.n > 0:
            outputs = torch.randn(batch.n, 5, device=device)
            outputs[:, 0] = 1.0
            
            attack_mask = torch.rand(batch.n, device=device) < 0.25  # 更高捕食压力
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
        
        # 记录季节
        if hasattr(env, 'energy_field'):
            season = (step % (2 * env.energy_field.season_length)) // env.energy_field.season_length
            history['season'].append(season)
        else:
            history['season'].append(0)
        
        if step % 200 == 0:
            s = agents.get_population_stats()
            batch = agents.get_active_batch()
            
            history['step'].append(step)
            history['population'].append(s['n_alive'])
            history['mean_energy'].append(s['mean_energy'])
            history['max_energy'].append(s['max_energy'])
            
            if batch.n > 0:
                nc = agents.state.node_counts[batch.indices].float()
                sn = agents.state.supernodes[batch.indices].float()
                effective = nc - sn * 0.5
                
                history['mean_nodes'].append(nc.mean().item())
                history['mean_effective_nodes'].append(effective.mean().item())
                history['mean_supernodes'].append(sn.mean().item())
                history['mean_age'].append(agents.state.ages[batch.indices].mean().item())
                
                for n in range(1, 30):
                    count = (agents.state.node_counts[batch.indices] == n).sum().item()
                    history['node_dist'][n].append(count)
            else:
                history['mean_nodes'].append(0)
                history['mean_effective_nodes'].append(0)
                history['mean_supernodes'].append(0)
                history['mean_age'].append(0)
            
            print(f"{step//200}", end=" ", flush=True)
        
        s = agents.get_population_stats()
        if s['n_alive'] < 30:
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                agents.alive_mask[dead[:30-s['n_alive']]] = True
                agents.state.energies[dead[:30-s['n_alive']]] = 100.0
                agents.state.ages[dead[:30-s['n_alive']]] = 0.0
                agents._indices_dirty = True
    
    print(f"\n完成!")
    
    print(f"\n{'='*60}")
    print("📈 绘制时间序列图...")
    print("="*60)
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    ax = axes[0, 0]
    ax.plot(history['step'], history['population'], 'b-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('Population (Winter Crashes)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history['step'], history['max_energy'], 'r-', label='Max', linewidth=1)
    ax.plot(history['step'], history['mean_energy'], 'orange', label='Mean', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy (Winter Crashes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history['step'], history['mean_nodes'], 'g-', label='Total Nodes', linewidth=1)
    ax.plot(history['step'], history['mean_effective_nodes'], 'purple', label='Effective Nodes', linewidth=1)
    ax.plot(history['step'], history['mean_supernodes'], 'orange', label='Supernodes', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Brain Complexity (Harsh Winter Selection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=9, color='red', linestyle='--', alpha=0.5, label='9 nodes')
    ax.axhline(y=10, color='purple', linestyle='--', alpha=0.5, label='10 nodes')
    
    ax = axes[1, 1]
    ax.plot(history['step'], history['mean_age'], 'purple', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Age')
    ax.set_title('Metabolic Senescence')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    for n in range(7, 15):
        if any(history['node_dist'][n]):
            ax.plot(history['step'], history['node_dist'][n], label=f'n{n}', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Node Distribution (Winter Selection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    node_matrix = []
    for n in range(7, 15):
        node_matrix.append(history['node_dist'][n])
    node_matrix = torch.tensor(node_matrix).float()
    im = ax.imshow(node_matrix.numpy(), aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Step (x200)')
    ax.set_ylabel('Node Count')
    ax.set_title('Heatmap')
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(7, 15))
    plt.colorbar(im, ax=ax)
    
    # 能量分布
    ax = axes[3, 0]
    batch = agents.get_active_batch()
    if batch.n > 0:
        energies = agents.state.energies[batch.indices].cpu().numpy()
        ax.hist(energies, bins=20, color='orange', alpha=0.7)
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title('Final Energy Distribution')
    
    ax = axes[3, 1]
    # 节点vs超级节点散点图
    if batch.n > 0:
        nc = agents.state.node_counts[batch.indices].cpu().numpy()
        sn = agents.state.supernodes[batch.indices].cpu().numpy()
        ax.scatter(nc, sn, alpha=0.6)
        ax.set_xlabel('Node Count')
        ax.set_ylabel('Supernode Count')
        ax.set_title('Nodes vs Supernodes')
    
    plt.tight_layout()
    plt.savefig('evolution_apocalypse.png', dpi=150)
    print(f"✅ 图表已保存: evolution_apocalypse.png")
    
    print(f"\n{'='*60}")
    print("🏆 最终结果 - 凛冬将至 (终极环境)")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(10).indices]
        
        print("\nTop 10 Agent:")
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            sn = agents.state.supernodes[idx].item()
            effective = n - sn * 0.5
            age = agents.state.ages[idx].item()
            print(f"  #{i+1}: Energy={e:.2f}, Nodes={n}(有效{effective:.1f}), Supernodes={sn}, Age={age:.0f}")
        
        print(f"\n【Final Stats】")
        print(f"  Population: {batch.n}")
        print(f"  Mean Energy: {batch.energies.mean():.2f}")
        print(f"  Mean Nodes: {agents.state.node_counts[batch.indices].float().mean():.2f}")
        nc = agents.state.node_counts[batch.indices].float()
        sn = agents.state.supernodes[batch.indices].float()
        effective = nc - sn * 0.5
        print(f"  Mean Effective Nodes: {effective.mean():.2f}")
        print(f"  Mean Supernodes: {sn.mean():.2f}")
        print(f"  Mean Age: {agents.state.ages[batch.indices].mean():.1f}")
        
        print(f"\n【Node Distribution】")
        for n in range(5, 20):
            count = (agents.state.node_counts[batch.indices] == n).sum().item()
            if count > 0:
                print(f"  Nodes {n}: {count} ({100*count/batch.n:.1f}%)")
        
        print(f"\n【Supernode Distribution】")
        for sn_cnt in range(0, 8):
            count = (agents.state.supernodes[batch.indices] == sn_cnt).sum().item()
            if count > 0:
                print(f"  Supernodes {sn_cnt}: {count} ({100*count/batch.n:.1f}%)")


if __name__ == "__main__":
    run()