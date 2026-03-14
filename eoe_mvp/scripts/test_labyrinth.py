"""
3000步 - 迷宫与绿洲 (空间记忆突破)
==================================
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
    import numpy as np
    field = torch.zeros(height, width, device=device)
    
    # 随机墙壁
    n_walls = int(width * height * density / 30)
    for _ in range(n_walls):
        start_x = torch.randint(5, width-5, (1,)).item()
        start_y = torch.randint(5, height-5, (1,)).item()
        direction = torch.randint(0, 2, (1,)).item()
        length = torch.randint(8, 20, (1,)).item()
        
        if direction == 0:  # 水平
            end_x = min(start_x + length, width - 3)
            field[start_y, start_x:end_x] = 50.0  # 高阻抗
        else:  # 垂直
            end_y = min(start_y + length, height - 3)
            field[start_y:end_y, start_x] = 50.0
    
    return field


def run():
    device = 'cuda:0'
    
    manifest = PhysicsManifest.from_yaml('labyrinth')
    print(f"🔧 迷宫配置:")
    print(f"  迷宫密度: {manifest.maze_density}")
    print(f"  迷宫阻抗: {manifest.maze_impedance}")
    print(f"  绿洲数量: {manifest.oasis_count}")
    print(f"  绿洲强度: {manifest.oasis_strength}")
    print(f"{'='*60}")
    print(f"🧪 3000步 - 迷宫与绿洲")
    print(f"{'='*60}")
    
    n_steps = 3000
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    
    # 生成迷宫
    if hasattr(env, 'impedance_field'):
        maze_field = generate_maze(100, 100, manifest.maze_density, device)
        env.impedance_field.field[0, 0] = maze_field
        print(f"  ✅ 迷宫已生成 (阻抗={manifest.maze_impedance})")
    
    # 配置能量绿洲
    if manifest.oases_enabled and hasattr(env, 'energy_field'):
        env.energy_field.n_sources = manifest.oasis_count
        env.energy_field.source_capacity = manifest.oasis_strength
        # 随机绿洲位置 (避开边缘)
        for i in range(manifest.oasis_count):
            env.energy_field.sources[i, 0] = torch.randint(10, 90, (1,), device=device).float()
            env.energy_field.sources[i, 1] = torch.randint(10, 90, (1,), device=device).float()
            env.energy_field.sources[i, 2] = manifest.oasis_strength / 5  # 每次脉冲强度
            env.energy_field.sources[i, 3] = 1.0  # active
            env.energy_field.sources[i, 4] = manifest.oasis_strength  # capacity
            env.energy_field.sources[i, 5] = manifest.oasis_strength
        print(f"  ✅ 能量绿洲已配置 ({manifest.oasis_count}个)")
    
    agents = BatchedAgents(
        initial_population=init_pop, 
        max_agents=max_agents, 
        device=device, 
        init_energy=80.0,
        env=env
    )
    
    # 初始分布: 偏向6节点
    agents.state.node_counts[:init_pop] = torch.randint(5, 8, (init_pop,), device=device)
    
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
            
            attack_mask = torch.rand(batch.n, device=device) < 0.20
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
    ax.set_title('Population (Maze Navigation)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history['step'], history['max_energy'], 'r-', label='Max', linewidth=1)
    ax.plot(history['step'], history['mean_energy'], 'orange', label='Mean', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('Energy (Oasis Hunt)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history['step'], history['mean_nodes'], 'g-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Nodes')
    ax.set_title('Brain Complexity (Spatial Memory)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=7, color='purple', linestyle='--', alpha=0.5, label='7 nodes')
    ax.axhline(y=8, color='red', linestyle='--', alpha=0.5, label='8 nodes')
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
    ax.set_title('Node Distribution')
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
    plt.savefig('evolution_labyrinth.png', dpi=150)
    print(f"✅ 图表已保存: evolution_labyrinth.png")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("🏆 最终结果 - 迷宫与绿洲")
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