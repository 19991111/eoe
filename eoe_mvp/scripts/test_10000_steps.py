"""
10000步 - 完整时间序列记录 + 绘图
==================================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents
from core.eoe.environment_gpu import EnvironmentGPU
import matplotlib.pyplot as plt


def run():
    device = 'cuda:0'
    print(f"设备: {device}")
    print(f"{'='*60}")
    print(f"🧪 10000步 - 完整时间序列")
    print(f"{'='*60}")
    
    n_steps = 10000
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=init_pop, max_agents=max_agents, device=device, init_energy=80.0)
    
    agents.state.node_counts[:init_pop] = torch.randint(3, 6, (init_pop,), device=device)
    
    # 时间序列历史
    history = {
        'step': [],
        'population': [],
        'mean_energy': [],
        'max_energy': [],
        'mean_nodes': [],
        'mean_age': [],
        'node_dist': {n: [] for n in range(1, 15)}
    }
    
    print(f"\n进度: ", end="", flush=True)
    
    step_count = 0
    for step in range(n_steps):
        batch = agents.get_active_batch()
        if batch.n > 0:
            outputs = torch.randn(batch.n, 5, device=device)
            outputs[:, 0] = 1.0
            # 红皇后机制: 15%概率产生捕食者
            attack_mask = torch.rand(batch.n, device=device) < 0.15
            outputs[attack_mask, 3] = torch.rand(attack_mask.sum(), device=device) * 3.0  # 更强攻击
            
            agents._apply_physics(batch, outputs, 0.1)
            agents._apply_metabolism(batch, 0.1)
            agents._apply_environment_interaction(batch, env)
            agents._apply_predation(batch, outputs)
            agents._process_deaths(batch, env)
            agents._process_reproduction(batch)
            agents._apply_boundaries(batch)
        
        env.step()
        step_count += 1
        
        # 每100步记录
        if step % 100 == 0:
            s = agents.get_population_stats()
            batch = agents.get_active_batch()
            
            history['step'].append(step)
            history['population'].append(s['n_alive'])
            history['mean_energy'].append(s['mean_energy'])
            history['max_energy'].append(s['max_energy'])
            
            if batch.n > 0:
                history['mean_nodes'].append(agents.state.node_counts[batch.indices].float().mean().item())
                history['mean_age'].append(agents.state.ages[batch.indices].mean().item())
                
                # 节点分布
                node_counts = agents.state.node_counts[batch.indices]
                for n in range(1, 15):
                    count = (node_counts == n).sum().item()
                    history['node_dist'][n].append(count)
            else:
                history['mean_nodes'].append(0)
                history['mean_age'].append(0)
                for n in range(1, 15):
                    history['node_dist'][n].append(0)
            
            print(f"{step//100}", end=" ", flush=True)
        
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
    
    # 1. 人口
    ax = axes[0, 0]
    ax.plot(history['step'], history['population'], 'b-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('种群数量 (Populations)')
    ax.grid(True, alpha=0.3)
    
    # 2. 能量
    ax = axes[0, 1]
    ax.plot(history['step'], history['max_energy'], 'r-', label='Max', linewidth=1)
    ax.plot(history['step'], history['mean_energy'], 'orange', label='Mean', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Energy')
    ax.set_title('能量 (Energy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 平均节点数
    ax = axes[1, 0]
    ax.plot(history['step'], history['mean_nodes'], 'g-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Nodes')
    ax.set_title('平均节点数 (Brain Complexity)')
    ax.grid(True, alpha=0.3)
    
    # 4. 平均年龄
    ax = axes[1, 1]
    ax.plot(history['step'], history['mean_age'], 'purple', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Age')
    ax.set_title('平均年龄 (Metabolic Senescence)')
    ax.grid(True, alpha=0.3)
    
    # 5. 节点分布堆叠图
    ax = axes[2, 0]
    for n in range(1, 10):
        if any(history['node_dist'][n]):
            ax.plot(history['step'], history['node_dist'][n], label=f'n{n}', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('节点数分布演化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 节点分布热力图
    ax = axes[2, 1]
    node_matrix = []
    for n in range(1, 10):
        node_matrix.append(history['node_dist'][n])
    node_matrix = torch.tensor(node_matrix).float()
    im = ax.imshow(node_matrix.numpy(), aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Step (x100)')
    ax.set_ylabel('Node Count')
    ax.set_yticks(range(9))
    ax.set_yticklabels(range(1, 10))
    ax.set_title('节点分布热力图')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('evolution_10000.png', dpi=150)
    print(f"✅ 图表已保存: evolution_10000.png")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("🏆 最终结果")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        top_idx = batch.indices[batch.energies.topk(10).indices]
        
        print("\nTop 10 Agent:")
        for i, idx in enumerate(top_idx.tolist()):
            e = agents.state.energies[idx].item()
            n = agents.state.node_counts[idx].item()
            age = agents.state.ages[idx].item()
            print(f"  #{i+1}: 能量={e:.2f}, 节点={n}, 年龄={age:.0f}")
        
        print(f"\n【最终统计】")
        print(f"  种群: {batch.n}")
        print(f"  平均能量: {batch.energies.mean():.2f}")
        print(f"  平均节点: {agents.state.node_counts[batch.indices].float().mean():.2f}")
        print(f"  平均年龄: {agents.state.ages[batch.indices].mean():.1f}")
        
        # 节点分布
        print(f"\n【节点分布】")
        for n in range(3, 12):
            count = (agents.state.node_counts[batch.indices] == n).sum().item()
            if count > 0:
                print(f"  节点{n}: {count}个")


if __name__ == "__main__":
    run()