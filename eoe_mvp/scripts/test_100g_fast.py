"""
100代演化测试 - 简化版
======================
"""

import torch
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType


def create_simple_genome():
    """创建简单基因组"""
    g = OperatorGenome()
    g.add_node(Node(node_id=0, node_type=NodeType.SENSOR))
    g.add_node(Node(node_id=1, node_type=NodeType.MULTIPLY))
    g.add_node(Node(node_id=2, node_type=NodeType.ACTUATOR))
    g.add_edge(0, 1, 1.0)
    g.add_edge(1, 2, 1.0)
    return g


def run_100_generations():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    n_gen = 100
    steps = 50  # 减少步数加速
    init_pop = 50
    max_agents = 500
    
    env = EnvironmentGPU(width=100, height=100, device=device, energy_field_enabled=True)
    agents = BatchedAgents(initial_population=init_pop, max_agents=max_agents, device=device, init_energy=80.0)
    
    # 初始化基因组
    batch = agents.get_active_batch()
    for idx in batch.indices.tolist():
        agents.genomes[idx] = create_simple_genome()
        agents.state.node_counts[idx] = 3
    
    print(f"\nGen | 存活 | 出生 | 死亡 | 最高能")
    print("-" * 45)
    
    for gen in range(n_gen):
        births = 0
        deaths = 0
        
        for _ in range(steps):
            stats = agents.step(env, dt=0.1)
            births += stats['births']
            deaths += stats['deaths']
            env.step()
        
        stats = agents.get_population_stats()
        
        if gen % 10 == 0:
            print(f"{gen:3d} | {stats['n_alive']:4d} | {births:4d} | {deaths:4d} | {stats['max_energy']:7.2f}")
        
        # 补充
        if stats['n_alive'] < init_pop:
            needed = init_pop - stats['n_alive']
            dead = (~agents.alive_mask).nonzero(as_tuple=True)[0]
            if len(dead) > 0:
                respawn = dead[:needed]
                agents.alive_mask[respawn] = True
                agents.state.energies[respawn] = 80.0
                agents._indices_dirty = True
    
    # 最终分析
    print("\n" + "="*50)
    print("🏆 Top 5 Agent")
    print("="*50)
    
    batch = agents.get_active_batch()
    energies = batch.energies
    top_idx = batch.indices[energies.topk(5).indices]
    
    for i, idx in enumerate(top_idx.tolist()):
        e = agents.state.energies[idx].item()
        n = agents.state.node_counts[idx].item()
        print(f"#{i+1}: 索引={idx}, 能量={e:.2f}, 节点={n}")
        
        if idx in agents.genomes:
            g = agents.genomes[idx]
            print(f"    节点: {[(n.node_type.name, n.node_id) for n in g.nodes.values()]}")
            print(f"    边: {[(e['source_id'], e['target_id'], e['weight']) for e in g.edges]}")


if __name__ == "__main__":
    run_100_generations()