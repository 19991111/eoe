"""
100代演化测试 - 观察最成功Agent
===============================
运行100代，追踪最强大的Agent结构
"""

import torch
import time
import sys
sys.path.insert(0, '.')

from core.eoe.batched_agents import BatchedAgents, PoolConfig
from core.eoe.environment_gpu import EnvironmentGPU
from core.eoe.genome import OperatorGenome


def run_100_generations():
    """运行100代演化"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🧪 100代演化测试")
    print(f"{'='*60}")
    
    # 配置
    n_generations = 100
    steps_per_gen = 100
    initial_pop = 100
    max_agents = 1000
    
    print(f"配置: {n_generations}代 x {steps_per_gen}步, 初始{initial_pop}, 最大{max_agents}")
    
    # 环境
    env = EnvironmentGPU(
        width=100, height=100,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=True
    )
    
    # Agent池
    agents = BatchedAgents(
        initial_population=initial_pop,
        max_agents=max_agents,
        env_width=100,
        env_height=100,
        device=device,
        init_energy=100.0  # 降低初始能量，减少早期分裂
    )
    
    # 初始化基因组 (简单大脑)
    for idx in agents.alive_mask.nonzero(as_tuple=True)[0]:
        agents.genomes[idx] = OperatorGenome()
        # 添加基本节点
        from core.eoe.node import Node, NodeType
        # 输入节点
        sensors = Node(node_id=0, node_type=NodeType.SENSOR)
        sensors.position = 0
        agents.genomes[idx].add_node(sensors)
        # 隐层
        hidden = Node(node_id=1, node_type=NodeType.MULTIPLY)
        hidden.position = 1
        agents.genomes[idx].add_node(hidden)
        # 输出
        actuator = Node(node_id=2, node_type=NodeType.ACTUATOR)
        actuator.position = 2
        agents.genomes[idx].add_node(actuator)
        # 边
        agents.genomes[idx].edges.append({
            'source_id': 0,
            'target_id': 1,
            'weight': 1.0,
            'enabled': True
        })
        agents.genomes[idx].edges.append({
            'source_id': 1,
            'target_id': 2,
            'weight': 1.0,
            'enabled': True
        })
        agents.state.node_counts[idx] = 3
    
    # 追踪最优秀Agent
    best_energy_history = []
    best_genome = None
    best_energy = 0
    
    print(f"\n{'Gen':>4} | {'存活':>6} | {'出生':>4} | {'死亡':>4} | {'最高能量':>10} | {'平均能量':>10}")
    print("-" * 70)
    
    for gen in range(n_generations):
        gen_births = 0
        gen_deaths = 0
        
        for step in range(steps_per_gen):
            stats = agents.step(env, dt=0.1)
            gen_births += stats['births']
            gen_deaths += stats['deaths']
            
            # 环境步进
            env.step()
        
        # 统计
        pop_stats = agents.get_population_stats()
        
        # 找最优秀Agent
        batch = agents.get_active_batch()
        if batch.n > 0:
            max_e = batch.energies.max().item()
            if max_e > best_energy:
                best_energy = max_e
                # 找到对应索引
                max_idx = batch.indices[batch.energies.argmax()].item()
                if max_idx in agents.genomes:
                    best_genome = agents.genomes[max_idx]
        
        best_energy_history.append(pop_stats['max_energy'])
        
        if gen % 10 == 0:
            print(f"{gen:>4} | {pop_stats['n_alive']:>6} | {gen_births:>4} | {gen_deaths:>4} | "
                  f"{pop_stats['max_energy']:>10.2f} | {pop_stats['mean_energy']:>10.2f}")
        
        # 代际补充 (保持种群稳定用于测试)
        if pop_stats['n_alive'] < initial_pop:
            needed = initial_pop - pop_stats['n_alive']
            dead_mask = ~agents.alive_mask
            if dead_mask.any():
                respawn = dead_mask.nonzero(as_tuple=True)[0][:needed]
                agents.alive_mask[respawn] = True
                agents.state.energies[respawn] = 100.0
                agents._indices_dirty = True
    
    # 最终结果
    print("\n" + "="*60)
    print("🏆 最终统计")
    print("="*60)
    
    final_stats = agents.get_population_stats()
    print(f"最终存活: {final_stats['n_alive']}")
    print(f"最高能量: {final_stats['max_energy']:.2f}")
    print(f"平均能量: {final_stats['mean_energy']:.2f}")
    
    # 分析最优秀结构
    print("\n" + "="*60)
    print("🧠 最厉害的结构分析")
    print("="*60)
    
    batch = agents.get_active_batch()
    if batch.n > 0:
        energies = batch.energies
        top_indices = batch.indices[energies.topk(5).indices]
        
        print("\nTop 5 Agent:")
        for i, idx in enumerate(top_indices.tolist()):
            e = agents.state.energies[idx].item()
            nodes = agents.state.node_counts[idx].item()
            print(f"  #{i+1} 索引={idx}, 能量={e:.2f}, 节点数={nodes}")
            
            if idx in agents.genomes:
                g = agents.genomes[idx]
                print(f"      节点: {len(g.nodes)}个, 边: {len(g.edges)}条")
                for node in list(g.nodes.values())[:5]:
                    print(f"        - {node.node_type.name} (id={node.node_id})")
    
    return agents, env, best_energy_history


if __name__ == "__main__":
    run_100_generations()