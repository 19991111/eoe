"""
EOE 演化最终结果生成器
======================

运行长代数演化，生成最终结果图片

功能:
- 长代数 (100+)
- 短生命周期 (20-30步)
- 最终生成可视化图片
- 测试种群是否能自然增长

作者: 104助手
日期: 2026-03-06
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import core
from core import Agent, Population, NodeType
from typing import List, Dict


def run_evolution_with_population_growth(
    n_generations: int = 100,
    population_size: int = 20,
    lifespan: int = 25,
    metabolic_alpha: float = 0.1,
    metabolic_beta: float = 0.05,
    growth_enabled: bool = True,
    growth_threshold: float = -10.0,  # 到达此适应度可繁殖
    verbose: bool = True
) -> tuple:
    """
    运行演化，支持种群自然增长
    
    参数:
        n_generations: 演化代数
        population_size: 初始种群大小
        lifespan: 每代生命周期
        metabolic_alpha: 节点代谢惩罚
        metabolic_beta: 边代谢惩罚
        growth_enabled: 是否允许种群增长
        growth_threshold: 达到此 fitness 可产生新智能体
    
    返回:
        (population, history)
    """
    pop = Population(
        population_size=population_size,
        elite_ratio=0.15,  # 稍微降低精英比例
        lifespan=lifespan,
        metabolic_alpha=metabolic_alpha,
        metabolic_beta=metabolic_beta
    )
    
    history = []
    
    if verbose:
        print("=" * 60)
        print("EOE Evolution with Population Growth")
        print("=" * 60)
        print(f"Generations: {n_generations}")
        print(f"Initial Population: {population_size}")
        print(f"Lifespan: {lifespan}")
        print(f"Growth Enabled: {growth_enabled}")
        print(f"Growth Threshold: {growth_threshold}")
        print("=" * 60)
    
    for gen in range(n_generations):
        # 运行一代
        stats = pop.epoch(verbose=False)
        history.append(stats)
        
        # 自然增长: 达到阈值的智能体可以复制
        if growth_enabled:
            new_agents = []
            for agent in pop.agents:
                if agent.fitness > growth_threshold:
                    # 精英复制
                    child_genome = agent.genome.copy()
                    child = Agent(
                        agent_id=len(pop.agents) + len(new_agents),
                        x=np.random.uniform(10, pop.environment.width - 10),
                        y=np.random.uniform(10, pop.environment.height - 10),
                        theta=np.random.uniform(0, 2 * np.pi)
                    )
                    child.genome = child_genome
                    # 轻微突变
                    child.genome.mutate_weight(sigma=0.05)
                    new_agents.append(child)
            
            # 限制最大种群
            max_pop = population_size * 3
            if len(new_agents) > 0 and len(pop.agents) < max_pop:
                # 添加部分子代
                n_to_add = min(len(new_agents), max_pop - len(pop.agents))
                pop.agents.extend(new_agents[:n_to_add])
                pop.environment.agents = pop.agents
                pop.population_size = len(pop.agents)
        
        # 打印进度
        if verbose and (gen % 10 == 0 or gen == n_generations - 1):
            best = stats['best_agent']
            info = best.genome.get_info()
            print(f"Gen {gen:3d}: best={stats['best_fitness']:7.2f}, "
                  f"pop={len(pop.agents):3d}, "
                  f"nodes={info['total_nodes']}, "
                  f"edges={info['enabled_edges']}")
        
        # 演化下一代
        if gen < n_generations - 1:
            pop.reproduce(verbose=False)
    
    return pop, history


def generate_final_visualization(population: Population, history: List[Dict], save_path: str = "eoe_final_result.png"):
    """生成最终可视化图片"""
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('EOE - Embodied Operator Evolution Final Result', 
                 fontsize=18, fontweight='bold')
    
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                  wspace=0.25, hspace=0.3)
    
    # 子图1: 2D 沙盒
    ax1 = fig.add_subplot(gs[:, 0])
    env = population.environment
    agents = population.agents
    
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'2D Sandbox - Final Generation ({population.generation})', fontsize=14)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 食物
    ax1.plot(env.target_pos[0], env.target_pos[1], 'go', markersize=15, 
             markeredgecolor='darkgreen', label='Food')
    
    # 所有智能体
    if agents:
        ax1.plot([a.x for a in agents], [a.y for a in agents], 'bo', 
                 markersize=8, alpha=0.5, label='Agents')
        
        # 最佳智能体
        best = max(agents, key=lambda a: a.fitness)
        ax1.plot(best.x, best.y, 'ro', markersize=12, markeredgecolor='darkred',
                 label=f'Best (fitness={best.fitness:.1f})')
    
    ax1.legend(loc='upper right', fontsize=10)
    
    # 子图2: 最佳大脑
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_brain(best, ax2)
    ax2.set_title(f'Best Agent Brain', fontsize=14)
    
    # 子图3: 演化曲线
    ax3 = fig.add_subplot(gs[1, 1])
    gens = list(range(len(history)))
    best_fits = [h['best_fitness'] for h in history]
    avg_fits = [h['avg_fitness'] for h in history]
    avg_nodes = [h['avg_nodes'] for h in history]
    
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Fitness', fontsize=12)
    ax3.set_title('Evolution Statistics', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    ax3.plot(gens, best_fits, 'b-', linewidth=2, label='Best Fitness')
    ax3.plot(gens, avg_fits, 'g--', linewidth=1.5, label='Avg Fitness')
    ax3.set_xlim(0, max(gens) + 1)
    ax3.legend(loc='upper left')
    
    # 添加节点数作为次坐标轴
    ax3_twin = ax3.twinx()
    ax3_twin.plot(gens, avg_nodes, 'r:', linewidth=2, label='Avg Nodes')
    ax3_twin.set_ylabel('Avg Nodes', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 统计信息框
    info_text = f"""Final Statistics:
Generations: {len(history)}
Population: {len(agents)}
Best Fitness: {best.fitness:.2f}
Best Nodes: {best.genome.get_info()['total_nodes']}
Best Edges: {best.genome.get_info()['enabled_edges']}"""
    
    fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 结果已保存到: {save_path}")
    
    return fig


def _draw_brain(agent: Agent, ax):
    """绘制大脑拓扑"""
    genome = agent.genome
    nodes = genome.nodes
    edges = genome.edges
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    if not nodes:
        ax.text(0.5, 0.5, 'Empty Brain', ha='center', va='center')
        return
    
    # 层次布局
    positions = {}
    layer_nodes = {0: [], 1: [], 2: [], 3: []}
    
    for node in nodes.values():
        if node.node_type == NodeType.SENSOR:
            layer_nodes[0].append(node.node_id)
        elif node.node_type == NodeType.CONSTANT:
            layer_nodes[1].append(node.node_id)
        elif node.node_type == NodeType.ACTUATOR:
            layer_nodes[3].append(node.node_id)
        else:
            layer_nodes[2].append(node.node_id)
    
    for layer, node_ids in layer_nodes.items():
        n = len(node_ids)
        for i, nid in enumerate(sorted(node_ids)):
            x = (i - (n - 1) / 2) * 1.8 if n > 0 else 0
            positions[nid] = (x, 3 - layer)
    
    # 边
    for edge in edges:
        if not edge['enabled']:
            continue
        start = positions.get(edge['source_id'])
        end = positions.get(edge['target_id'])
        if start and end:
            color = 'blue' if edge['weight'] > 0 else 'red'
            lw = max(0.5, abs(edge['weight']) * 2)
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                      connectionstyle='arc3,rad=0.1'))
    
    # 节点
    styles = {
        NodeType.SENSOR: ('green', 'o', 'S'),
        NodeType.ACTUATOR: ('red', 'o', 'A'),
        NodeType.ADD: ('blue', 's', '+'),
        NodeType.MULTIPLY: ('orange', 's', '×'),
        NodeType.DELAY: ('purple', 'D', '⏱'),
        NodeType.CONSTANT: ('gray', '^', 'C')
    }
    
    for nid, (x, y) in positions.items():
        node = nodes[nid]
        color, marker, label = styles.get(node.node_type, ('black', 'o', '?'))
        size = 400 + min(abs(node.activation) * 100, 600)
        ax.scatter(x, y, s=size, c=color, marker=marker, 
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', zorder=20)
        ax.text(x, y - 0.25, f'{nid}', ha='center', va='top', 
                fontsize=7, color='gray')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, 3.8)
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='SENSOR'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='ACTUATOR'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='ADD (+)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', label='MULTIPLY'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', label='DELAY'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)


if __name__ == "__main__":
    import sys
    
    # 参数
    N_GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # 100 代
    POP_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 15  # 15 个智能体
    LIFESPAN = int(sys.argv[3]) if len(sys.argv) > 3 else 20  # 20 步/代
    GROWTH = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True  # 允许种群增长
    
    # 运行演化
    pop, history = run_evolution_with_population_growth(
        n_generations=N_GEN,
        population_size=POP_SIZE,
        lifespan=LIFESPAN,
        growth_enabled=GROWTH,
        growth_threshold=-15.0
    )
    
    # 生成可视化
    generate_final_visualization(pop, history)
    
    # 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Generations: {len(history)}")
    print(f"Final Population: {len(pop.agents)}")
    
    # v7.2: 保存物种结构
    import pickle
    import json
    import numpy as np
    import os
    
    os.makedirs("species_archive", exist_ok=True)
    
    # 按适应度排序
    sorted_agents = sorted(pop.agents, key=lambda a: a.fitness, reverse=True)
    n_top = max(1, int(len(sorted_agents) * 0.2))
    
    # 保存最强结构
    best_agent = sorted_agents[0]
    best_data = {
        'generation': len(history),
        'fitness': best_agent.fitness,
        'food_eaten': best_agent.food_eaten,
        'genome_pkl': pickle.dumps(best_agent.genome)
    }
    with open("species_archive/best_structure.pkl", 'wb') as f:
        pickle.dump(best_data, f)
    
    # 保存top 20%
    top_agents = sorted_agents[:n_top]
    top_data = []
    for i, agent in enumerate(top_agents):
        top_data.append({
            'rank': i + 1,
            'fitness': agent.fitness,
            'food_eaten': agent.food_eaten,
            'nodes': len(agent.genome.nodes),
            'edges': len(agent.genome.edges),
            'genome_pkl': pickle.dumps(agent.genome)
        })
    with open("species_archive/top_20_percent.pkl", 'wb') as f:
        pickle.dump(top_data, f)
    
    # 元数据
    metadata = {
        'generation': len(history),
        'total_agents': len(pop.agents),
        'top_20_count': n_top,
        'best_fitness': best_agent.fitness,
        'avg_fitness': float(np.mean([a.fitness for a in top_agents])),
        'avg_nodes': float(np.mean([len(a.genome.nodes) for a in top_agents])),
        'avg_edges': float(np.mean([len(a.genome.edges) for a in top_agents]))
    }
    with open("species_archive/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ 物种已保存到: species_archive/")
    print(f"   最强结构: best_structure.pkl")
    print(f"   Top 20%: top_20_percent.pkl ({n_top}个)")
    print(f"Best Fitness: {history[-1]['best_fitness']:.2f}")
    print(f"Avg Nodes: {history[-1]['avg_nodes']:.1f}")
    print(f"Avg Edges: {history[-1]['avg_edges']:.1f}")