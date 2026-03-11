"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
EOE 可视化模块
==============
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
提供脑结构和演化结果的可视化功能
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import core
from core import Agent, Population, NodeType


def print_brain_text(agent: Agent) -> str:
    """
    生成脑结构的ASCII文本表示
    
    参数:
        agent: 智能体
    
    返回:
        ASCII文本
    """
    genome = agent.genome
    nodes = genome.nodes
    edges = genome.edges
    
    lines = []
    lines.append("=" * 50)
    lines.append(f"Agent Brain Structure")
    lines.append(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    lines.append(f"Fitness: {agent.fitness:.2f}")
    lines.append("=" * 50)
    
    # 按类型分组
    by_type = {}
    for node_id, node in nodes.items():
        t = node.node_type.name
        if t not in by_type:
            by_type[t] = []
        by_type[t].append((node_id, node))
    
    # 打印节点
    lines.append("\n[Nodes]")
    for node_type, node_list in sorted(by_type.items()):
        ids = [str(nid) for nid, _ in node_list]
        lines.append(f"  {node_type}: {', '.join(ids)}")
    
    # 打印边
    lines.append("\n[Edges]")
    for edge_key, edge in edges.items():
        src, tgt = edge_key.split("->")
        w = edge.get('weight', 1.0)
        lines.append(f"  {src} -> {tgt} (w={w:.2f})")
    
    return "\n".join(lines)


def visualize_brain_ascii(agent: Agent) -> None:
    """在终端打印脑结构"""
    print(print_brain_text(agent))


def visualize_brain(
    agent: Agent,
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None
):
    """
    可视化智能体的大脑拓扑图
    
    参数:
        agent: 智能体
        ax: matplotlib axes (可选)
        figsize: 图形大小
        title: 标题 (可选)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    genome = agent.genome
    nodes = genome.nodes
    
    if not nodes:
        print("Warning: Empty genome!")
        return
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 层次布局
    layers = {
        NodeType.SENSOR: [],
        NodeType.CONSTANT: [],
        NodeType.ADD: [],
        NodeType.MULTIPLY: [],
        NodeType.DELAY: [],
        NodeType.THRESHOLD: [],
        NodeType.PREDICTOR: [],
        NodeType.ACTUATOR: [],
    }
    
    for node_id, node in nodes.items():
        if node.node_type in layers:
            layers[node.node_type].append((node_id, node))
        else:
            layers[NodeType.ADD].append((node_id, node))
    
    # 计算位置
    positions = {}
    y_positions = {
        NodeType.SENSOR: 3.5,
        NodeType.PREDICTOR: 3.0,
        NodeType.CONSTANT: 2.5,
        NodeType.ADD: 2.0,
        NodeType.MULTIPLY: 1.5,
        NodeType.DELAY: 1.5,
        NodeType.THRESHOLD: 1.0,
        NodeType.ACTUATOR: 0.5,
    }
    
    for node_type, node_list in layers.items():
        if not node_list:
            continue
        y = y_positions.get(node_type, 2.0)
        n = len(node_list)
        spacing = 6.0 / (n + 1)
        
        for i, (node_id, _) in enumerate(node_list):
            x = -3.0 + spacing * (i + 1)
            positions[node_id] = (x, y)
    
    # 绘制边
    for edge_key, edge in genome.edges.items():
        if not edge.get('enabled', True):
            continue
        src, tgt = edge_key.split("->")
        src_id, tgt_id = int(src), int(tgt)
        
        if src_id in positions and tgt_id in positions:
            x1, y1 = positions[src_id]
            x2, y2 = positions[tgt_id]
            weight = edge.get('weight', 1.0)
            color = 'green' if weight > 0 else 'red'
            alpha = min(1.0, abs(weight) / 2.0 + 0.3)
            
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle="->", color=color, alpha=alpha,
                                      lw=max(1, abs(weight))))
    
    # 绘制节点
    node_colors = {
        NodeType.SENSOR: 'green',
        NodeType.ACTUATOR: 'red',
        NodeType.ADD: 'blue',
        NodeType.MULTIPLY: 'orange',
        NodeType.DELAY: 'purple',
        NodeType.THRESHOLD: 'yellow',
        NodeType.PREDICTOR: 'cyan',
        NodeType.CONSTANT: 'gray',
    }
    
    for node_id, node in nodes.items():
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        color = node_colors.get(node.node_type, 'white')
        
        circle = plt.Circle((x, y), 0.3, color=color, ec='black', zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(node_id), ha='center', va='center', 
               fontsize=10, fontweight='bold', zorder=11)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    info = genome.get_info()
    if title is None:
        title = f"Brain (Nodes: {info['total_nodes']}, Edges: {info['enabled_edges']})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='SENSOR'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='ACTUATOR'),
        mpatches.Patch(facecolor='blue', edgecolor='black', label='ADD'),
        mpatches.Patch(facecolor='orange', edgecolor='black', label='MULTIPLY'),
        mpatches.Patch(facecolor='purple', edgecolor='black', label='DELAY'),
        mpatches.Patch(facecolor='gray', edgecolor='black', label='CONSTANT')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    return ax


def visualize_population_best(population: Population, generation: int = None):
    """可视化当前种群中最佳智能体"""
    import matplotlib.pyplot as plt
    
    best_agent = max(population.agents, key=lambda a: a.fitness)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    gen_num = generation if generation is not None else population.generation
    visualize_brain(best_agent, ax=ax, title=f"Gen {gen_num} - Best")
    
    plt.tight_layout()
    return fig, best_agent