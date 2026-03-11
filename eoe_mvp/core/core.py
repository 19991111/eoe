"""
具身算子演化 (Embodied Operator Evolution, EOE) - 核心模块
==============================================================

本模块是EOE系统的主入口，通过eoe子模块提供完整功能。

作者: 104助手
日期: 2026-03-09
版本: v0.74 (模块化重构)
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

# 确保eoe模块在路径中
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# 从eoe子模块导入所有核心类
# ============================================================
from eoe.node import Node, NodeType, SuperNode
from eoe.genome import OperatorGenome
from eoe.agent import Agent
from eoe.environment import Environment, ChunkManager
from eoe.population import Population

# ============================================================
# v7.2: Numba 加速函数 (性能优化)
# ============================================================
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: Numba not available, using NumPy fallback")

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def compute_distances_vectorized(x1, y1, x2_arr, y2_arr):
        """向量化计算距离矩阵"""
        n = len(x2_arr)
        distances = np.empty(n, dtype=np.float64)
        for i in prange(n):
            dx = x2_arr[i] - x1
            dy = y2_arr[i] - y1
            distances[i] = np.sqrt(dx*dx + dy*dy)
        return distances
    
    @jit(nopython=True, cache=True)
    def update_positions_vectorized(x, y, theta, left_force, right_force, max_speed, turn_rate, width, height):
        """向量化位置更新"""
        diff = right_force - left_force
        new_theta = theta + diff * turn_rate
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
        
        avg_force = (left_force + right_force) / 2.0
        speed = max(-max_speed, min(max_speed, avg_force))
        
        new_x = x + np.cos(new_theta) * speed
        new_y = y + np.sin(new_theta) * speed
        
        new_x = new_x % width
        new_y = new_y % height
        
        return new_x, new_y, new_theta
    
    @jit(nopython=True, cache=True)
    def compute_sensor_vectorized(agent_x, agent_y, agent_theta, target_x, target_y, sensor_range):
        """向量化传感器计算"""
        dx = target_x - agent_x
        dy = target_y - agent_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:
            return np.array([1.0, 1.0])
        
        target_angle = np.arctan2(dy, dx)
        relative_angle = target_angle - agent_theta
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        sensor_offset = np.radians(30)
        sigma = np.radians(45)
        left_sensor = np.exp(-0.5 * ((relative_angle + sensor_offset) / sigma) ** 2)
        right_sensor = np.exp(-0.5 * ((relative_angle - sensor_offset) / sigma) ** 2)
        
        distance_decay = sensor_range / (distance + 1.0)
        distance_decay = max(0.0, min(1.0, distance_decay))
        
        return np.array([left_sensor * distance_decay, right_sensor * distance_decay])
else:
    def compute_distances_vectorized(x1, y1, x2_arr, y2_arr):
        return np.sqrt((x2_arr - x1)**2 + (y2_arr - y1)**2)
    
    def update_positions_vectorized(x, y, theta, left_force, right_force, max_speed, turn_rate, width, height):
        diff = right_force - left_force
        new_theta = (theta + diff * turn_rate) % (2 * np.pi)
        speed = np.clip((left_force + right_force) / 2.0, -max_speed, max_speed)
        return (x + np.cos(new_theta) * speed) % width, (y + np.sin(new_theta) * speed) % height, new_theta
    
    def compute_sensor_vectorized(agent_x, agent_y, agent_theta, target_x, target_y, sensor_range):
        dx, dy = target_x - agent_x, target_y - agent_y
        dist = np.hypot(dx, dy)
        if dist < 0.1:
            return np.array([1.0, 1.0])
        angle = np.arctan2(dy, dx) - agent_theta
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        sigma = np.radians(45)
        decay = np.clip(sensor_range / (dist + 1.0), 0.0, 1.0)
        return np.array([
            np.exp(-0.5 * ((angle + np.radians(30)) / sigma) ** 2),
            np.exp(-0.5 * ((angle - np.radians(30)) / sigma) ** 2)
        ]) * decay


# ============================================================
# v0.74: 纯生存适应度模式 (作为Population的参数)
# ============================================================
# 这个功能通过Population的pure_survival_mode参数启用
# 代码已在eoe.population.Population中实现


# ============================================================
# 独立工具函数
# ============================================================

def create_simple_agent(
    agent_id: int = 0,
    x: float = 50.0,
    y: float = 50.0,
    theta: float = 0.0,
    connection_type: str = "chemotaxis"
) -> Agent:
    """创建一个带有默认连接的智能体"""
    agent = Agent(agent_id=agent_id, x=x, y=y, theta=theta)
    
    if connection_type == "direct":
        agent.genome.add_edge(source_id=0, target_id=2, weight=1.0)
        agent.genome.add_edge(source_id=1, target_id=3, weight=1.0)
    elif connection_type == "chemotaxis":
        agent.genome.add_edge(source_id=0, target_id=3, weight=1.5)
        agent.genome.add_edge(source_id=1, target_id=2, weight=1.5)
        agent.genome.add_edge(source_id=0, target_id=2, weight=0.5)
        agent.genome.add_edge(source_id=1, target_id=3, weight=0.5)
    
    return agent


def demo():
    """演示函数"""
    print("=" * 60)
    print("EOE MVP 演示 - 具身算子演化")
    print("=" * 60)
    
    # 创建环境
    env = Environment(
        width=100.0,
        height=100.0,
        n_food=5,
        population_size=10
    )
    
    # 创建种群
    pop = Population(
        population_size=10,
        environment=env,
        lifespan=100
    )
    
    # 运行几代
    for gen in range(5):
        pop.epoch(verbose=True)
        pop.reproduce(verbose=False)
    
    # 可视化最佳个体
    best = max(pop.agents, key=lambda a: a.fitness)
    print(f"\n最佳适应度: {best.fitness:.2f}")
    print_brain_text(best)


def print_brain_text(agent: Agent) -> str:
    """打印脑结构文本表示"""
    lines = []
    lines.append(f"Agent {agent.id} 脑结构:")
    lines.append(f"  节点数: {len(agent.genome.nodes)}")
    lines.append(f"  边数: {len(agent.genome.edges)}")
    
    # 按类型统计节点
    type_counts = {}
    for node in agent.genome.nodes.values():
        t = node.node_type.name
        type_counts[t] = type_counts.get(t, 0) + 1
    
    lines.append("  节点类型分布:")
    for t, count in sorted(type_counts.items()):
        lines.append(f"    {t}: {count}")
    
    # 打印边
    lines.append("  边:")
    for i, edge in enumerate(agent.genome.edges[:10]):  # 只显示前10条
        src = edge.get('source_id', edge.get('src'))
        dst = edge.get('target_id', edge.get('dst'))
        w = edge.get('weight', 1.0)
        lines.append(f"    {src} → {dst} (w={w:.2f})")
    
    if len(agent.genome.edges) > 10:
        lines.append(f"    ... 还有 {len(agent.genome.edges) - 10} 条边")
    
    text = "\n".join(lines)
    print(text)
    return text


def visualize_brain(agent: Agent, max_nodes: int = 50):
    """可视化脑结构 (ASCII art)"""
    lines = []
    lines.append("=" * 50)
    lines.append(f"🧠 Agent {agent.id} 脑结构可视化")
    lines.append("=" * 50)
    
    # 节点信息
    lines.append(f"\n📊 统计: {len(agent.genome.nodes)} 节点, {len(agent.genome.edges)} 边")
    
    # 层次结构
    layers = {}
    for node in agent.genome.nodes.values():
        layer = getattr(node, 'layer', 1)
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
    
    lines.append("\n🔢 层次结构:")
    for layer in sorted(layers.keys()):
        nodes = layers[layer]
        node_strs = [f"{n.node_type.name[:4]}:{n.node_id}" for n in nodes[:10]]
        lines.append(f"  Layer {layer}: {', '.join(node_strs)}")
    
    # 活跃节点
    active = [n for n in agent.genome.nodes.values() if abs(n.activation) > 0.1]
    lines.append(f"\n⚡ 活跃节点: {len(active)}/{len(agent.genome.nodes)}")
    
    print("\n".join(lines))
    return "\n".join(lines)


def visualize_population_best(population: Population, generation: int = None):
    """可视化种群最佳个体"""
    if not population.agents:
        return
    
    best = max(population.agents, key=lambda a: a.fitness)
    
    print(f"\n{'='*50}")
    if generation is not None:
        print(f"🏆 第 {generation} 代最佳个体")
    else:
        print("🏆 最佳个体")
    print(f"{'='*50}")
    print(f"适应度: {best.fitness:.2f}")
    print(f"食物: {best.food_eaten}")
    print(f"存活: {'是' if best.is_alive else '否'}")
    print(f"节点: {len(best.genome.nodes)}, 边: {len(best.genome.edges)}")
    
    return best


def demo_evolution():
    """演化演示"""
    print("🚀 开始演化演示...")
    
    # 简化的演化流程
    pop = Population(
        population_size=20,
        lifespan=50,
        use_champion=True
    )
    
    for gen in range(10):
        pop.epoch(verbose=False)
        best = max(pop.agents, key=lambda a: a.fitness)
        print(f"Gen {gen}: best_fitness={best.fitness:.0f}, food={best.food_eaten}")
        pop.reproduce(verbose=False)
    
    print("✅ 演化完成!")
    visualize_population_best(pop, generation=10)


# ============================================================
# 兼容性导出
# ============================================================
__all__ = [
    # 核心类
    'Node',
    'NodeType', 
    'SuperNode',
    'OperatorGenome',
    'Agent',
    'Environment',
    'ChunkManager',
    'Population',
    # 工具函数
    'create_simple_agent',
    'demo',
    'print_brain_text',
    'visualize_brain',
    'visualize_population_best',
    'demo_evolution',
    # 性能优化
    'compute_distances_vectorized',
    'update_positions_vectorized',
    'compute_sensor_vectorized',
    'NUMBA_AVAILABLE',
]

__version__ = "0.74"