"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
EOE 演化可视化 - 终端版本
=========================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

无需 matplotlib 的 ASCII 可视化
在每一代结束时输出状态快照

功能:
- ASCII 艺术 2D 沙盒
- 脑结构文本打印
- 适应度曲线 (使用字符绘图)
"""

import numpy as np
import core
from core import Agent, Population, NodeType
from typing import List, Dict, Optional


class ASCIIVisualizer:
    """
    终端 ASCII 可视化器
    """
    
    def __init__(self, population: Population):
        self.population = population
        self.history: List[Dict] = []
        
        # 曲线数据
        self.best_fitness = []
        self.avg_fitness = []
        self.avg_nodes = []
    
    def _draw_sandbox(self, ax=None, ay=None, width=40, height=15) -> str:
        """绘制 ASCII 沙盒"""
        env = self.population.environment
        
        # 创建空网格
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # 映射坐标
        x_scale = width / env.width
        y_scale = height / env.height
        
        # 放置食物
        fx = int(env.target_pos[0] * x_scale)
        fy = int(env.target_pos[1] * y_scale)
        if 0 <= fx < width and 0 <= fy < height:
            grid[height - 1 - fy][fx] = '●'  # 食物
        
        # 找到最佳智能体
        best_agent = max(self.population.agents, key=lambda a: a.fitness)
        
        # 放置智能体
        for agent in self.population.agents:
            ax = int(agent.x * x_scale)
            ay = int(agent.y * y_scale)
            
            if 0 <= ax < width and 0 <= ay < height:
                if agent is best_agent:
                    grid[height - 1 - ay][ax] = '◆'  # 最佳
                else:
                    grid[height - 1 - ay][ax] = '·'  # 普通
        
        # 转换为字符串
        lines = []
        lines.append("┌" + "─" * width + "┐")
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * width + "┘")
        
        return "\n".join(lines)
    
    def _draw_brain(self, agent: Agent) -> str:
        """绘制 ASCII 大脑结构"""
        genome = agent.genome
        nodes = genome.nodes
        edges = genome.edges
        
        if not nodes:
            return "Empty brain"
        
        # 简化的层次表示
        lines = []
        lines.append("  SENSOR: " + " ".join(f"#{n.node_id}" for n in nodes.values() if n.node_type == NodeType.SENSOR))
        
        # 中间层
        middle = [n for n in nodes.values() if n.node_type in (NodeType.ADD, NodeType.MULTIPLY, NodeType.DELAY)]
        if middle:
            middle_str = "  MIDDLE: " + " ".join(f"#{n.node_id}({n.node_type.name[:3]})" for n in sorted(middle, key=lambda x: x.node_id))
            lines.append(middle_str)
        
        lines.append("  ACTUATR: " + " ".join(f"#{n.node_id}" for n in nodes.values() if n.node_type == NodeType.ACTUATOR))
        
        lines.append("")
        
        # 边列表
        lines.append("  Connections:")
        for edge in edges:
            if not edge['enabled']:
                continue
            src = nodes.get(edge['source_id'])
            tgt = nodes.get(edge['target_id'])
            if src and tgt:
                lines.append(f"    #{edge['source_id']}--[{edge['weight']:+.2f}]-->#{edge['target_id']}")
        
        # 统计
        info = genome.get_info()
        lines.append(f"\n  [Nodes: {info['total_nodes']}, Edges: {info['enabled_edges']}, Fitness: {agent.fitness:.2f}]")
        
        return "\n".join(lines)
    
    def _draw_curve(self) -> str:
        """绘制 ASCII 适应度曲线"""
        if not self.best_fitness:
            return "No data yet"
        
        lines = []
        lines.append("  Evolution Curve:")
        
        # 归一化数据
        all_vals = self.best_fitness + self.avg_fitness + self.avg_nodes
        vmin, vmax = min(all_vals), max(all_vals)
        vrange = vmax - vmin if vmax != vmin else 1
        
        width = 30
        
        for i, (best, avg, nodes) in enumerate(zip(self.best_fitness, self.avg_fitness, self.avg_nodes)):
            # 进度条
            best_bar = int((best - vmin) / vrange * width)
            avg_bar = int((avg - vmin) / vrange * width)
            nodes_bar = int(nodes / 10 * width)  # 假设节点数 0-10
            
            line = f"  Gen{i:2d}: "
            line += "B" * best_bar + "·" * (width - best_bar)
            line += f" {best:.1f}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def run(self, n_generations: int = 10, verbose: bool = True):
        """运行可视化演化"""
        print("=" * 70)
        print(" EOE Evolution - ASCII Visualization")
        print("=" * 70)
        print(f" Population: {self.population.population_size}")
        print(f" Lifespan: {self.population.lifespan}")
        print(f" Generations: {n_generations}")
        print("=" * 70)
        
        for gen in range(n_generations):
            # 运行一代
            stats = self.population.epoch(verbose=False)
            self.history.append(stats)
            
            # 记录数据
            self.best_fitness.append(stats['best_fitness'])
            self.avg_fitness.append(stats['avg_fitness'])
            self.avg_nodes.append(stats['avg_nodes'])
            
            # 找到最佳智能体
            best_agent = stats['best_agent']
            
            if verbose:
                print(f"\n{'='*70}")
                print(f" GENERATION {gen}")
                print(f"{'='*70}")
                
                # 沙盒
                print("\n[2D SANDBOX]")
                print(self._draw_sandbox())
                
                # 大脑
                print("\n[BEST AGENT BRAIN]")
                print(self._draw_brain(best_agent))
                
                # 曲线
                print("\n[EVOLUTION STATS]")
                print(f"  Best Fitness: {stats['best_fitness']:.2f}")
                print(f"  Avg Fitness:  {stats['avg_fitness']:.2f}")
                print(f"  Avg Nodes:    {stats['avg_nodes']:.1f}")
                print(f"  Avg Edges:    {stats['avg_edges']:.1f}")
                
                print("\n[FITNESS CURVE]")
                print(self._draw_curve())
            
            # 演化下一代
            if gen < n_generations - 1:
                self.population.reproduce(verbose=False)
        
        return self.history


def run_ascii_visualization(
    population_size: int = 30,
    n_generations: int = 10,
    lifespan: int = 100
):
    """运行 ASCII 可视化"""
    pop = Population(
        population_size=population_size,
        elite_ratio=0.2,
        lifespan=lifespan,
        metabolic_alpha=0.1,
        metabolic_beta=0.05
    )
    
    visualizer = ASCIIVisualizer(pop)
    history = visualizer.run(n_generations=n_generations, verbose=True)
    
    return pop, history


if __name__ == "__main__":
    import sys
    
    n_gens = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    lifespan = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    pop, history = run_ascii_visualization(
        population_size=pop_size,
        n_generations=n_gens,
        lifespan=lifespan
    )
    
    print("\n" + "="*70)
    print(" FINAL RESULTS")
    print("="*70)
    print(f" Final Best Fitness: {history[-1]['best_fitness']:.2f}")
    print(f" Final Avg Nodes: {history[-1]['avg_nodes']:.1f}")
    print(f" Final Avg Edges: {history[-1]['avg_edges']:.1f}")