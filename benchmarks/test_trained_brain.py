"""
使用真实训练的大脑进行基准测试
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import json
import random
import pickle
from pathlib import Path

from core.eoe.batched_agents import PoolConfig
from core.eoe.agent import Agent
from benchmarks.benchmark_runner import BenchmarkRunner, TaskFactory
from benchmarks.visualization import BenchmarkReportGenerator


def load_trained_genome(brain_file: str = None, top_n: int = 1):
    """
    从保存的大脑pkl文件中加载训练好的大脑
    
    Args:
        brain_file: 大脑pkl文件路径
        top_n: 使用第N个大脑 (1=最复杂)
    
    Returns:
        OperatorGenome: 训练好的大脑
    """
    if brain_file is None:
        # 尝试从v18共演化结果加载
        exp_dir = Path("/home/node/.openclaw/workspace/eoe_mvp/outputs/v18_coevolution")
        brain_file = exp_dir / "best_brain.pkl"
        
        if not brain_file.exists():
            raise FileNotFoundError(f"No brain file found at {brain_file}")
    
    print(f"Loading brain from: {brain_file}")
    
    with open(brain_file, 'rb') as f:
        brain_template = pickle.load(f)
    
    # brain_template可能是一个genome或genome列表
    if isinstance(brain_template, list):
        if top_n > len(brain_template):
            top_n = len(brain_template)
        brain = brain_template[top_n - 1]
    else:
        brain = brain_template
    
    print(f"Loaded brain:")
    print(f"  Nodes: {len(brain.nodes)}")
    print(f"  Edges: {len(brain.edges)}")
    
    return brain
    """
    从保存的结构文件中加载训练好的大脑
    
    Args:
        structure_file: 结构JSON文件路径
        top_n: 使用第N复杂的结构 (1=最复杂)
    
    Returns:
        OperatorGenome: 训练好的大脑
    """
    if structure_file is None:
        # v16.17 (欺骗性景观实验) - 最新进化的大脑
        exp_dir = Path("/home/node/.openclaw/workspace/eoe_mvp/experiments/v16_deceptive_landscape/saved_structures")
        if not exp_dir.exists():
            # 后备: v15
            exp_dir = Path("/home/node/.openclaw/workspace/eoe_mvp/experiments/v15_cognitive_premium/saved_structures")
        files = [f for f in exp_dir.glob("complexity_step*.json") if "test" not in f.name]
        if not files:
            raise FileNotFoundError("No saved structures found")
        structure_file = str(sorted(files)[-1])  # 最新
    
    print(f"Loading from: {structure_file}")
    
    with open(structure_file, 'r') as f:
        data = json.load(f)
    
    structures = data.get('structures', {})
    if isinstance(structures, dict):
        structures = list(structures.values())
    
    # 按复杂度排序
    structures = sorted(structures, key=lambda s: s.get('complexity_score', 0), reverse=True)
    
    if top_n > len(structures):
        top_n = len(structures)
    
    # 选择第N复杂的结构
    selected = structures[top_n - 1]
    
    print(f"Selected structure (rank {top_n}):")
    print(f"  Complexity: {selected.get('complexity_score', 0):.2f}")
    print(f"  Nodes: {selected.get('nodes', [])}")
    print(f"  Population: {selected.get('population_count', 0)}")
    
    # 转换为OperatorGenome
    return _structure_to_genome(selected)


def _structure_to_genome(structure: dict) -> 'OperatorGenome':
    """将保存的结构转换为OperatorGenome"""
    from core.eoe.genome import OperatorGenome
    from core.eoe.node import Node, NodeType
    
    nodes = structure.get('nodes', [])
    edges = structure.get('edges', [])
    
    # 创建genome
    genome = OperatorGenome()
    
    # 添加节点
    for i, node_type_id in enumerate(nodes):
        try:
            node_type = NodeType(node_type_id)
        except (ValueError, TypeError):
            node_type = NodeType.SENSE_EPF_CENTER  # 默认
        
        node = Node(
            node_id=i,
            node_type=node_type,
            constant_value=random.uniform(-1, 1)
        )
        genome.nodes[i] = node
    
    # 添加连接
    for edge in edges:
        src, dst, weight = edge[0], edge[1], edge[2]
        genome.add_edge(
            source_id=src,
            target_id=dst,
            weight=weight * random.uniform(0.5, 1.5)  # 添加一些变化
        )
    
    # 添加必要的传感器和执行器
    if 1 not in genome.nodes:
        genome.nodes[1] = Node(1, NodeType.SENSE_EPF_CENTER, 0.0)
    if 2 not in genome.nodes:
        genome.nodes[2] = Node(2, NodeType.ACTUATOR, 0.0)
    
    return genome


def create_complex_brain():
    """创建一个较复杂的测试大脑 (用于没有保存结构时)"""
    from core.eoe.genome import OperatorGenome
    from core.eoe.node import Node, NodeType
    
    genome = OperatorGenome()
    
    # 输入层 - 能量场传感器
    genome.nodes[1] = Node(1, NodeType.SENSE_EPF_CENTER, 0.0)
    genome.nodes[2] = Node(2, NodeType.SENSE_EPF_GRAD_X, 0.0)
    genome.nodes[3] = Node(3, NodeType.SENSE_EPF_GRAD_Y, 0.0)
    
    # 隐藏层 - 多种算子
    genome.nodes[4] = Node(4, NodeType.ADD, 0.1)
    genome.nodes[5] = Node(5, NodeType.MULTIPLY, -0.1)
    genome.nodes[6] = Node(6, NodeType.DELAY, 0.0)
    
    # 输出层 - 执行器
    genome.nodes[10] = Node(10, NodeType.ACTUATOR, 0.0)
    
    # 连接
    connections = [
        (1, 4, 0.8), (2, 4, 0.5), (3, 4, 0.3),
        (1, 5, 0.6), (4, 5, -0.4),
        (4, 6, 0.7), (6, 10, 0.9),
        (5, 10, -0.5),
    ]
    
    for src, dst, w in connections:
        genome.add_edge(src, dst, w)
    
    return genome


def run_benchmark_with_trained_brain():
    """使用训练好的大脑运行基准测试"""
    print("="*60)
    print("使用真实大脑的基准测试")
    print("="*60)
    
    # 尝试加载训练好的结构
    try:
        brain = load_trained_genome(top_n=1)
        print("\n✓ 成功加载训练好的大脑\n")
    except Exception as e:
        print(f"\n⚠ 加载失败: {e}")
        print("使用模拟的复杂大脑\n")
        brain = create_complex_brain()
    
    # 创建Agent
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    
    # 替换为训练好的brain
    agent.genome = brain
    
    print(f"Brain结构: {len(brain.nodes)} nodes, {len(brain.edges)} connections")
    
    # 运行测试
    runner = BenchmarkRunner(verbose=True)
    
    # 测试Level 1-3
    results = []
    for level in range(1, 4):
        print(f"\n{'='*60}")
        print(f"Testing Level {level}")
        print(f"{'='*60}")
        
        level_results = runner.run_level(level, brain, start_pos=(10.0, 50.0))
        results.extend(level_results)
        
        for r in level_results:
            status = "✓" if r.success else "✗"
            print(f"  {status} {r.task_name}: {r.steps_taken} steps, fitness={r.fitness:.2f}")
    
    # 生成报告
    print(f"\n{'='*60}")
    print("生成报告")
    print(f"{'='*60}")
    
    generator = BenchmarkReportGenerator("benchmarks/results")
    report = generator.generate_report(results, title="Trained Brain Benchmark Report")
    
    print(report)
    
    return results


if __name__ == "__main__":
    run_benchmark_with_trained_brain()