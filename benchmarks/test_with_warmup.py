#!/usr/bin/env python3
"""
Benchmark测试 - 带后天学习 (Warm-up)
===================================
测试演化brain在热身后的表现
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import json
from core.eoe.genome import OperatorGenome
from core.eoe.node import Node, NodeType
from benchmarks.benchmark_runner import BenchmarkRunner


def load_brain(path: str) -> OperatorGenome:
    """从JSON加载brain"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    genome = OperatorGenome()
    
    # 添加节点
    node_map = {}
    for node_data in data['nodes']:
        node = Node(node_data['id'], NodeType[node_data['type']])
        genome.add_node(node)
        node_map[node_data['id']] = node
    
    # 添加边
    for edge_data in data['edges']:
        if edge_data['enabled']:
            genome.add_edge(
                edge_data['source_id'],
                edge_data['target_id'],
                weight=edge_data.get('weight', 0.0),
                enabled=True
            )
    
    return genome


def main():
    print("=" * 60)
    print("Benchmark with Warm-up (Lifetime Learning)")
    print("=" * 60)
    
    # 加载brain
    brain_path = '/home/node/.openclaw/workspace/eoe_mvp/benchmarks/results/trained_brain.json'
    brain = load_brain(brain_path)
    
    print(f"\n加载brain: {brain_path}")
    print(f"  节点数: {len(brain.nodes)}")
    print(f"  边数: {len([e for e in brain.edges if e['enabled']])}")
    
    # 运行benchmark (Level 1-3)
    runner = BenchmarkRunner(verbose=True)
    
    print("\n" + "=" * 60)
    print("Level 1: T-Maze Straight")
    print("=" * 60)
    results_l1 = runner.run_level(1, brain)
    
    print("\n" + "=" * 60)
    print("Level 2: T-Maze Delayed")
    print("=" * 60)
    results_l2 = runner.run_level(2, brain)
    
    print("\n" + "=" * 60)
    print("Level 3: T-Maze Stigmergy")
    print("=" * 60)
    results_l3 = runner.run_level(3, brain)
    
    # 汇总结果
    all_results = results_l1 + results_l2 + results_l3
    success_count = sum(1 for r in all_results if r.success)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(all_results)} tasks")
    print(f"Success: {success_count}/{len(all_results)} ({100*success_count/len(all_results):.1f}%)")
    
    for r in all_results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.task_name}: {r.steps_taken} steps, fitness={r.fitness:.2f}")


if __name__ == "__main__":
    main()