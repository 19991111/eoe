#!/usr/bin/env python3
"""
脑结构加载器
============
从保存的复杂结构初始化Agent种群

使用方式:
    python scripts/load_brain_structures.py --structures FILE --population N

或者在实验脚本中导入:
    from scripts.load_brain_structures import BrainStructureLoader
    
    loader = BrainStructureLoader('saved_structures.json')
    genomes = loader.load_top_n(n=50)
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from core.eoe.genome import OperatorGenome


class BrainStructureLoader:
    """脑结构加载器"""
    
    def __init__(self, filepath: str):
        """加载结构文件"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 支持两种格式
        structures = data.get('structures', {})
        if isinstance(structures, dict):
            self.structures = list(structures.values())
        else:
            self.structures = structures
        
        # 按复杂度排序
        self.structures = sorted(
            self.structures, 
            key=lambda s: s.get('complexity_score', 0), 
            reverse=True
        )
        
        print(f"📂 加载 {len(self.structures)} 个脑结构")
    
    def get_top_n(self, n: int = 10) -> List[Dict]:
        """获取Top N最复杂结构"""
        return self.structures[:n]
    
    def get_by_id(self, structure_id: str) -> Dict:
        """根据ID获取结构"""
        for s in self.structures:
            if s['structure_id'] == structure_id:
                return s
        return None
    
    def genome_from_structure(self, struct: Dict) -> OperatorGenome:
        """从结构创建基因组"""
        from core.eoe.node import Node, NodeType
        
        genome = OperatorGenome()
        
        # 添加节点
        node_map = {}
        next_node_id = 0
        
        for i, node_type in enumerate(struct['nodes']):
            node = Node(node_id=next_node_id, node_type=NodeType(node_type))
            genome.add_node(node)
            node_map[i] = next_node_id
            next_node_id += 1
        
        # 添加边
        for src, tgt, weight in struct['edges']:
            if src in node_map and tgt in node_map:
                genome.add_edge(
                    node_map[src], 
                    node_map[tgt], 
                    weight
                )
        
        return genome
    
    def load_top_n_genomes(self, n: int = 10) -> List[OperatorGenome]:
        """加载Top N结构为基因组"""
        top_structs = self.get_top_n(n)
        genomes = []
        
        for s in top_structs:
            try:
                g = self.genome_from_structure(s)
                genomes.append(g)
                print(f"  ✅ {s['structure_id']}: {s['complexity_score']:.2f}分, "
                      f"{len(s['nodes'])}节点/{len(s['edges'])}边")
            except Exception as e:
                print(f"  ❌ {s['structure_id']}: {e}")
        
        return genomes
    
    def create_population(
        self, 
        n_agents: int,
        n_unique: int = 10,
        duplicate_factor: int = 1
    ) -> Dict[int, OperatorGenome]:
        """
        创建种群
        
        Args:
            n_agents: 总Agent数量
            n_unique: 使用多少种独特结构
            duplicate_factor: 每种结构复制次数
        
        Returns:
            {agent_id: genome}
        """
        # 加载独特结构
        unique_genomes = self.load_top_n_genomes(n_unique)
        
        if not unique_genomes:
            raise ValueError("没有可用的结构")
        
        # 创建种群
        population = {}
        agent_id = 0
        
        # 循环填充
        while agent_id < n_agents:
            for genome in unique_genomes:
                if agent_id >= n_agents:
                    break
                
                # 复制基因组
                from copy import deepcopy
                population[agent_id] = deepcopy(genome)
                agent_id += 1
        
        print(f"\n🎯 创建种群: {len(population)} 个Agent, "
              f"使用 {len(unique_genomes)} 种结构")
        
        return population
    
    def get_statistics(self) -> Dict:
        """获取结构统计"""
        if not self.structures:
            return {}
        
        complexity_scores = [s.get('complexity_score', 0) for s in self.structures]
        node_counts = [len(s.get('nodes', [])) for s in self.structures]
        edge_counts = [len(s.get('edges', [])) for s in self.structures]
        
        return {
            'total_structures': len(self.structures),
            'avg_complexity': np.mean(complexity_scores),
            'max_complexity': max(complexity_scores),
            'avg_nodes': np.mean(node_counts),
            'max_nodes': max(node_counts),
            'avg_edges': np.mean(edge_counts),
            'max_edges': max(edge_counts),
        }


def create_experiment_with_loaded_brains(
    structure_file: str,
    n_agents: int = 200,
    output_file: str = "loaded_population.json"
):
    """创建带有加载脑结构的实验初始化文件"""
    loader = BrainStructureLoader(structure_file)
    
    # 打印统计
    stats = loader.get_statistics()
    print(f"\n📊 结构统计:")
    print(f"   总结构数: {stats['total_structures']}")
    print(f"   平均复杂度: {stats['avg_complexity']:.2f}")
    print(f"   最大复杂度: {stats['max_complexity']:.2f}")
    print(f"   平均节点: {stats['avg_nodes']:.1f}")
    print(f"   最大节点: {stats['max_nodes']}")
    print(f"   平均边: {stats['avg_edges']:.1f}")
    print(f"   最大边: {stats['max_edges']}")
    
    # 创建种群
    population = loader.create_population(n_agents=n_agents, n_unique=20)
    
    # 保存为可重用格式
    population_data = {
        'n_agents': len(population),
        'structures_used': len(set(
            s['structure_id'] for s in loader.get_top_n(20)
        )),
        'source_file': structure_file,
        'genomes': [
            {
                'id': aid,
                'nodes': [
                    {'node_id': i, 'node_type': n.node_type.value}
                    for i, n in enumerate(g.nodes.values())
                ] if hasattr(g.nodes, 'values') else [],
                'edges': g.edges
            }
            for aid, g in list(population.items())[:50]  # 只保存前50个示例
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(population_data, f, indent=2)
    
    print(f"\n💾 种群配置已保存: {output_file}")
    
    return population


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='加载脑结构')
    parser.add_argument('--structures', 
                       default='experiments/v15_cognitive_premium/saved_structures/complexity_step30000.json',
                       help='结构文件')
    parser.add_argument('--population', type=int, default=200, help='种群大小')
    parser.add_argument('--top-n', type=int, default=20, help='使用的独特结构数')
    parser.add_argument('--output', default='loaded_population.json', help='输出文件')
    args = parser.parse_args()
    
    # 加载并显示统计
    loader = BrainStructureLoader(args.structures)
    stats = loader.get_statistics()
    
    print(f"\n📊 结构统计:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # 创建种群
    population = loader.create_population(
        n_agents=args.population, 
        n_unique=args.top_n
    )
    
    print(f"\n✅ 完成! 种群大小: {len(population)}")