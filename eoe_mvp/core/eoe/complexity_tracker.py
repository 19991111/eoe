"""
复杂结构追踪器 - 保存演化过程中涌现的高复杂度结构
用于后续测试和分析
"""

import torch
import json
import os
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BrainStructure:
    """脑结构快照"""
    structure_id: str
    nodes: List[int]  # 节点类型列表
    edges: List[Tuple[int, int, float]]  # (源, 目标, 权重)
    complexity_score: float
    first_seen_step: int
    population_count: int  # 当前拥有此结构的个体数
    avg_fitness: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @staticmethod
    def from_genome(genome, step: int, fitness: float = 0.0) -> 'BrainStructure':
        """从基因组创建结构快照"""
        nodes = list(genome.nodes.values()) if hasattr(genome.nodes, 'values') else genome.nodes
        node_types = [n.node_type.value if hasattr(n, 'node_type') else n.get('node_type', 0) for n in nodes]
        
        edges = []
        for e in genome.edges:
            if isinstance(e, dict):
                src = e.get('source_id', e.get('source'))
                tgt = e.get('target_id', e.get('target'))
                w = e.get('weight', 0.5)
            else:
                src = e.source if hasattr(e, 'source') else e.source_id
                tgt = e.target if hasattr(e, 'target') else e.target_id
                w = e.weight if hasattr(e, 'weight') else 0.5
            edges.append((src, tgt, w))
        
        # 计算复杂度分数
        complexity = BrainStructure._calculate_complexity(node_types, edges)
        
        # 生成唯一ID
        structure_id = BrainStructure._generate_id(node_types, edges)
        
        return BrainStructure(
            structure_id=structure_id,
            nodes=node_types,
            edges=edges,
            complexity_score=complexity,
            first_seen_step=step,
            population_count=1,
            avg_fitness=fitness
        )
    
    @staticmethod
    def _calculate_complexity(nodes: List[int], edges: List[Tuple]) -> float:
        """计算复杂度分数"""
        if not nodes:
            return 0.0
        
        # 基础分数: 节点数
        node_score = len(nodes)
        
        # 连接分数: 边数的对数
        edge_score = np.log1p(len(edges)) * 2
        
        # 结构分数: 唯一节点类型数
        type_score = len(set(nodes)) * 0.5
        
        # 循环检测: 简单图中的环路
        cycle_score = BrainStructure._detect_cycles(nodes, edges) * 3
        
        return node_score + edge_score + type_score + cycle_score
    
    @staticmethod
    def _detect_cycles(nodes: List[int], edges: List[Tuple]) -> int:
        """简单环路检测"""
        if len(edges) < 3:
            return 0
        
        # 构建邻接表
        adj = defaultdict(set)
        for src, tgt, _ in edges:
            adj[src].add(tgt)
        
        # DFS检测环路
        visited = set()
        rec_stack = set()
        cycles = 0
        
        def dfs(node, path):
            nonlocal cycles
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    if dfs(neighbor, path + [neighbor]):
                        return True
                elif neighbor in rec_stack:
                    cycles += 1
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in adj:
            if node not in visited:
                dfs(node, [node])
        
        return min(cycles, 5)  # 最多5分
    
    @staticmethod
    def _generate_id(nodes: List[int], edges: List[Tuple]) -> str:
        """生成唯一结构ID"""
        # 排序边作为结构签名
        edge_signature = tuple(sorted((src, tgt) for src, tgt, _ in edges))
        node_signature = tuple(sorted(nodes))
        return f"struct_{hash((node_signature, edge_signature)) % 100000:05d}"


class ComplexityTracker:
    """复杂结构追踪器"""
    
    def __init__(
        self,
        save_dir: str = "saved_structures",
        top_k: int = 50,           # 保存Top K最复杂结构
        min_complexity: float = 3.0,  # 最小复杂度阈值
        save_interval: int = 1000,    # 保存间隔(步)
    ):
        self.save_dir = save_dir
        self.top_k = top_k
        self.min_complexity = min_complexity
        self.save_interval = save_interval
        
        # 结构存储
        self.structures: Dict[str, BrainStructure] = {}
        self.structure_counts: Dict[str, int] = defaultdict(int)  # 结构出现次数
        self.complexity_rank: List[str] = []  # 按复杂度排序的结构ID
        
        # 统计
        self.total_snapshots = 0
        self.last_save_step = 0
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, genomes: List, fitnesses: List[float], step: int):
        """更新结构数据库"""
        self.total_snapshots += len(genomes)
        
        # 当前步的结构计数
        current_counts: Dict[str, int] = defaultdict(int)
        
        for genome, fitness in zip(genomes, fitnesses):
            try:
                struct = BrainStructure.from_genome(genome, step, fitness)
                
                # 复杂度过滤
                if struct.complexity_score < self.min_complexity:
                    continue
                
                sid = struct.structure_id
                
                if sid in self.structures:
                    # 更新已有结构
                    existing = self.structures[sid]
                    existing.population_count += 1
                    # 移动平均更新适应度
                    existing.avg_fitness = (existing.avg_fitness + fitness) / 2
                else:
                    # 新结构
                    self.structures[sid] = struct
                    current_counts[sid] = 1
                
                current_counts[sid] += 1
                
            except Exception as e:
                continue
        
        # 更新出现次数
        for sid, count in current_counts.items():
            self.structure_counts[sid] += count
        
        # 重新排序
        self._rerank()
        
        # 定期保存
        if step - self.last_save_step >= self.save_interval:
            self.save(step)
            self.last_save_step = step
    
    def _rerank(self):
        """按复杂度重新排序"""
        self.complexity_rank = sorted(
            self.structures.keys(),
            key=lambda sid: self.structures[sid].complexity_score,
            reverse=True
        )
    
    def get_top_complex(self, n: int = 10) -> List[BrainStructure]:
        """获取最复杂的N个结构"""
        return [self.structures[sid] for sid in self.complexity_rank[:n]]
    
    def get_structure_by_id(self, sid: str) -> BrainStructure:
        """根据ID获取结构"""
        return self.structures.get(sid)
    
    def get_percentile(self, percentile: float) -> List[BrainStructure]:
        """获取复杂度高于给定百分位的所有结构"""
        if not self.complexity_rank:
            return []
        
        idx = int(len(self.complexity_rank) * (1 - percentile / 100))
        idx = max(0, min(idx, len(self.complexity_rank) - 1))
        
        return [self.structures[sid] for sid in self.complexity_rank[idx:]]
    
    def save(self, step: int):
        """保存到文件"""
        # 保存完整数据库
        data = {
            'step': step,
            'total_snapshots': self.total_snapshots,
            'unique_structures': len(self.structures),
            'structures': {
                sid: struct.to_dict() 
                for sid, struct in self.structures.items()
            },
            'complexity_rank': self.complexity_rank,
            'top_k': [s.to_dict() for s in self.get_top_complex(self.top_k)]
        }
        
        filepath = os.path.join(self.save_dir, f"complexity_step{step}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # 保存Top K简表
        top_k_data = {
            'step': step,
            'structures': [s.to_dict() for s in self.get_top_complex(self.top_k)]
        }
        top_k_path = os.path.join(self.save_dir, "top_k_structures.json")
        with open(top_k_path, 'w') as f:
            json.dump(top_k_data, f, indent=2)
        
        print(f"  💾 保存复杂结构: {len(self.structures)}个, Top{self.top_k}已保存")
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.structures = {
            sid: BrainStructure(**struct_data)
            for sid, struct_data in data['structures'].items()
        }
        self.complexity_rank = data.get('complexity_rank', [])
        self.total_snapshots = data.get('total_snapshots', 0)
        
        print(f"  📂 加载复杂结构: {len(self.structures)}个")
    
    def print_summary(self):
        """打印摘要"""
        print(f"\n📊 复杂结构统计:")
        print(f"   总快照数: {self.total_snapshots}")
        print(f"   独特结构: {len(self.structures)}")
        
        if self.complexity_rank:
            top = self.get_top_complex(5)
            print(f"   Top 5 复杂度:")
            for i, s in enumerate(top, 1):
                print(f"     {i}. {s.structure_id}: {s.complexity_score:.2f}分, "
                      f"{len(s.nodes)}节点/{len(s.edges)}边, 出现{self.structure_counts[s.structure_id]}次")