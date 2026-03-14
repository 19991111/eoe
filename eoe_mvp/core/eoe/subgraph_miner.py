#!/usr/bin/env python3
"""
v14.0 子图挖掘模块 (Subgraph Miner)
===================================
运行时模式识别 + 基因封装

核心功能:
- 扫描Top 10% Elite Agent的大脑拓扑
- 挖掘频繁子图 (Frequent Subgraph Mining)
- 发现高度保守的结构并提取为SuperNode

算法: 简化版gSpan
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SubgraphPattern:
    """子图模式"""
    # 节点类型序列 (按DFS顺序)
    node_types: Tuple[int, ...]
    # 边类型序列 (按DFS顺序)
    edge_types: Tuple[int, ...]
    # 邻接矩阵 (压缩)
    adjacency: Tuple[Tuple[int, ...], ...]
    # 出现次数
    support: int = 0
    # 涉及的Agent索引
    agent_indices: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash((self.node_types, self.edge_types))


class SubgraphMiner:
    """
    频繁子图挖掘器

    使用简化版gSpan算法:
    1. 收集所有Agent的大脑拓扑
    2. 编码为规范形式 (canonical form)
    3. 统计出现频率
    4. 提取超过阈值的模式
    """

    def __init__(
        self,
        min_support: float = 0.002,  # 最小支持度 (0.2%) - 极低,只要>=2个
        min_size: int = 3,           # 最小节点数
        max_size: int = 8,           # 扩大范围
        device: str = 'cuda:0'
    ):
        self.min_support = min_support
        self.min_size = min_size
        self.max_size = max_size
        self.device = device

        # 已发现的模式 (避免重复)
        self.discovered_patterns: Set[SubgraphPattern] = set()

        # SuperNode计数器
        self.supernode_counter = 0

    def encode_brain(
        self,
        node_types: torch.Tensor,    # [N] 节点类型
        edge_sources: torch.Tensor,  # [E] 边源节点
        edge_targets: torch.Tensor,  # [E] 边目标节点
        edge_weights: torch.Tensor   # [E] 边权重 (可选)
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[int, ...]]]:
        """
        编码大脑为规范子图形式

        Returns:
            (node_types_tuple, edge_types_tuple, adjacency_tuple)
        """
        n_nodes = len(node_types)

        # 节点类型 (按索引排序保证唯一性)
        node_types_tuple = tuple(node_types.cpu().numpy().tolist())

        # 边类型 (权重量化)
        if edge_weights is not None:
            edge_types_tuple = tuple(
                (edge_weights.cpu().numpy() > 0).astype(int).tolist()
            )
        else:
            edge_types_tuple = tuple([1] * len(edge_sources))

        # 邻接矩阵 (压缩: 只存上三角)
        adj = [[0] * n_nodes for _ in range(n_nodes)]
        for s, t in zip(edge_sources.cpu().numpy(), edge_targets.cpu().numpy()):
            if s < n_nodes and t < n_nodes:
                adj[s][t] = 1
                adj[t][s] = 1

        adjacency_tuple = tuple(tuple(row) for row in adj)

        return node_types_tuple, edge_types_tuple, adjacency_tuple

    def mine(
        self,
        genomes: Dict[int, 'OperatorGenome'],
        alive_indices: torch.Tensor,
        top_k: int = 50
    ) -> List[SubgraphPattern]:
        """
        挖掘频繁子图

        参数:
            genomes: Agent基因组字典
            alive_indices: 活跃Agent索引
            top_k: 只扫描Top K个能量最高的Agent

        返回:
            发现的频繁模式列表
        """
        # 1. 收集Top K Agent
        n_alive = len(alive_indices)
        if n_alive == 0:
            return []

        # 转换为Python列表 (解决tensor key问题)
        alive_list = alive_indices.cpu().tolist()

        # 按能量排序，取Top K (如果没有energy属性，用节点数代替)
        energy_list = []
        for idx in alive_list:
            g = genomes[idx]
            if hasattr(g, 'total_energy'):
                e = g.total_energy
            elif hasattr(g, 'energy'):
                e = g.energy
            else:
                e = len(g.nodes)  # 用节点数作为后备
            energy_list.append(float(e))
        energies = torch.tensor(energy_list, device=self.device)

        _, top_indices = torch.topk(energies, min(top_k, n_alive))
        top_agent_indices = [alive_list[i] for i in top_indices.tolist()]

        # 2. 编码每个大脑
        encoded_brains = []
        unique_node_types = set()
        
        for agent_idx in top_agent_indices:
            genome = genomes.get(agent_idx)
            if genome is None or len(genome.nodes) < self.min_size:
                continue

            # 提取节点类型 (转换为整数)
            node_types = torch.tensor(
                [genome.nodes[nid].node_type.value for nid in sorted(genome.nodes.keys())],
                device=self.device
            )

            # 提取边
            edge_sources = []
            edge_targets = []
            edge_weights = []
            for edge in genome.edges:
                if not edge.get('enabled', True):
                    continue
                edge_sources.append(edge['source_id'])
                edge_targets.append(edge['target_id'])
                edge_weights.append(edge.get('weight', 1.0))

            if len(edge_sources) == 0:
                continue

            encoding = self.encode_brain(
                node_types,
                torch.tensor(edge_sources, device=self.device),
                torch.tensor(edge_targets, device=self.device),
                torch.tensor(edge_weights, device=self.device) if edge_weights else None
            )

            encoded_brains.append((agent_idx, encoding))
            
            # 调试: 收集唯一的节点类型
            unique_node_types.add(encoding[0])

        # 3. 统计频率 - 简化版：只关注节点类型序列
        node_type_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        node_type_agents: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

        for agent_idx, encoding in encoded_brains:
            node_types, edge_types, adjacency = encoding
            
            # 简化：直接使用整个节点类型序列作为模式
            # 忽略边细节，只关注节点类型
            nt_tuple = tuple(node_types)
            
            # 只统计长度在min_size到max_size之间的
            if self.min_size <= len(nt_tuple) <= self.max_size:
                node_type_counts[nt_tuple] += 1
                node_type_agents[nt_tuple].append(agent_idx)

        # 4. 转换为SubgraphPattern
        pattern_counts: Dict[SubgraphPattern, int] = {}
        pattern_agents: Dict[SubgraphPattern, List[int]] = {}
        
        # 使用固定最小支持数
        min_count = 5
        
        for nt_tuple, count in node_type_counts.items():
            if count >= min_count:
                pattern = SubgraphPattern(
                    node_types=nt_tuple,
                    edge_types=tuple([1] * (len(nt_tuple) - 1)),
                    adjacency=tuple([[0]*len(nt_tuple) for _ in range(len(nt_tuple))]),
                    support=count
                )
                pattern_counts[pattern] = count
                pattern_agents[pattern] = node_type_agents[nt_tuple]

        # 4. 筛选超过支持度阈值的模式 (使用固定的min_count=5)
        frequent_patterns = []
        for pattern, count in pattern_counts.items():
            if count >= 5:  # 固定阈值
                pattern.support = count
                pattern.agent_indices = pattern_agents[pattern]
                frequent_patterns.append(pattern)

        # 按支持度排序
        frequent_patterns.sort(key=lambda p: p.support, reverse=True)

        return frequent_patterns

    def _extract_subpattern(
        self,
        node_types: Tuple[int, ...],
        edge_types: Tuple[int, ...],
        adjacency: Tuple[Tuple[int, ...], ...],
        size: int
    ) -> Optional[SubgraphPattern]:
        """提取固定大小的子模式"""
        n_nodes = len(node_types)
        if n_nodes < size:
            return None
        
        # 简化: 取前size个节点
        sub_node_types = node_types[:size]
        # 转换为list以便修改
        sub_adj = [list(row) for row in adjacency[:size]]
        for i in range(size):
            sub_adj[i] = sub_adj[i][:size]
        sub_adj = [tuple(row) for row in sub_adj]

        # 边类型简化为只反映连通性
        n_edges = sum(1 for i in range(size) for j in range(i+1, size) if sub_adj[i][j])
        sub_edge_types = tuple([1] * n_edges)

        return SubgraphPattern(
            node_types=sub_node_types,
            edge_types=sub_edge_types,
            adjacency=tuple(sub_adj),
            support=0
        )

    def get_supernode_name(self) -> str:
        """生成新的SuperNode名称"""
        name = f"SUPERNODE_{self.supernode_counter}"
        self.supernode_counter += 1
        return name


def test_subgraph_miner():
    """测试子图挖掘"""
    print("🧪 测试子图挖掘")

    from core.eoe.genome import OperatorGenome
    from core.eoe.node import Node, NodeType

    # 创建一些测试基因组
    genomes = {}
    for i in range(20):
        g = OperatorGenome()
        # 添加节点
        node_types = [NodeType.SENSOR, NodeType.ADD, NodeType.ADD, NodeType.ACTUATOR]
        for j, nt in enumerate(node_types):
            g.add_node(Node(node_id=j, node_type=nt))
        # 添加边
        g.add_edge(0, 1, weight=1.0)
        g.add_edge(1, 2, weight=1.0)
        g.add_edge(2, 3, weight=1.0)
        g.energy = 100.0 - i  # 递减能量
        genomes[i] = g

    # 挖掘
    miner = SubgraphMiner(min_support=0.3, min_size=3, max_size=4)
    alive_indices = torch.arange(20)
    patterns = miner.mine(genomes, alive_indices, top_k=10)

    print(f"\n📊 发现 {len(patterns)} 个频繁模式")
    for p in patterns[:3]:
        print(f"  - 节点: {p.node_types}, 边: {len(p.edge_types)}, 支持度: {p.support}")

    print(f"\n✅ 子图挖掘测试完成")
    return True


if __name__ == "__main__":
    test_subgraph_miner()