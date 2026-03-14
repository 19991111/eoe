#!/usr/bin/env python3
"""
v14.0 SuperNode 注册表 (SuperNode Registry)
============================================
运行时基因封装 + 代谢成本优惠

核心功能:
- 动态注册新发现的SuperNode
- 计算代谢成本优惠
- 压缩/解压Agent大脑

经济学:
- 如果一个3节点子图被冻结为SuperNode
- 它的代谢成本 = 原组件成本 × 0.7 (7折!)
- 这就是演化棘轮的动力!
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class SuperNodeSpec:
    """SuperNode规范"""
    id: int
    name: str
    # 节点类型序列
    node_types: Tuple[int, ...]
    # 边连接序列 (from, to)
    connections: List[Tuple[int, int]]
    # 原始代谢成本
    original_cost: float
    # 冻结后成本 (7折)
    frozen_cost: float
    # 发现的步数
    discovered_at_step: int
    # 使用此SuperNode的Agent数
    usage_count: int = 0


class SuperNodeRegistry:
    """
    SuperNode动态注册表
    
    核心API:
    - register(): 发现新模式后注册为SuperNode
    - compress(): 将Agent大脑中的子图替换为SuperNode引用
    - get_cost(): 获取给定大脑结构的代谢成本
    """
    
    def __init__(
        self,
        cost_discount: float = 0.7,  # 7折优惠
        max_supernodes: int = 1000,  # 取消上限
        device: str = 'cuda:0'
    ):
        self.cost_discount = cost_discount
        self.max_supernodes = max_supernodes
        self.device = device
        
        # 注册表: name -> SuperNodeSpec
        self.supernodes: OrderedDict[str, SuperNodeSpec] = OrderedDict()
        
        # 节点类型成本 (与brain_thermodynamics.py同步)
        self.node_costs = {
            0: 0.001,   # CONSTANT
            1: 0.001,   # ADD
            2: 0.002,   # MULTIPLY
            3: 0.001,   # THRESHOLD
            4: 0.005,   # DELAY
            5: 0.01,    # SENSOR
            6: 0.02,    # ACTUATOR
        }
        
        self.edge_cost = 0.0005
        
        # 统计
        self.total_compressions = 0
    
    def register(
        self,
        pattern: 'SubgraphPattern',
        discovered_at_step: int = 0
    ) -> Optional[SuperNodeSpec]:
        """
        注册新的SuperNode
        
        参数:
            pattern: 挖掘到的子图模式
            discovered_at_step: 发现时的步数
            
        返回:
            SuperNodeSpec 或 None (如果已达上限)
        """
        # 检查是否已满
        if len(self.supernodes) >= self.max_supernodes:
            return None
        
        # 检查是否已存在
        pattern_key = self._pattern_key(pattern)
        for spec in self.supernodes.values():
            if self._pattern_key_from_spec(spec) == pattern_key:
                return None  # 已存在
        
        # 计算原始成本
        original_cost = self._calculate_original_cost(pattern)
        
        # 计算冻结后成本 (7折!)
        frozen_cost = original_cost * self.cost_discount
        
        # 创建规范
        spec = SuperNodeSpec(
            id=len(self.supernodes),
            name=f"SUPERNODE_{len(self.supernodes)}",
            node_types=pattern.node_types,
            connections=self._extract_connections(pattern),
            original_cost=original_cost,
            frozen_cost=frozen_cost,
            discovered_at_step=discovered_at_step,
            usage_count=0
        )
        
        self.supernodes[spec.name] = spec
        
        print(f"  🧬 注册新SuperNode: {spec.name}")
        print(f"     节点: {pattern.node_types}")
        print(f"     原始成本: {original_cost:.4f} → 冻结成本: {frozen_cost:.4f} (省{100*(1-self.cost_discount):.0f}%)")
        
        return spec
    
    def _calculate_original_cost(self, pattern: 'SubgraphPattern') -> float:
        """计算原始代谢成本"""
        cost = 0.0
        for node_type in pattern.node_types:
            cost += self.node_costs.get(node_type, 0.001)
        cost += len(pattern.edge_types) * self.edge_cost
        return cost
    
    def _extract_connections(self, pattern: 'SubgraphPattern') -> List[Tuple[int, int]]:
        """从邻接矩阵提取连接"""
        connections = []
        adj = pattern.adjacency
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                if adj[i][j] > 0:
                    connections.append((i, j))
        return connections
    
    def _pattern_key(self, pattern: 'SubgraphPattern') -> str:
        """生成模式键"""
        return str(pattern.node_types) + str(pattern.adjacency)
    
    def _pattern_key_from_spec(self, spec: SuperNodeSpec) -> str:
        """从规范生成模式键"""
        return str(spec.node_types)
    
    def compress(
        self,
        node_types: List[int],
        edges: List[Tuple[int, int]]
    ) -> Tuple[List[Any], float]:
        """
        压缩大脑结构
        
        将匹配的子图替换为SuperNode引用
        
        参数:
            node_types: 节点类型列表
            edges: 边列表
            
        返回:
            (压缩后的节点列表, 总代谢成本)
        """
        compressed_nodes = []
        total_cost = 0.0
        
        # 检查是否匹配任何已注册的SuperNode
        matched_supernode = None
        for spec in self.supernodes.values():
            if self._match_pattern(node_types, edges, spec):
                matched_supernode = spec
                break
        
        if matched_supernode:
            # 使用SuperNode
            compressed_nodes.append(matched_supernode.name)
            total_cost = matched_supernode.frozen_cost
            matched_supernode.usage_count += 1
            self.total_compressions += 1
        else:
            # 常规成本
            for nt in node_types:
                compressed_nodes.append(nt)
                total_cost += self.node_costs.get(nt, 0.001)
        
        return compressed_nodes, total_cost
    
    def _match_pattern(
        self,
        node_types: List[int],
        edges: List[Tuple[int, int]],
        spec: SuperNodeSpec
    ) -> bool:
        """检查是否匹配模式"""
        # 简化匹配: 检查节点类型序列
        if len(node_types) < len(spec.node_types):
            return False
        
        # 检查节点类型匹配
        nt_tuple = tuple(node_types[:len(spec.node_types)])
        if nt_tuple != spec.node_types:
            return False
        
        return True
    
    def get_cost(self, node_count: int, supernode_names: List[str]) -> float:
        """
        计算给定大脑结构的代谢成本
        
        参数:
            node_count: 总节点数
            supernode_names: 使用的SuperNode名称列表
            
        返回:
            代谢成本
        """
        cost = 0.0
        
        # SuperNode成本
        for name in supernode_names:
            spec = self.supernodes.get(name)
            if spec:
                cost += spec.frozen_cost
                node_count -= len(spec.node_types)  # 减去已计算的节点
        
        # 剩余节点成本
        remaining_nodes = max(0, node_count)
        for i in range(remaining_nodes):
            cost += self.node_costs.get(0, 0.001)  # 假设CONSTANT
        
        return cost
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_original = sum(s.original_cost for s in self.supernodes.values())
        total_frozen = sum(s.frozen_cost for s in self.supernodes.values())
        
        return {
            'n_supernodes': len(self.supernodes),
            'total_savings': total_original - total_frozen,
            'usage_count': sum(s.usage_count for s in self.supernodes.values()),
            'compressions': self.total_compressions,
        }
    
    def __repr__(self):
        return f"SuperNodeRegistry({len(self.supernodes)} nodes, discount={self.cost_discount})"


def test_supernode_registry():
    """测试SuperNode注册表"""
    print("🧪 测试SuperNode注册表")
    
    from core.eoe.subgraph_miner import SubgraphPattern
    
    registry = SuperNodeRegistry(cost_discount=0.7)
    
    # 模拟注册一个模式
    pattern = SubgraphPattern(
        node_types=(5, 3, 6),  # SENSOR -> THRESHOLD -> ACTUATOR
        edge_types=(1, 1),
        adjacency=((0,1,0), (1,0,1), (0,1,0)),
        support=10
    )
    
    spec = registry.register(pattern, discovered_at_step=1000)
    print(f"\n📊 注册结果: {spec.name if spec else 'Failed'}")
    
    # 测试压缩
    compressed, cost = registry.compress([5, 3, 6, 0, 1], [(0,1), (1,2)])
    print(f"   压缩后: {compressed}")
    print(f"   成本: {cost:.4f}")
    
    # 统计
    stats = registry.get_stats()
    print(f"\n📈 统计: {stats}")
    
    print(f"\n✅ SuperNode注册表测试完成")
    return True


if __name__ == "__main__":
    test_supernode_registry()