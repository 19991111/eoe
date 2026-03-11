"""
EOE 工具函数模块
================
提供公共工具函数，减少代码重复
"""

import numpy as np
from typing import List, Tuple


def clip_weight(weight: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """权重裁剪"""
    return float(np.clip(weight, min_val, max_val))


def sigmoid(x: float) -> float:
    """Sigmoid激活"""
    return float(1.0 / (1.0 + np.exp(-x)))


def tanh_activation(x: float) -> float:
    """Tanh激活"""
    return float(np.tanh(x))


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """2D欧氏距离"""
    return float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def angle_diff(a: float, b: float) -> float:
    """角度差归一化到[-pi, pi]"""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def normalize_angle(theta: float) -> float:
    """角度归一化到[-pi, pi]"""
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def rotate_point(x: float, y: float, theta: float) -> Tuple[float, float]:
    """绕原点旋转点"""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return float(x * cos_t - y * sin_t), float(x * sin_t + y * cos_t)


def batch_clip(weights: List[float], min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
    """批量权重裁剪"""
    return np.clip(weights, min_val, max_val)


def compute_angle(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """计算从(from_x, from_y)到(to_x, to_y)的方向角"""
    return float(np.arctan2(to_y - from_y, to_x - from_x))


def gaussian(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """高斯函数"""
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def softmax(inputs: List[float], temperature: float = 1.0) -> List[float]:
    """Softmax归一化"""
    exp_inputs = np.exp(np.array(inputs) / temperature)
    return (exp_inputs / exp_inputs.sum()).tolist()


def topological_sort(nodes: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    """
    拓扑排序 (Kahn算法)
    
    参数:
        nodes: 节点ID列表
        edges: 边列表 [(源, 目标), ...]
    
    返回:
        排序后的节点ID列表
    """
    # 构建邻接表和入度
    in_degree = {n: 0 for n in nodes}
    adjacency = {n: [] for n in nodes}
    
    for src, dst in edges:
        if src in adjacency and dst in in_degree:
            adjacency[src].append(dst)
            in_degree[dst] += 1
    
    # BFS
    queue = [n for n in nodes if in_degree[n] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result


def check_cycle(nodes: List[int], edges: List[Tuple[int, int]], 
                allow_delays: bool = True) -> bool:
    """
    检测是否有环
    
    参数:
        nodes: 节点ID列表
        edges: 边列表
        allow_delays: 是否允许含DELAY节点的环
    
    返回:
        True表示有环
    """
    # 简化的DFS环检测
    visited = set()
    rec_stack = set()
    
    adj = {n: [] for n in nodes}
    for src, dst in edges:
        if src in adj:
            adj[src].append(dst)
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in nodes:
        if node not in visited:
            if dfs(node):
                return True
    return False