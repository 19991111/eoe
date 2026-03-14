#!/usr/bin/env python3
"""
v14.0 感知场映射 (Perception Field Mapping)
============================================
核心思想：将感官抽象为"向量内积" + 具身空间采样

- 物理环境 → 多通道张量 [1, C, H, W]
- 传感器 → 可演化受体密钥 [C] + 空间偏移 [2]
- 感知 → 环境张量与受体密钥的内积

演化涌现的形态:
- "触角" (dx > 0): 向前伸展，探测前方食物
- "尾巴" (dx < 0): 向后伸展，探测追兵
- "侧翼" (dy ≠ 0): 左右伸展，探测周围环境
"""

import torch
import torch.nn.functional as F
from typing import Optional


def sample_embodied_fields(
    env_tensor: torch.Tensor,
    agents_pos: torch.Tensor,       # [N, 2] Agent 中心位置
    sensor_offsets: torch.Tensor,   # [N, S, 2] 已旋转的传感器世界偏移
    env_width: float,
    env_height: float
) -> torch.Tensor:
    """
    🚀 具身空间采样 - 核心创新！
    
    每个 Agent 的每个传感器采样身体不同位置的场值
    
    Args:
        env_tensor: [1, C, H, W] 环境张量 (已归一化)
        agents_pos: [N, 2] Agent 中心坐标 (x, y)
        sensor_offsets: [N, S, 2] 世界坐标系中的传感器偏移 (dx, dy)
        env_width: 环境宽度
        env_height: 环境高度
        
    Returns:
        local_fields: [N, S, C] 每个 Agent 每个传感器的局部场值
    """
    N, S, _ = sensor_offsets.shape  # N=Agent数, S=传感器数
    
    # ===== 步骤1: 计算所有传感器的世界坐标 =====
    # agents_pos.unsqueeze(1): [N, 2] → [N, 1, 2]
    # 广播相加: [N, 1, 2] + [N, S, 2] = [N, S, 2]
    absolute_sensor_pos = agents_pos.unsqueeze(1) + sensor_offsets
    
    # 环形世界: 坐标wrap
    absolute_sensor_pos[..., 0] = absolute_sensor_pos[..., 0] % env_width
    absolute_sensor_pos[..., 1] = absolute_sensor_pos[..., 1] % env_height
    
    # ===== 步骤2: 归一化到 grid_sample 要求的 [-1, 1] =====
    norm_x = (absolute_sensor_pos[..., 0] / env_width) * 2.0 - 1.0
    norm_y = (absolute_sensor_pos[..., 1] / env_height) * 2.0 - 1.0
    
    # ===== 步骤3: 构建 4D 采样网格 =====
    # grid: [1, N, S, 2] (Batch=1, Height=N, Width=S, Coords=xy)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)
    
    # ===== 步骤4: 🚀 魔法时刻！一次 GPU 采样所有传感器！=====
    # env_tensor: [1, C, H, W]
    # grid:       [1, N, S, 2]
    # 输出:       [1, C, N, S]
    sampled = F.grid_sample(
        env_tensor, 
        grid, 
        mode='bilinear', 
        align_corners=True,
        padding_mode='border'  # 边界外的值设为边界值
    )
    
    # ===== 步骤5: 调整形状为 [N, S, C] =====
    # sampled: [1, C, N, S] → [C, N, S] → [N, S, C]
    local_fields = sampled.squeeze(0).permute(1, 2, 0)
    
    return local_fields  # [N, S, C]


def compute_sensor_activations(
    local_fields: torch.Tensor,
    sensor_keys: torch.Tensor
) -> torch.Tensor:
    """
    计算传感器激活 - 核心内积运算
    
    智能体"感觉"到的 = 局部物理现实 · 受体密钥
    
    Args:
        local_fields: [N, S, C] 每个 Agent 每个传感器位置的物理场
        sensor_keys: [N, S, C] (N个Agent, 每个S个传感器, 每个密钥长度C)
        
    Returns:
        activations: [N, S] 每个 Agent 的每个传感器的激活值
    """
    # 逐元素相乘后求和: [N, S, C] * [N, S, C] → [N, S]
    # 等价于 local_fields · sensor_keys (内积)
    activations = torch.sum(local_fields * sensor_keys, dim=2)
    
    return activations  # [N, S]


class PerceptionEngine:
    """
    感知引擎 - 具身智能版本
    ========================
    统一管理环境通道化、空间采样和感知计算
    """
    
    def __init__(self, n_channels: int = 4, device: str = 'cuda:0'):
        self.n_channels = n_channels
        self.device = device
    
    def compute_sensor_world_offsets(
        self,
        agent_thetas: torch.Tensor,    # [N] 朝向角
        sensor_relative_offsets: torch.Tensor  # [N, S, 2] 身体相对偏移
    ) -> torch.Tensor:
        """
        将相对偏移转换为世界坐标偏移
        
        Args:
            agent_thetas: [N] Agent 朝向角 (弧度)
            sensor_relative_offsets: [N, S, 2] 身体相对偏移
            
        Returns:
            world_offsets: [N, S, 2] 世界坐标偏移
        """
        N, S, _ = sensor_relative_offsets.shape
        
        # 旋转矩阵 [N, 2, 2]
        cos_t = torch.cos(agent_thetas).unsqueeze(1)  # [N, 1]
        sin_t = torch.sin(agent_thetas).unsqueeze(1)  # [N, 1]
        
        # 构建旋转矩阵
        rotation = torch.zeros(N, 2, 2, device=agent_thetas.device)
        rotation[:, 0, 0] = cos_t.squeeze(1)
        rotation[:, 0, 1] = -sin_t.squeeze(1)
        rotation[:, 1, 0] = sin_t.squeeze(1)
        rotation[:, 1, 1] = cos_t.squeeze(1)
        
        # 矩阵乘法: [N, S, 2] × [N, 2, 2]^T = [N, S, 2]
        world_offsets = torch.bmm(
            sensor_relative_offsets,
            rotation.transpose(1, 2)
        )
        
        return world_offsets
    
    def get_perception(
        self,
        env: 'EnvironmentGPU',
        agents: 'BatchedAgents',
        sensor_keys: torch.Tensor,         # [N, S, C] 受体密钥
        sensor_offsets: torch.Tensor,      # [N, S, 2] 身体相对偏移
        use_embodied_sampling: bool = True # 🚀 开关
    ) -> torch.Tensor:
        """
        获取所有 Agent 的感知输入
        
        Args:
            env: GPU 环境
            agents: 批量 Agent
            sensor_keys: 所有传感器密钥 [N, S, C]
            sensor_offsets: 传感器相对偏移 [N, S, 2]
            use_embodied_sampling: 是否启用具身采样
            
        Returns:
            Tensor [N, S] 感知激活值
        """
        # 1. 获取环境张量 (已归一化!)
        env_tensor = env.get_env_tensor()  # [1, C, H, W]
        
        if use_embodied_sampling and sensor_offsets is not None:
            # 🚀 路径A: 具身空间采样 (推荐!)
            
            # 计算世界坐标偏移
            world_offsets = self.compute_sensor_world_offsets(
                agents.state.thetas,
                sensor_offsets
            )  # [N, S, 2]
            
            # 批量采样
            local_fields = sample_embodied_fields(
                env_tensor,
                agents.state.positions,
                world_offsets,
                agents.env_width,
                agents.env_height
            )  # [N, S, C]
        else:
            # 路径B: 退化模式 (所有传感器在同一位置)
            # 只采样 Agent 中心位置
            center_pos = agents.state.positions  # [N, 2]
            
            # 扩展为所有传感器 [N, S, 2]
            N = center_pos.shape[0]
            S = sensor_keys.shape[1]
            center_pos_expanded = center_pos.unsqueeze(1).expand(N, S, 2)
            
            # 使用零偏移采样中心
            zero_offsets = torch.zeros(N, S, 2, device=center_pos.device)
            
            local_fields = sample_embodied_fields(
                env_tensor,
                center_pos_expanded.reshape(N * S, 2),  # [N*S, 2]
                zero_offsets.reshape(N * S, 2),
                agents.env_width,
                agents.env_height
            ).reshape(N, S, -1)  # [N, S, C]
        
        # 2. 计算内积感知
        activations = compute_sensor_activations(
            local_fields,
            sensor_keys
        )  # [N, S]
        
        return activations


def build_sensor_matrices(
    genomes: list,
    device: str = 'cuda:0'
) -> tuple:
    """
    从基因组构建传感器密钥和偏移矩阵
    
    Args:
        genomes: Agent 基因组列表
        device: 计算设备
        
    Returns:
        sensor_keys: [N, S, C] 受体密钥
        sensor_offsets: [N, S, 2] 空间偏移
    """
    from core.eoe.node import NodeType
    
    # 统计每个 Agent 的传感器数量
    sensor_counts = []
    for g in genomes:
        sensor_nodes = [n for n in g.nodes.values() if n.node_type == NodeType.SENSOR]
        sensor_counts.append(len(sensor_nodes))
    
    max_sensors = max(sensor_counts) if sensor_counts else 0
    
    if max_sensors == 0:
        return None, None
    
    N = len(genomes)
    C = 4  # ENERGY, IMPEDANCE, STRESS, STIGMERGY
    
    # 构建密钥矩阵 [N, S, C]
    sensor_keys = torch.zeros(N, max_sensors, C, device=device, dtype=torch.float32)
    
    # 构建偏移矩阵 [N, S, 2]
    sensor_offsets = torch.zeros(N, max_sensors, 2, device=device, dtype=torch.float32)
    
    # 填充矩阵
    for i, g in enumerate(genomes):
        sensor_nodes = [n for n in g.nodes.values() if n.node_type == NodeType.SENSOR]
        for j, node in enumerate(sensor_nodes):
            if node.receptor_key is not None:
                sensor_keys[i, j] = node.receptor_key
            if node.spatial_offset is not None:
                sensor_offsets[i, j] = node.spatial_offset
    
    return sensor_keys, sensor_offsets