"""
v16.0 风场 (Wind Field) - 构成性环境生态位测试

物理模型:
- 固定风向和风速
- 暴露在风中会受到持续伤害
- 墙壁可以阻挡风 (通过 ray-cast 检测遮挡)

用途:
- Phase 3 测试: 智能体学会"挡风墙"行为
- 观测"工具使用"现象级涌现
"""

import numpy as np
import torch
from typing import Optional, Tuple


class WindField:
    """
    环境风场 - 对智能体造成持续伤害
    
    物理模型:
    - direction: 风向 (弧度, 0=东)
    - base_speed: 基础风速
    - damage_rate: 每步伤害率
    - 伤害计算: 通过 ray-cast 检测是否有遮挡
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        direction: float = 0.0,        # 风向 (弧度)
        base_speed: float = 5.0,       # 基础风速
        damage_rate: float = 0.1,      # 每步伤害
        enabled: bool = True,
        resolution: float = 1.0        # 射线投射分辨率
    ):
        self.width = width
        self.height = height
        self.direction = direction
        self.base_speed = base_speed
        self.damage_rate = damage_rate
        self.enabled = enabled
        self.resolution = resolution
        
        # 预计算方向向量
        self.dir_x = np.cos(direction)
        self.dir_y = np.sin(direction)
    
    def get_damage(self, x: float, y: float, env) -> float:
        """
        计算智能体受到的风伤害
        
        Args:
            x, y: 智能体位置
            env: 环境对象 (用于 ray-cast)
            
        Returns:
            伤害值 (0 if 有遮挡)
        """
        if not self.enabled:
            return 0.0
        
        # 使用环境进行 ray-cast 检测
        if hasattr(env, 'ray_cast'):
            hit, dist = env.ray_cast(
                (x, y),
                self.direction,
                max_distance=max(self.width, self.height)
            )
            if hit:
                # 有遮挡，无伤害
                return 0.0
        
        # 无遮挡，受到伤害
        return self.damage_rate
    
    def get_damage_batch(self, positions: np.ndarray, env) -> np.ndarray:
        """
        批量计算风伤害 (NumPy)
        
        Args:
            positions: [N, 2] 位置数组
            env: 环境对象
            
        Returns:
            [N] 伤害数组
        """
        if not self.enabled:
            return np.zeros(len(positions))
        
        damages = np.zeros(len(positions))
        
        # 简单实现: 对每个位置检查是否有遮挡
        for i, (x, y) in enumerate(positions):
            damages[i] = self.get_damage(x, y, env)
        
        return damages


class WindFieldGPU:
    """
    GPU 加速风场 - 使用张量运算
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        direction: float = 0.0,
        base_speed: float = 5.0,
        damage_rate: float = 0.1,
        device: str = 'cuda:0',
        enabled: bool = True,
        resolution: float = 1.0
    ):
        self.width = width
        self.height = height
        self.direction = direction
        self.base_speed = base_speed
        self.damage_rate = damage_rate
        self.device = device
        self.enabled = enabled
        self.resolution = resolution
        
        # 预计算方向向量 (GPU tensor)
        self.dir_vec = torch.tensor(
            [np.cos(direction), np.sin(direction)],
            device=device, dtype=torch.float32
        )
    
    def get_damage_batch(self, positions: torch.Tensor, matter_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        批量计算风伤害 (GPU)
        
        Args:
            positions: [N, 2] 位置张量
            matter_grid: [1, 1, H, W] 物质网格 (可选)
            
        Returns:
            [N] 伤害张量
        """
        if not self.enabled:
            return torch.zeros(positions.shape[0], device=self.device)
        
        N = positions.shape[0]
        damages = torch.full((N,), self.damage_rate, device=self.device)
        
        if matter_grid is None:
            # 无物质网格，全部受伤
            return damages
        
        # 简化实现: 采样智能体位置是否有物质
        # 如果智能体位置有物质，认为它"在墙后"，无伤害
        # (简化版 ray-cast - 实际应该投射射线)
        
        grid_w = matter_grid.shape[3]
        grid_h = matter_grid.shape[2]
        
        gx = (positions[:, 0] / self.resolution).long() % grid_w
        gy = (positions[:, 1] / self.resolution).long() % grid_h
        
        # 检查智能体位置是否有物质 (简化: 智能体在墙里 = 受保护)
        in_wall = matter_grid[0, 0, gy, gx].bool()
        
        # 如果在墙里，无伤害
        damages[in_wall] = 0.0
        
        return damages
    
    def ray_cast_batch(
        self,
        positions: torch.Tensor,
        matter_grid: torch.Tensor,
        max_distance: float = 100.0
    ) -> torch.Tensor:
        """
        批量射线投射 - 检查每个位置沿风向是否有遮挡
        
        Args:
            positions: [N, 2] 位置张量
            matter_grid: [1, 1, H, W] 物质网格
            max_distance: 最大投射距离
            
        Returns:
            [N] 遮挡标记 (True = 有遮挡)
        """
        if matter_grid is None:
            return torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        
        N = positions.shape[0]
        grid_w = matter_grid.shape[3]
        grid_h = matter_grid.shape[2]
        
        # 步进采样
        step_size = 1.0  # 1单位步长
        n_steps = int(max_distance / step_size)
        
        occlusion = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        for step in range(n_steps):
            # 采样点位置
            sample_pos = positions + self.dir_vec.unsqueeze(0) * step * step_size
            
            # 转换为网格坐标
            gx = (sample_pos[:, 0] / self.resolution).long() % grid_w
            gy = (sample_pos[:, 1] / self.resolution).long() % grid_h
            
            # 检查是否有物质
            hit = matter_grid[0, 0, gy, gx].bool()
            occlusion = occlusion | hit
            
            # 如果全部有遮挡，提前结束
            if occlusion.all():
                break
        
        return occlusion
    
    def get_damage_with_raycast(
        self,
        positions: torch.Tensor,
        matter_grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用射线投射计算精确风伤害
        
        Args:
            positions: [N, 2] 位置张量
            matter_grid: [1, 1, H, W] 物质网格
            
        Returns:
            [N] 伤害张量
        """
        if not self.enabled:
            return torch.zeros(positions.shape[0], device=self.device)
        
        if matter_grid is None:
            # 无物质场，全部受伤
            return torch.full((positions.shape[0],), self.damage_rate, device=self.device)
        
        # 使用射线投射检测遮挡
        occlusion = self.ray_cast_batch(positions, matter_grid)
        
        # 有遮挡 = 无伤害
        damages = torch.where(
            occlusion,
            torch.tensor(0.0, device=self.device),
            torch.tensor(self.damage_rate, device=self.device)
        )
        
        return damages