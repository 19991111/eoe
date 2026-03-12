"""
能量场 (EPF)
============
Energy Field - 为Agent提供能量

特性:
- 多个能量源持续注入能量
- 能量扩散和衰减
- GPU加速支持

物理模型:
E(x,y,t+1) = E(x,y,t) * decay + sources
"""

import torch
import numpy as np
from typing import Tuple, Optional
from .base import SourceField


class EnergyField(SourceField):
    """
    能量场 (EPF)
    
    Attributes:
        field: 能量网格 [H, W]
        sources: 能量源 [N, 4] (x, y, strength, active)
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cpu',
        n_sources: int = 3,
        source_strength: float = 50.0,
        decay_rate: float = 0.99
    ):
        super().__init__(
            width=width, height=height,
            resolution=resolution, device=device,
            n_sources=n_sources
        )
        
        self.source_strength = source_strength
        self.decay_rate = decay_rate
        
        # 初始化场
        self._init_field()
        
        # 初始化能量源
        self._init_sources()
    
    def _init_field(self):
        """初始化场"""
        if self.device == 'cpu':
            self.field = torch.zeros(
                self.grid_height, self.grid_width,
                dtype=torch.float32
            )
        else:
            self.field = torch.zeros(
                self.grid_height, self.grid_width,
                device=self.device, dtype=torch.float32
            )
    
    def _init_sources(self):
        """初始化能量源"""
        if self.device == 'cpu':
            self.sources = torch.zeros(self.n_sources, 4)
        else:
            self.sources = torch.zeros(self.n_sources, 4, device=self.device)
        
        # 随机位置和强度
        for i in range(self.n_sources):
            self.sources[i, 0] = torch.rand(1) * self.width
            self.sources[i, 1] = torch.rand(1) * self.height
            self.sources[i, 2] = self.source_strength * (0.5 + torch.rand(1))
            self.sources[i, 3] = 1.0  # active
    
    def step(self):
        """单步更新: 衰减 + 能量源注入"""
        # 能量衰减
        self.field *= self.decay_rate
        
        # 能量源注入
        self._inject_sources()
    
    def _inject_sources(self):
        """向场注入能量"""
        # 转换源位置到网格坐标
        source_x = (self.sources[:, 0] / self.resolution).long()
        source_y = (self.sources[:, 1] / self.resolution).long()
        
        # 边界裁剪
        source_x = torch.clamp(source_x, 0, self.grid_width - 1)
        source_y = torch.clamp(source_y, 0, self.grid_height - 1)
        
        # 批量注入
        for i in range(self.n_sources):
            if self.sources[i, 3] > 0:  # active
                self.field[source_y[i], source_x[i]] += self.sources[i, 2]
    
    def sample(self, x: float, y: float) -> float:
        """采样位置的能量值"""
        gx, gy = self.get_grid_coords(x, y)
        return self.field[gy, gx].item()
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度"""
        grad_y, grad_x = torch.gradient(self.field)
        return grad_x, grad_y
    
    def get_value_tensor(self, positions: torch.Tensor) -> torch.Tensor:
        """
        批量获取场值 (GPU优化)
        
        Args:
            positions: [N, 2] 坐标张量
        Returns:
            [N] 场值张量
        """
        gx = (positions[:, 0] / self.resolution).long()
        gy = (positions[:, 1] / self.resolution).long()
        
        gx = torch.clamp(gx, 0, self.grid_width - 1)
        gy = torch.clamp(gy, 0, self.grid_height - 1)
        
        return self.field[gy, gx]
    
    def total_energy(self) -> float:
        """总能量"""
        return torch.sum(self.field).item()


# 兼容旧版本
EnergyFieldGPU = EnergyField