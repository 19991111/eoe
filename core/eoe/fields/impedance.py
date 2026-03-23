"""
阻抗场 (KIF)
============
Kinetic Impedance Field - 模拟地形障碍

特性:
- Perlin-like噪声生成
- 静态障碍物
- GPU加速支持
"""

import torch
import numpy as np
from typing import Tuple
from .base import StaticField


class ImpedanceField(StaticField):
    """
    阻抗场 (KIF)
    
    使用多频率正弦波模拟Perlin噪声
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cpu',
        noise_scale: float = 1.0,
        obstacle_density: float = 0.15
    ):
        super().__init__(
            width=width, height=height,
            resolution=resolution, device=device
        )
        
        self.noise_scale = noise_scale
        self.obstacle_density = obstacle_density
        
        # 初始化场
        self._init_field()
    
    def _init_field(self):
        """初始化阻抗场 (多频率噪声)"""
        h, w = self.grid_height, self.grid_width
        
        # 使用GPU或CPU
        if self.device.startswith('cuda'):
            # GPU: 使用torch生成
            field = torch.zeros(h, w, device=self.device, dtype=torch.float32)
            
            # 多频率叠加
            for freq in [0.02, 0.05, 0.1, 0.2]:
                phase_x = torch.rand(1, device=self.device) * 2 * np.pi
                phase_y = torch.rand(1, device=self.device) * 2 * np.pi
                
                y_coords = torch.arange(h, device=self.device, dtype=torch.float32) * freq
                x_coords = torch.arange(w, device=self.device, dtype=torch.float32) * freq
                
                # 外积生成网格
                field += torch.sin(
                    y_coords.unsqueeze(1) + y_coords.unsqueeze(0) * 0 + phase_x
                ) * torch.sin(
                    x_coords.unsqueeze(0) + x_coords.unsqueeze(1) * 0 + phase_y
                )
        else:
            # CPU: 使用numpy更高效
            y_coords = np.arange(h)[:, np.newaxis]
            x_coords = np.arange(w)[np.newaxis, :]
            
            field = np.zeros((h, w), dtype=np.float32)
            
            for freq in [0.02, 0.05, 0.1, 0.2]:
                phase_x = np.random.rand() * 2 * np.pi
                phase_y = np.random.rand() * 2 * np.pi
                
                field += np.sin(y_coords * freq + phase_x) * np.sin(x_coords * freq + phase_y)
            
            field = torch.from_numpy(field)
        
        # 添加障碍物
        n_obstacles = int(w * h * self.obstacle_density)
        
        if self.device.startswith('cuda'):
            obstacle_x = torch.randint(0, w, (n_obstacles,), device=self.device)
            obstacle_y = torch.randint(0, h, (n_obstacles,), device=self.device)
            
            for ox, oy in zip(obstacle_x, obstacle_y):
                field[oy, ox] = 10.0  # 高阻抗
        else:
            obstacle_x = np.random.randint(0, w, n_obstacles)
            obstacle_y = np.random.randint(0, h, n_obstacles)
            
            field[obstacle_y, obstacle_x] = 10.0
        
        # 归一化到 [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        
        self.field = field
    
    def sample(self, x: float, y: float) -> float:
        """采样"""
        gx, gy = self.get_grid_coords(x, y)
        return self.field[gy, gx].item()
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度"""
        grad_y, grad_x = torch.gradient(self.field)
        return grad_x, grad_y
    
    def get_value_tensor(self, positions: torch.Tensor) -> torch.Tensor:
        """批量获取场值"""
        gx = (positions[:, 0] / self.resolution).long()
        gy = (positions[:, 1] / self.resolution).long()
        
        gx = torch.clamp(gx, 0, self.grid_width - 1)
        gy = torch.clamp(gy, 0, self.grid_height - 1)
        
        return self.field[gy, gx]


# 兼容旧版本
KineticImpedanceFieldGPU = ImpedanceField