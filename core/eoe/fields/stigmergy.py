"""
压痕场 (ISF)
============
Stigmergy Field - Agent间通过环境间接通信

特性:
- 信号扩散 (Laplacian卷积)
- 衰减
- 软饱和
- GPU加速支持
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from .base import DiffusiveField


class StigmergyField(DiffusiveField):
    """
    压痕场 (ISF)
    
    通过Laplacian卷积实现扩散
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cpu',
        diffusion_rate: float = 0.1,
        decay_rate: float = 0.98,
        max_value: float = 100.0
    ):
        super().__init__(
            width=width, height=height,
            resolution=resolution, device=device,
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate
        )
        
        self.max_value = max_value
        
        # 初始化场
        self._init_field()
        
        # 创建扩散核
        self._init_kernel()
    
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
    
    def _init_kernel(self):
        """创建Laplacian扩散核"""
        # 5-point Laplacian
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=self.device, dtype=torch.float32)
        
        self.diffusion_kernel = kernel * self.diffusion_rate
    
    def step(self):
        """单步更新: 扩散 + 衰减 + 饱和"""
        self.diffuse()
        self.field *= self.decay_rate
        self.field = torch.clamp(self.field, 0, self.max_value)
    
    def diffuse(self):
        """执行扩散"""
        # 添加batch维度 [H, W] -> [1, 1, H, W]
        field_4d = self.field.unsqueeze(0).unsqueeze(0)
        
        # 卷积扩散
        diffused = F.conv2d(
            field_4d,
            self.diffusion_kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )
        
        # 更新场
        self.field = self.field + diffused.squeeze()
    
    def deposit(self, x: float, y: float, amount: float):
        """注入信号"""
        gx, gy = self.get_grid_coords(x, y)
        self.field[gy, gx] += amount
    
    def deposit_batch(self, positions: torch.Tensor, amounts: torch.Tensor):
        """
        批量注入信号
        
        Args:
            positions: [N, 2] 坐标
            amounts: [N] 信号量
        """
        gx = (positions[:, 0] / self.resolution).long()
        gy = (positions[:, 1] / self.resolution).long()
        
        gx = torch.clamp(gx, 0, self.grid_width - 1)
        gy = torch.clamp(gy, 0, self.grid_height - 1)
        
        # 批量添加 (需要循环，因为一个位置可能多个注入)
        for i in range(len(positions)):
            self.field[gy[i], gx[i]] += amounts[i]
    
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
StigmergyFieldGPU = StigmergyField