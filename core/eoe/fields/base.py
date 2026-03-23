"""
物理场抽象基类
==============
定义所有场的统一接口

v13.0 统一场物理系统:
- EPF (Energy Field): 能量场
- KIF (Kinetic Impedance Field): 阻抗场  
- ISF (Stigmergy Field): 压痕场
- ESF (Stress Field): 应力场
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Field(ABC):
    """
    物理场抽象基类
    
    所有场实现必须继承此类并实现:
    - step(): 单步更新
    - sample(): 采样位置的值
    - compute_gradient(): 计算梯度
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cpu'
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        # 计算网格大小
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 场数据张量 [H, W]
        self.field: Optional[torch.Tensor] = None
    
    @abstractmethod
    def step(self):
        """执行单步更新"""
        pass
    
    @abstractmethod
    def sample(self, x: float, y: float) -> float:
        """
        采样位置的值
        
        Args:
            x, y: 世界坐标
        Returns:
            场值
        """
        pass
    
    @abstractmethod
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算梯度场
        
        Returns:
            (grad_x, grad_y): 梯度张量
        """
        pass
    
    def get_grid_coords(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return gx, gy
    
    @property
    def shape(self) -> Tuple[int, int]:
        """场形状"""
        return (self.grid_height, self.grid_width)
    
    def to(self, device: str):
        """移动到设备"""
        self.device = device
        if self.field is not None:
            self.field = self.field.to(device)
        return self


class SourceField(Field):
    """带能量源的场 (如 EPF)"""
    
    def __init__(self, *args, n_sources: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sources = n_sources
        self.sources = None  # 能量源位置和强度
    
    def update_sources(self):
        """更新能量源 (可选实现)"""
        pass


class DiffusiveField(Field):
    """带扩散的场 (如 ISF)"""
    
    def __init__(self, *args, diffusion_rate: float = 0.1, decay_rate: float = 0.98, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.diffusion_kernel = None
    
    @abstractmethod
    def diffuse(self):
        """执行扩散"""
        pass


class StaticField(Field):
    """静态场 (如 KIF) - 初始化后不变化"""
    
    def step(self):
        """静态场不做更新"""
        pass