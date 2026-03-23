"""
v13.0 GPU 加速环境引擎
======================
基于 PyTorch 的统一场物理引擎 - 100% VRAM 常驻

特性:
- 所有场数据常驻 GPU 显存
- F.conv2d 实现扩散计算
- torch.gradient 实现梯度计算  
- F.grid_sample 实现批量传感器采样

使用方式:
    env = EnvironmentGPU(width=100, height=100)
    for _ in range(1500):
        env.step()  # 毫秒级步进
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class EnergyFieldGPU:
    """
    GPU 加速能量场 (EPF) - 动态可枯竭版本
    ======================================
    特性:
    - 脉冲式能量注入
    - 能量源可枯竭
    - 枯竭后随机迁移到新位置
    - 迫使Agent演化空间迁徙能力
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        n_sources: int = 5,           # 增加源数量
        source_strength: float = 400.0, # v16.28: 大幅提高能量密度
        source_capacity: float = 500.0, # 增加总容量
        decay_rate: float = 0.995,    # v16.28: 降低衰减，收支平衡
        respawn_threshold: float = 0.15, # 更早枯竭(15%)
        seasonal_multiplier: float = 1.0,  # 季节能量倍率
        seasons_enabled: bool = True,
        season_length: int = 3000,
        winter_multiplier: float = 0.15,
        summer_multiplier: float = 1.8,
        drought_intensity: float = 0.08
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        self.seasonal_multiplier = seasonal_multiplier
        
        # 季节参数
        self.seasons_enabled = seasons_enabled
        self.season_length = season_length
        self.winter_multiplier = winter_multiplier
        self.summer_multiplier = summer_multiplier
        self.drought_intensity = drought_intensity
        
        # 计算网格大小
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # GPU 张量 - 显存常驻 [1, 1, H, W]
        self.field = torch.zeros(
            1, 1, self.grid_height, self.grid_width,
            device=device, dtype=torch.float32
        )
        
        # 能量源 (GPU) - 扩展为6列: [x, y, strength, active, capacity, max_capacity]
        self.sources = torch.zeros(n_sources, 6, device=device)
        self.n_sources = n_sources
        self.source_strength = source_strength
        self.source_capacity = source_capacity
        self.decay_rate = decay_rate
        self.respawn_threshold = respawn_threshold
        self.step_count = 0
        
        # 初始化源
        self._init_sources()
        
        # ========== v16.16: GPU 渲染卷积核 (P0 优化) ==========
        # 预计算高斯衰减核，用于 F.conv2d 批量渲染
        self._render_radius = 5
        kernel_size = 2 * self._render_radius + 1
        
        y, x = torch.meshgrid(
            torch.arange(-self._render_radius, self._render_radius + 1, device=self.device, dtype=torch.float32),
            torch.arange(-self._render_radius, self._render_radius + 1, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        distance = torch.sqrt(x**2 + y**2)
        # 高斯衰减: exp(-dist^2 / 8)，与原实现一致
        self._render_kernel = torch.exp(-distance**2 / 8.0).view(1, 1, kernel_size, kernel_size)
        
        # 预计算一个更大的核用于 _inject_energy_pulse (半径10)
        self._inject_radius = 10
        inject_kernel_size = 2 * self._inject_radius + 1
        y_inj, x_inj = torch.meshgrid(
            torch.arange(-self._inject_radius, self._inject_radius + 1, device=self.device, dtype=torch.float32),
            torch.arange(-self._inject_radius, self._inject_radius + 1, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        dist_inj = torch.sqrt(x_inj**2 + y_inj**2)
        # 简化的2D高斯: 1 / (1 + dist^2 * 0.1)
        self._inject_kernel = (1.0 / (1.0 + dist_inj**2 * 0.1)).view(1, 1, inject_kernel_size, inject_kernel_size)
        # ========================================================
    
    def _render_field(self):
        """
        v16.16: GPU 向量化渲染 (P0 优化)
        
        使用 F.conv2d 批量计算所有能量源的光环辐射，
        替代原有的 Python 嵌套循环 + .item() 同步调用。
        """
        # 获取可见且有剩余容量的源
        active_mask = (self.sources[:, 3] > 0) & (self.sources[:, 4] > 0)
        if not active_mask.any():
            return
        
        active_sources = self.sources[active_mask]
        src_x = active_sources[:, 0]
        src_y = active_sources[:, 1]
        
        # 坐标转换为网格索引
        gx = (src_x / self.resolution).long().clamp(0, self.grid_width - 1)
        gy = (src_y / self.resolution).long().clamp(0, self.grid_height - 1)
        
        # 创建脉冲网格
        B, C, H, W = self.field.shape
        impulses = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.float32)
        
        # 批量写入
        flat_indices = gy * self.grid_width + gx
        impulses.view(-1).scatter_add_(0, flat_indices, torch.ones_like(src_x))
        
        # 单次卷积
        rendered = F.conv2d(impulses, self._render_kernel, padding=self._render_radius)
        
        self.field += rendered * 0.05
    
    def get_seasonal_multiplier(self) -> float:
        """获取当前季节的能量倍率 (含干旱期)"""
        if not self.seasons_enabled:
            return 1.0
        
        # 四季循环 + 干旱期
        season_cycle = self.season_length * 4
        phase = (self.step_count % season_cycle) / season_cycle
        
        # 0.0-0.25: 春季 (恢复)
        # 0.25-0.5: 夏季 (繁荣)
        # 0.5-0.75: 秋季 (衰退)
        # 0.75-1.0: 冬季/干旱 (最艰难)
        
        if phase < 0.25:
            t = phase / 0.25
            multiplier = self.winter_multiplier + t * (1.0 - self.winter_multiplier)
        elif phase < 0.5:
            t = (phase - 0.25) / 0.25
            multiplier = 1.0 + t * (self.summer_multiplier - 1.0)
        elif phase < 0.75:
            t = (phase - 0.5) / 0.25
            multiplier = self.summer_multiplier - t * (self.summer_multiplier - self.winter_multiplier)
        else:
            t = (phase - 0.75) / 0.25
            if hasattr(self, 'drought_intensity'):
                multiplier = self.winter_multiplier * (1 - t * (1 - self.drought_intensity))
            else:
                multiplier = self.winter_multiplier
        
        return multiplier
    
    def _init_sources(self):
        """初始化能量源"""
        for i in range(self.n_sources):
            self._spawn_source(i)
    
    def _spawn_source(self, idx: int):
        """在随机位置生成新能量源"""
        # 随机位置 (避开边缘)
        self.sources[idx, 0] = torch.rand(1, device=self.device) * (self.width - 10) + 5
        self.sources[idx, 1] = torch.rand(1, device=self.device) * (self.height - 10) + 5
        # 脉冲强度 (随机化)
        self.sources[idx, 2] = self.source_strength * (0.5 + torch.rand(1, device=self.device))
        # 激活状态
        self.sources[idx, 3] = 1.0
        # 当前容量和最大容量
        self.sources[idx, 4] = self.source_capacity * (0.8 + torch.rand(1, device=self.device) * 0.4)
        self.sources[idx, 5] = self.sources[idx, 4].clone()
    
    def step(self):
        """单步更新"""
        self.step_count += 1
        
        # 0. 动态季节计算 (使用已有的get_seasonal_multiplier)
        if self.seasons_enabled:
            self.seasonal_multiplier = self.get_seasonal_multiplier()
        
        # 1. 能量自然衰减
        self.field *= self.decay_rate
        
        # 2. 脉冲式能量注入 (每10步一个脉冲)
        if self.step_count % 10 == 0:
            self._inject_energy_pulse()
        
        # 3. 检查并处理枯竭的源
        self._check_and_respawn()
    
    def _inject_energy_pulse(self):
        """
        v16.16: GPU 向量化能量注入 (P0 优化)
        
        使用 F.conv2d 批量计算所有能量源的注入，
        替代原有的 Python 嵌套循环 + .item() 同步调用。
        """
        # 季节调整后的注入量
        seasonal_strength = self.seasonal_multiplier
        
        # 获取活跃且有剩余容量的源
        active_mask = (self.sources[:, 3] > 0) & (self.sources[:, 4] > 0)
        if not active_mask.any():
            return
        
        active_sources = self.sources[active_mask]
        
        # 计算每个源的注入量（受季节调整，不超过剩余容量）
        base_amounts = active_sources[:, 2]  # 脉冲强度
        remaining = active_sources[:, 4]     # 剩余容量
        inject_amounts = (base_amounts * seasonal_strength).clamp(max=remaining)
        
        # 坐标转换为网格索引
        src_x = active_sources[:, 0]
        src_y = active_sources[:, 1]
        gx = ((src_x / self.resolution).long() % self.grid_width).clamp(0, self.grid_width - 1)
        gy = ((src_y / self.resolution).long() % self.grid_height).clamp(0, self.grid_height - 1)
        
        # 创建脉冲网格
        B, C, H, W = self.field.shape
        impulses = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.float32)
        
        # 批量写入脉冲信号（每个位置一个脉冲，强度=注入量）
        flat_indices = gy * self.grid_width + gx
        impulses.view(-1).scatter_add_(0, flat_indices, inject_amounts)
        
        # 单次卷积完成高斯扩散
        rendered = F.conv2d(impulses, self._inject_kernel, padding=self._inject_radius)
        
        # 注入到场
        self.field += rendered * 0.05
        
        # 更新剩余容量（向量化）
        self.sources[active_mask, 4] -= inject_amounts
        
        # 确保非负
        self.sources[:, 4] = self.sources[:, 4].clamp(min=0)
    
    def _check_and_respawn(self):
        """检查能量源是否枯竭，必要时迁移 (向量化版本)"""
        # 计算所有源的枯竭阈值
        max_capacities = self.sources[:, 5]
        min_capacities = max_capacities * self.respawn_threshold
        
        # 找出需要重生的源
        remaining = self.sources[:, 4]
        need_respawn = remaining <= min_capacities
        
        # 向量化处理：批量重生
        if need_respawn.any():
            n_respawn = need_respawn.sum()
            
            # 批量生成新位置
            new_x = torch.rand(n_respawn, device=self.device) * (self.width - 10) + 5
            new_y = torch.rand(n_respawn, device=self.device) * (self.height - 10) + 5
            
            # 更新位置
            respawn_indices = torch.where(need_respawn)[0]
            self.sources[respawn_indices, 0] = new_x
            self.sources[respawn_indices, 1] = new_y
            
            # 重置容量
            new_capacity = self.source_capacity * (0.8 + torch.rand(n_respawn, device=self.device) * 0.4)
            self.sources[respawn_indices, 4] = new_capacity
            self.sources[respawn_indices, 5] = new_capacity
    
    def get_source_info(self):
        """获取当前能量源状态 (用于调试)"""
        info = []
        for i in range(self.n_sources):
            info.append({
                'x': self.sources[i, 0].item(),
                'y': self.sources[i, 1].item(),
                'strength': self.sources[i, 2].item(),
                'active': self.sources[i, 3].item(),
                'capacity': self.sources[i, 4].item(),
                'max_capacity': self.sources[i, 5].item()
            })
        return info
    
    def extract_energy(self, positions: torch.Tensor, amounts: torch.Tensor):
        """
        从场中提取能量 (Agent吸取) - 向量化版本
        
        Args:
            positions: [N, 2] 位置
            amounts: [N] 吸取量
        """
        if positions.shape[0] == 0:
            return
        
        gx = (positions[:, 0] / self.resolution).long().clamp(0, self.grid_width - 1)
        gy = (positions[:, 1] / self.resolution).long().clamp(0, self.grid_height - 1)
        
        # 批量获取当前位置的能量值
        field_flat = self.field.view(-1)
        flat_idx = gy * self.grid_width + gx
        current_energy = field_flat[flat_idx]
        
        # 计算提取量 (不超过当前能量，也不超过请求量)
        valid_mask = (amounts > 0) & (current_energy > 0)
        if not valid_mask.any():
            return
        
        valid_idx = torch.where(valid_mask)[0]
        extract_amounts = torch.min(current_energy[valid_mask], amounts[valid_mask])
        
        # 批量更新场
        valid_flat_idx = flat_idx[valid_mask]
        field_flat[valid_flat_idx] -= extract_amounts
    
    def sample(self, x: float, y: float) -> float:
        """采样位置的能量值 (CPU 调用时)"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return self.field[0, 0, gy, gx].item()
    
    def sample_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """批量采样位置的能量值"""
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        return self.field[0, 0, gy, gx]
    
    def consume_batch(self, positions: torch.Tensor, amounts: torch.Tensor) -> torch.Tensor:
        """批量消耗能量 (从场中扣除)
        
        Returns:
            [N] actual energy consumed per agent
        """
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        
        # 确保索引在范围内
        valid = (gx >= 0) & (gx < self.grid_width) & (gy >= 0) & (gy < self.grid_height)
        
        # 计算实际消耗量 (不能超过场中可用能量)
        actual_consumed = torch.zeros_like(amounts)
        
        if valid.any():
            gx_valid = gx[valid]
            gy_valid = gy[valid]
            amounts_valid = amounts[valid]
            
            # 获取当前位置的场值
            field_values = self.field[0, 0, gy_valid, gx_valid]
            
            # 实际消耗 = min(请求量, 可用量)
            actual_valid = torch.min(amounts_valid, field_values)
            
            # 从场中扣除能量
            self.field[0, 0, gy_valid, gx_valid] = field_values - actual_valid
            
            # 填充结果
            valid_indices = torch.where(valid)[0]
            actual_consumed[valid_indices] = actual_valid
        
        return actual_consumed
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度 - GPU 加速"""
        # torch.gradient 返回 (grad_y, grad_x) - 注意顺序
        grad_y, grad_x = torch.gradient(self.field[0, 0])
        return grad_x.unsqueeze(0).unsqueeze(0), grad_y.unsqueeze(0).unsqueeze(0)


class KineticImpedanceFieldGPU:
    """GPU 加速阻抗场 (KIF) - 包含迷宫墙壁"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        noise_scale: float = 1.0,
        obstacle_density: float = 0.15,
        wall_density: float = 0.0,
        wall_strength: float = 10.0
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.wall_strength = wall_strength
        
        # GPU 张量 - 使用 Perlin-like 噪声初始化
        self.field = self._generate_impedance_field(
            self.grid_width, self.grid_height, noise_scale, obstacle_density, device
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 添加迷宫墙壁
        if wall_density > 0:
            self._generate_maze_walls(wall_density)
        
        # 预计算梯度
        self._grad_x = None
        self._grad_y = None
    
    def _generate_maze_walls(self, density: float):
        """生成迷宫墙壁 (高阻抗区域)"""
        h, w = self.grid_height, self.grid_width
        
        # 随机起点和方向生成墙壁
        n_walls = int(h * w * density / 20)  # 约密度/20的墙壁
        
        for _ in range(n_walls):
            # 随机起点
            start_x = torch.randint(5, w-5, (1,)).item()
            start_y = torch.randint(5, h-5, (1,)).item()
            
            # 随机方向 (0=水平, 1=垂直)
            direction = torch.randint(0, 2, (1,)).item()
            length = torch.randint(5, 15, (1,)).item()
            
            if direction == 0:  # 水平墙
                end_x = min(start_x + length, w - 3)
                self.field[0, 0, start_y, start_x:end_x] = self.wall_strength
            else:  # 垂直墙
                end_y = min(start_y + length, h - 3)
                self.field[0, 0, start_y:end_y, start_x] = self.wall_strength
    
    def _generate_impedance_field(
        self, w: int, h: int, noise_scale: float, 
        density: float, device: str
    ) -> torch.Tensor:
        """生成阻抗场 (多频率正弦波叠加)"""
        # 多频率叠加模拟 Perlin 噪声
        field = torch.zeros(h, w, device=device)
        
        # 基础噪声
        for freq in [0.02, 0.05, 0.1, 0.2]:
            phase_x = torch.rand(1, device=device) * 2 * np.pi
            phase_y = torch.rand(1, device=device) * 2 * np.pi
            
            y_coords = torch.arange(h, device=device, dtype=torch.float32) * freq
            x_coords = torch.arange(w, device=device, dtype=torch.float32) * freq
            
            # 外积生成网格
            field += torch.sin(
                y_coords.unsqueeze(1) + y_coords.unsqueeze(0) * 0 + phase_x
            ) * torch.sin(
                x_coords.unsqueeze(0) + x_coords.unsqueeze(1) * 0 + phase_y
            )
        
        # 添加障碍物
        n_obstacles = int(w * h * density)
        obstacle_x = torch.randint(0, w, (n_obstacles,), device=device)
        obstacle_y = torch.randint(0, h, (n_obstacles,), device=device)
        
        for ox, oy in zip(obstacle_x, obstacle_y):
            field[oy, ox] = 10.0  # 高阻抗障碍
        
        # 归一化到 [0, 1]
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)
        
        return field
    
    def step(self):
        """KIF 在演化中通常不变化 (静态障碍场)"""
        pass
    
    def sample(self, x: float, y: float) -> float:
        """采样"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return self.field[0, 0, gy, gx].item()
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度"""
        grad_y, grad_x = torch.gradient(self.field[0, 0])
        return grad_x.unsqueeze(0).unsqueeze(0), grad_y.unsqueeze(0).unsqueeze(0)


class StigmergyFieldGPU:
    """GPU 加速压痕场 (ISF) - 支持扩散"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        diffusion_rate: float = 0.1,
        decay_rate: float = 0.98
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # GPU 张量 [1, 1, H, W]
        self.field = torch.zeros(
            1, 1, self.grid_height, self.grid_width,
            device=device, dtype=torch.float32
        )
        
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        
        # 扩散核 (GPU) - Laplacian 卷积核
        self._diffusion_kernel = self._create_laplacian_kernel(device)
    
    def _create_laplacian_kernel(self, device: str) -> torch.Tensor:
        """创建 Laplacian 扩散核"""
        # 5-point Laplacian
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel * self.diffusion_rate
    
    def step(self, matter_grid: Optional[torch.Tensor] = None):
        """
        单步扩散
        
        v16.0 增强: MatterGrid 遮挡 (防止量子隧穿)
        - 必须在卷积前和后都应用掩码
        """
        # v16.0: 应用前置掩码
        if matter_grid is not None:
            # matter_grid: [1, 1, H, W], 0=墙, 1=空
            mask = (matter_grid == 0).float()
            masked_field = self.field * mask
        else:
            masked_field = self.field
        
        # 卷积扩散 (GPU 加速)
        diffused = F.conv2d(
            masked_field, 
            self._diffusion_kernel, 
            padding=1
        )
        
        # 更新场
        new_field = self.field + diffused
        
        # v16.0: 应用后置掩码
        if matter_grid is not None:
            new_field = new_field * mask
        
        self.field = new_field
        
        # 衰减
        self.field *= self.decay_rate
        
        # 软饱和 (防止无限增长)
        self.field = torch.clamp(self.field, 0, 100.0)
    
    def deposit(self, x: float, y: float, amount: float):
        """注入信号 (单点)"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        self.field[0, 0, gy, gx] += amount
    
    def deposit_batch(self, positions: torch.Tensor, amounts: torch.Tensor):
        """
        批量注入信号 (GPU加速)
        
        Args:
            positions: [N, 2] x, y 坐标 (GPU张量)
            amounts: [N] 注入量 (GPU张量)
        """
        # 计算网格坐标
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        
        # 使用索引批量更新 (GPU上)
        self.field[0, 0, gy, gx] += amounts
    
    def sample(self, x: float, y: float) -> float:
        """采样"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return self.field[0, 0, gy, gx].item()
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度"""
        grad_y, grad_x = torch.gradient(self.field[0, 0])
        return grad_x.unsqueeze(0).unsqueeze(0), grad_y.unsqueeze(0).unsqueeze(0)


class DangerFieldGPU:
    """
    GPU 加速危险场 (Danger Field)
    ==============================
    Agent 攻击时写入伤害值，其他 Agent 踩到时扣血
    
    特性：
    - 写入：Channel 4 (ATTACK) 向场中写入瞬时伤害
    - 读取：每步 Agent 读取所在网格的危险值并扣血
    - 衰减：伤害值快速衰减 (模拟血迹消失)
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        decay_rate: float = 0.8  # 快速衰减
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # GPU 张量 [1, 1, H, W]
        self.field = torch.zeros(
            1, 1, self.grid_height, self.grid_width,
            device=device, dtype=torch.float32
        )
        
        self.decay_rate = decay_rate
    
    def attack_batch(
        self,
        positions: torch.Tensor,      # [N, 2] Agent 位置
        attack_strength: torch.Tensor,  # [N] 攻击强度
        offsets: torch.Tensor = None    # [N, 2] 攻击偏移 (可选)
    ):
        """
        批量攻击：向危险场写入伤害值
        
        O(N) 操作替代 O(N²) 的 Agent 间检测
        """
        if attack_strength is None or attack_strength.sum() == 0:
            return
        
        # 计算网格坐标
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        
        # 散点叠加 (GPU 加速)
        # 使用索引计算避免 scatter_add 警告
        valid = (gx >= 0) & (gx < self.grid_width) & (gy >= 0) & (gy < self.grid_height)
        if valid.any():
            self.field[0, 0, gy[valid], gx[valid]] += attack_strength[valid]
    
    def sample_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """批量读取危险值"""
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        return self.field[0, 0, gy, gx]
    
    def step(self):
        """单步衰减"""
        self.field *= self.decay_rate


class EnvironmentGPU:
    """
    GPU 加速环境引擎
    =================
    100% VRAM 常驻 - 所有数据保持在 GPU 显存中
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        energy_field_enabled: bool = True,
        impedance_field_enabled: bool = True,
        stigmergy_field_enabled: bool = True,
        danger_field_enabled: bool = True,
        seasons_enabled: bool = True,
        season_length: int = 3000,
        winter_multiplier: float = 0.15,
        summer_multiplier: float = 1.8,
        drought_intensity: float = 0.08,
        # v16.0: 构成性物质场
        matter_grid_enabled: bool = False,
        matter_resolution: float = 1.0,
        # v16.0: 风场
        wind_field_enabled: bool = False,
        wind_direction: float = 0.0,
        wind_damage_rate: float = 0.1,
        # v16.1: Flickering Energy Field
        flickering_energy_enabled: bool = False,
        flickering_period: int = 25,
        flickering_invisible_moves: int = 75,
        flickering_speed: float = 0.5,
        flickering_n_sources: int = 50,  # v16.31
        flickering_source_strength: float = 200.0,  # v16.31
        # v17.5: 冰河世纪协议
        ice_age_enabled: bool = False,
        ice_age_start_step: int = 2000,
        energy_dynamic_enabled: bool = False,
        energy_move_interval: int = 3,
        energy_jump_prob: float = 0.4,
        energy_jump_dist: float = 15.0,
        kif_storm_enabled: bool = False,
        kif_storm_count: int = 5,
        kif_storm_intensity: float = 800.0,
        kif_storm_move_speed: float = 1.0,
        kif_storm_radius: float = 15.0
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        # 季节参数
        self.seasons_enabled = seasons_enabled
        self.season_length = season_length
        self.winter_multiplier = winter_multiplier
        self.summer_multiplier = summer_multiplier
        self.drought_intensity = drought_intensity
        
        print(f"[EnvironmentGPU] 初始化 GPU 环境 {width}x{height} on {device}")
        
        # 场初始化
        self.energy_field_enabled = energy_field_enabled
        self.impedance_field_enabled = impedance_field_enabled
        self.stigmergy_field_enabled = stigmergy_field_enabled
        self.danger_field_enabled = danger_field_enabled
        
        if energy_field_enabled:
            self.energy_field = EnergyFieldGPU(
                width, height, resolution, device,
                seasons_enabled=seasons_enabled,
                season_length=season_length,
                winter_multiplier=winter_multiplier,
                summer_multiplier=summer_multiplier,
                drought_intensity=drought_intensity
            )
            print(f"  ✅ EPF: {self.energy_field.field.shape}")
        
        if impedance_field_enabled:
            self.impedance_field = KineticImpedanceFieldGPU(
                width, height, resolution, device
            )
            print(f"  ✅ KIF: {self.impedance_field.field.shape}")
        
        if stigmergy_field_enabled:
            self.stigmergy_field = StigmergyFieldGPU(
                width, height, resolution, device
            )
            print(f"  ✅ ISF: {self.stigmergy_field.field.shape}")
        
        if danger_field_enabled:
            self.danger_field = DangerFieldGPU(
                width, height, resolution, device
            )
            print(f"  ✅ DANGER: {self.danger_field.field.shape}")
        
        # v16.1: Flickering Energy Field
        self.flickering_energy_enabled = flickering_energy_enabled
        if flickering_energy_enabled:
            self.flickering_energy_field = FlickeringEnergyFieldGPU(
                width, height, resolution, device,
                n_sources=flickering_n_sources,
                source_strength=flickering_source_strength,
                flicker_period=400,  # 80%可见
                invisible_moves=100,  # 20%隐身
                source_speed=flickering_speed
            )
            print(f"  ✅ FPEF: {self.flickering_energy_field.field.shape}")
        
        # ============================================================
        # v17.5: 冰河世纪协议 (Ice Age Protocol)
        # ============================================================
        self.ice_age_enabled = ice_age_enabled
        self.ice_age_start_step = ice_age_start_step
        self.energy_dynamic_enabled = energy_dynamic_enabled
        self.energy_move_interval = energy_move_interval
        self.energy_jump_prob = energy_jump_prob
        self.energy_jump_dist = energy_jump_dist
        self.kif_storm_enabled = kif_storm_enabled
        self.kif_storm_count = kif_storm_count
        self.kif_storm_intensity = kif_storm_intensity
        self.kif_storm_move_speed = kif_storm_move_speed
        self.kif_storm_radius = kif_storm_radius
        
        # 冰河世纪内部状态
        self._ice_age_active = False
        self._energy_move_timer = 0
        self.kif_storms = []
        
        if ice_age_enabled:
            print(f"  🧊 冰河世纪协议已启用 (Step {ice_age_start_step})")
            if energy_dynamic_enabled:
                print(f"      能量源动态: 间隔{energy_move_interval}步, 跳跃概率{energy_jump_prob}")
            if kif_storm_enabled:
                print(f"      KIF风暴: {kif_storm_count}个, 强度{kif_storm_intensity}")
        
        # 预计算梯度矩阵 (GPU)
        self.epf_grad_x = None
        self.epf_grad_y = None
        self.kif_grad_x = None
        self.kif_grad_y = None
        self.isf_grad_x = None
        self.isf_grad_y = None

        # ============================================================
        # v16.0: 构成性物质场 (Matter Grid)
        # ============================================================
        self.matter_grid_enabled = matter_grid_enabled
        self.matter_resolution = matter_resolution

        if matter_grid_enabled:
            self.matter_grid_width = int(width / matter_resolution)
            self.matter_grid_height = int(height / matter_resolution)
            # Boolean grid: 0 = empty, 1 = solid matter
            self.matter_grid = torch.zeros(
                1, 1, self.matter_grid_height, self.matter_grid_width,
                device=device, dtype=torch.int8
            )
            # 能量存储网格 (用于全局能量守恒)
            self.matter_energy = torch.zeros(
                1, 1, self.matter_grid_height, self.matter_grid_width,
                device=device, dtype=torch.float32
            )
            print(f"  ✅ MatterGrid: {self.matter_grid_width}x{self.matter_grid_height}")
            print(f"    Energy storage enabled for conservation")
        else:
            self.matter_grid = None
            self.matter_energy = None
        
        # ============================================================
        # v16.0: 风场 (Wind Field) - Phase 3 挡风墙测试
        # ============================================================
        self.wind_field_enabled = wind_field_enabled
        
        if wind_field_enabled:
            try:
                from core.eoe.fields.wind import WindFieldGPU
                self.wind_field = WindFieldGPU(
                    width=width,
                    height=height,
                    direction=wind_direction,
                    damage_rate=wind_damage_rate,
                    device=device,
                    enabled=True,
                    resolution=matter_resolution
                )
                print(f"  ✅ WindField: direction={wind_direction} rad, damage={wind_damage_rate}")
            except ImportError as e:
                print(f"  ⚠️  WindField import failed: {e}")
                self.wind_field = None
        else:
            self.wind_field = None
        
        # 性能统计
        self.step_count = 0
        self.step_times = []
        
        # 季节系统
        self.seasons_enabled = False
        self.season_length = 500
        self.winter_multiplier = 0.2
        self.summer_multiplier = 1.5
    
    def set_seasons(self, enabled: bool, length: int = 500, winter: float = 0.2, summer: float = 1.5):
        """配置季节系统"""
        self.seasons_enabled = enabled
        self.season_length = length
        self.winter_multiplier = winter
        self.summer_multiplier = summer
    
    def get_seasonal_multiplier(self) -> float:
        """获取当前季节的能量倍率 (含干旱期)"""
        if not self.seasons_enabled:
            return 1.0
        
        # 四季循环 + 干旱期
        season_cycle = self.season_length * 4
        phase = (self.step_count % season_cycle) / season_cycle
        
        # 0.0-0.25: 春季 (恢复)
        # 0.25-0.5: 夏季 (繁荣)
        # 0.5-0.75: 秋季 (衰退)
        # 0.75-1.0: 冬季/干旱 (最艰难)
        
        if phase < 0.25:
            # 春季: 逐渐恢复
            t = phase / 0.25
            multiplier = self.winter_multiplier + t * (1.0 - self.winter_multiplier)
        elif phase < 0.5:
            # 夏季: 繁荣期
            t = (phase - 0.25) / 0.25
            multiplier = 1.0 + t * (self.summer_multiplier - 1.0)
        elif phase < 0.75:
            # 秋季: 逐渐衰退
            t = (phase - 0.5) / 0.25
            multiplier = self.summer_multiplier - t * (self.summer_multiplier - self.winter_multiplier)
        else:
            # 冬季/干旱: 最艰难时期
            t = (phase - 0.75) / 0.25
            if hasattr(self, 'drought_intensity'):
                # 干旱期: 能量极少
                multiplier = self.winter_multiplier * (1 - t * (1 - self.drought_intensity))
            else:
                multiplier = self.winter_multiplier
        
        return multiplier
    
    def step(self) -> float:
        """执行单步 - 返回耗时 (ms)"""
        import time
        start = time.perf_counter()
        
        # 0. 更新季节倍率
        if self.seasons_enabled and self.energy_field_enabled:
            seasonal_mult = self.get_seasonal_multiplier()
            self.energy_field.seasonal_multiplier = seasonal_mult
        
        # 1. 更新所有场
        if self.energy_field_enabled:
            self.energy_field.step()
        
        if self.impedance_field_enabled:
            self.impedance_field.step()
        
        if self.stigmergy_field_enabled:
            # v16.0: 传入 matter_grid 实现遮挡
            self.stigmergy_field.step(matter_grid=self.matter_grid)
        
        # v16.1: Flickering Energy Field
        if self.flickering_energy_enabled:
            self.flickering_energy_field.step()
        
        # v17.5: 冰河世纪协议 - 动态环境变化
        if self.ice_age_enabled and hasattr(self, 'ice_age_step'):
            self.ice_age_step()
        
        # 2. 预计算梯度 (每个场每步计算一次)
        self._compute_gradients()
        
        self.step_count += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        self.step_times.append(elapsed)
        
        return elapsed
    
    def ice_age_step(self):
        """v17.5: 冰河世纪协议 - 在每个step调用"""
        import numpy as np
        
        # 检查是否激活冰河世纪
        if not self._ice_age_active and self.step_count >= self.ice_age_start_step:
            self._ice_age_active = True
            print(f"\n  🧊 [冰河世纪] Step {self.step_count} 降临！")
            
            if self.energy_dynamic_enabled:
                print(f"      能量源开始逃逸！")
            
            if self.kif_storm_enabled:
                print(f"      🌩️ [KIF风暴] 生成 {self.kif_storm_count} 个风暴")
                for i in range(self.kif_storm_count):
                    self.kif_storms.append({
                        'x': np.random.uniform(10, 90),
                        'y': np.random.uniform(10, 90),
                        'vx': np.random.uniform(-1, 1) * self.kif_storm_move_speed,
                        'vy': np.random.uniform(-1, 1) * self.kif_storm_move_speed,
                        'radius': self.kif_storm_radius
                    })
        
        # 如果冰河世纪未激活，跳过
        if not self._ice_age_active:
            return
        
        # 1. 能量源动态移动
        if self.energy_dynamic_enabled and self.energy_field_enabled:
            self._energy_move_timer += 1
            if self._energy_move_timer >= self.energy_move_interval:
                self._energy_move_timer = 0
                
                # 执行Lévy飞行移动
                sources = self.energy_field.sources.cpu().numpy()
                for i in range(len(sources)):
                    if np.random.random() < self.energy_jump_prob:
                        # 跳跃模式
                        angle = np.random.uniform(0, 2 * np.pi)
                        dist = self.energy_jump_dist * (0.5 + np.random.random() * 0.5)
                        sources[i, 0] = (sources[i, 0] + dist * np.cos(angle)) % self.width
                        sources[i, 1] = (sources[i, 1] + dist * np.sin(angle)) % self.height
                
                self.energy_field.sources = torch.tensor(sources, device=self.device)
        
        # 2. KIF风暴移动和注入
        if self.kif_storm_enabled and self.kif_storms and self.impedance_field_enabled:
            field = self.impedance_field.field[0, 0].cpu().numpy()
            h, w = field.shape
            
            for storm in self.kif_storms:
                # 移动风暴
                storm['x'] = (storm['x'] + storm['vx']) % 100
                storm['y'] = (storm['y'] + storm['vy']) % 100
                
                # 随机改变方向
                if np.random.random() < 0.1:
                    storm['vx'] = np.random.uniform(-1, 1) * self.kif_storm_move_speed
                    storm['vy'] = np.random.uniform(-1, 1) * self.kif_storm_move_speed
                
                # 注入KIF
                gx = int(storm['x'])
                gy = int(storm['y'])
                r = int(storm['radius'])
                
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx = (gx + dx) % w
                        ny = (gy + dy) % h
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < r:
                            field[ny, nx] = max(field[ny, nx], 
                                                self.kif_storm_intensity * (1 - dist/r))
            
            self.impedance_field.field[0, 0] = torch.tensor(field, device=self.device)
    
    def _compute_gradients(self):
        """预计算所有场的梯度"""
        if self.energy_field_enabled:
            self.epf_grad_x, self.epf_grad_y = self.energy_field.compute_gradient()
        
        if self.impedance_field_enabled:
            self.kif_grad_x, self.kif_grad_y = self.impedance_field.compute_gradient()
        
        if self.stigmergy_field_enabled:
            self.isf_grad_x, self.isf_grad_y = self.stigmergy_field.compute_gradient()
    
    def get_field_values(
        self, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        批量采样场值 - 索引查找实现
        
        Args:
            positions: Tensor [N, 2] (x, y) - GPU 上的坐标
        Returns:
            Tensor [N, 9] - [EPF×3, KIF×3, ISF×3]
        """
        N = positions.shape[0]
        
        # 使用索引查找 (比 grid_sample 更简单高效)
        results = []
        
        # 计算网格坐标
        gx_idx = (positions[:, 0] / self.resolution).long()
        gy_idx = (positions[:, 1] / self.resolution).long()
        
        # 边界裁剪
        max_x = self.energy_field.grid_width - 1 if self.energy_field_enabled else 99
        max_y = self.energy_field.grid_height - 1 if self.energy_field_enabled else 99
        gx_idx = torch.clamp(gx_idx, 0, max_x)
        gy_idx = torch.clamp(gy_idx, 0, max_y)
        
        # EPF 采样 (中心 + 梯度)
        if self.energy_field_enabled:
            epf_field = self.energy_field.field[0, 0]  # [H, W]
            
            epf_c = epf_field[gy_idx, gx_idx]
            epf_gx = self.epf_grad_x[0, 0, gy_idx, gx_idx] if self.epf_grad_x is not None else torch.zeros(N, device=self.device)
            epf_gy = self.epf_grad_y[0, 0, gy_idx, gx_idx] if self.epf_grad_y is not None else torch.zeros(N, device=self.device)
            
            results.extend([epf_c, epf_gx, epf_gy])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        # KIF 采样
        if self.impedance_field_enabled:
            kif_field = self.impedance_field.field[0, 0]  # [H, W]
            
            kif_c = kif_field[gy_idx, gx_idx]
            kif_gx = self.kif_grad_x[0, 0, gy_idx, gx_idx] if self.kif_grad_x is not None else torch.zeros(N, device=self.device)
            kif_gy = self.kif_grad_y[0, 0, gy_idx, gx_idx] if self.kif_grad_y is not None else torch.zeros(N, device=self.device)
            
            results.extend([kif_c, kif_gx, kif_gy])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        # ISF 采样
        if self.stigmergy_field_enabled:
            isf_field = self.stigmergy_field.field[0, 0]  # [H, W]
            
            isf_c = isf_field[gy_idx, gx_idx]
            isf_gx = self.isf_grad_x[0, 0, gy_idx, gx_idx] if self.isf_grad_x is not None else torch.zeros(N, device=self.device)
            isf_gy = self.isf_grad_y[0, 0, gy_idx, gx_idx] if self.isf_grad_y is not None else torch.zeros(N, device=self.device)
            
            results.extend([isf_c, isf_gx, isf_gy])
        else:
            results.extend([torch.zeros(N, device=self.device)] * 3)
        
        return torch.stack(results, dim=1)  # [N, 9]
    
    def get_env_tensor(self, normalize: bool = True) -> torch.Tensor:
        """
        获取多通道环境张量 (Perception Field Mapping)
        
        Args:
            normalize: 是否归一化到 [0, 1]，强烈建议开启！
            
        Returns:
            Tensor [1, C, H, W] - 所有通道值域 [0, 1]
            
        通道顺序:
            Channel 0: ENERGY (能量场) - 归一化后 0~1
            Channel 1: IMPEDANCE (阻抗场) - 0~1
            Channel 2: STRESS (压力场) - 0~1 (独立!)
            Channel 3: STIGMERGY (信息素场) - 0~1
        """
        # 从子场获取网格大小
        if self.energy_field_enabled:
            H, W = self.energy_field.field.shape[2], self.energy_field.field.shape[3]
        elif self.impedance_field_enabled:
            H, W = self.impedance_field.field.shape[2], self.impedance_field.field.shape[3]
        else:
            H, W = int(self.height), int(self.width)
        
        channels = []
        
        # Channel 0: Energy Field (原始范围 0~200)
        if self.energy_field_enabled:
            energy = self.energy_field.field[0, 0]  # [H, W]
            if normalize:
                energy = energy / 200.0  # 归一化到 0~1
            channels.append(energy)
        else:
            channels.append(torch.zeros(H, W, device=self.device))
        
        # Channel 1: Impedance Field (原始范围 0~1)
        if self.impedance_field_enabled:
            impedance = self.impedance_field.field[0, 0]
            channels.append(impedance)  # 已是 0~1
        else:
            channels.append(torch.ones(H, W, device=self.device))
        
        # Channel 2: Stress Field (独立通道，不再与 IMPEDANCE 耦合)
        # 如果有独立的 stress 场则使用，否则留空让神经网络自己学习
        if hasattr(self, 'stress_field') and self.stress_field_enabled:
            stress = self.stress_field.field[0, 0]
            channels.append(stress)
        else:
            channels.append(torch.zeros(H, W, device=self.device))
        
        # Channel 3: Stigmergy Field (原始范围 0~1)
        if self.stigmergy_field_enabled:
            stigmergy = self.stigmergy_field.field[0, 0]
            channels.append(stigmergy)
        else:
            channels.append(torch.zeros(H, W, device=self.device))
        
        # 堆叠为 [C, H, W]，然后扩展为 [1, C, H, W]
        env_tensor = torch.stack(channels, dim=0).unsqueeze(0)
        
        # 安全检查：确保值域正确
        if normalize:
            env_tensor = torch.clamp(env_tensor, 0, 1)
        
        return env_tensor  # [1, C, H, W], 所有通道 [0, 1]
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        if not self.step_times:
            return {}
        
        return {
            'step_count': self.step_count,
            'avg_step_time_ms': np.mean(self.step_times),
            'min_step_time_ms': np.min(self.step_times),
            'max_step_time_ms': np.max(self.step_times),
        }

    # ============================================================
    # v16.0: 构成性物质场辅助方法 (Matter Grid)
    # ============================================================

    def is_solid(self, x: float, y: float) -> bool:
        """检查坐标是否为固体物质 (GPU tensor version)"""
        if self.matter_grid is None:
            return False
        gx = int(x / self.matter_resolution) % self.matter_grid_width
        gy = int(y / self.matter_resolution) % self.matter_grid_height
        return self.matter_grid[0, 0, gy, gx].item() == 1

    def add_matter(self, x: float, y: float, stored_energy: float = 0.0) -> bool:
        """
        在指定坐标添加物质，返回是否成功

        Args:
            x, y: 目标坐标
            stored_energy: 物质中存储的能量（用于守恒）
        """
        if self.matter_grid is None:
            return False
        gx = int(x / self.matter_resolution) % self.matter_grid_width
        gy = int(y / self.matter_resolution) % self.matter_grid_height
        if self.matter_grid[0, 0, gy, gx].item() == 0:
            self.matter_grid[0, 0, gy, gx] = 1
            self.matter_energy[0, 0, gy, gx] = stored_energy
            # 调试: 跟踪建造
            # print(f"    [墙壁] 建造于 ({gx}, {gy}), 共 {self.matter_grid.sum().item()} 个")
            return True
        return False

    def remove_matter(self, x: float, y: float) -> bool:
        """移除指定坐标的物质"""
        if self.matter_grid is None:
            return False
        gx = int(x / self.matter_resolution) % self.matter_grid_width
        gy = int(y / self.matter_resolution) % self.matter_grid_height
        if self.matter_grid[0, 0, gy, gx].item() == 1:
            self.matter_grid[0, 0, gy, gx] = 0
            self.matter_energy[0, 0, gy, gx] = 0.0
            return True
        return False

    def get_matter_energy(self, x: float, y: float) -> Optional[float]:
        """获取指定坐标物质存储的能量"""
        if self.matter_grid is None or self.matter_energy is None:
            return None
        gx = int(x / self.matter_resolution) % self.matter_grid_width
        gy = int(y / self.matter_resolution) % self.matter_grid_height
        if self.matter_grid[0, 0, gy, gx].item() == 1:
            return self.matter_energy[0, 0, gy, gx].item()
        return None

    # ========== v16.16: 批量 GPU 操作 (P0 优化) ==========
    
    def is_solid_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """
        批量检查多个坐标是否为固体
        
        Args:
            positions: [N, 2] tensor of (x, y) coordinates
            
        Returns:
            [N] bool tensor, True 表示该位置是固体
        """
        if self.matter_grid is None:
            return torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        
        gx = ((positions[:, 0] / self.matter_resolution).long() % self.matter_grid_width).clamp(0, self.matter_grid_width - 1)
        gy = ((positions[:, 1] / self.matter_resolution).long() % self.matter_grid_height).clamp(0, self.matter_grid_height - 1)
        
        # 批量采样 matter_grid
        flat_idx = gy * self.matter_grid_width + gx
        grid_flat = self.matter_grid.view(-1)
        return grid_flat[flat_idx] == 1
    
    def add_matter_batch(
        self, 
        positions: torch.Tensor, 
        stored_energy: float = 0.0,
        indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        批量添加物质
        
        Args:
            positions: [N, 2] tensor of (x, y) coordinates
            stored_energy: 存储的能量标量（所有位置相同）
            indices: 可选，指定要更新的 indices（用于去重）
            
        Returns:
            [N] bool tensor, True 表示该位置成功添加物质
        """
        if self.matter_grid is None or positions.shape[0] == 0:
            return torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        
        gx = ((positions[:, 0] / self.matter_resolution).long() % self.matter_grid_width).clamp(0, self.matter_grid_width - 1)
        gy = ((positions[:, 1] / self.matter_resolution).long() % self.matter_grid_height).clamp(0, self.matter_grid_height - 1)
        
        flat_idx = gy * self.matter_grid_width + gx
        
        # 如果指定了 indices，先去重
        if indices is not None:
            flat_idx = flat_idx[indices]
        
        # 检查当前位置是否为空
        grid_flat = self.matter_grid.view(-1)
        is_empty = grid_flat[flat_idx] == 0
        
        # 只在空位置添加
        success_indices = flat_idx[is_empty]
        
        if success_indices.shape[0] > 0:
            grid_flat[success_indices] = 1
            
            # 更新能量存储
            if self.matter_energy is not None:
                energy_flat = self.matter_energy.view(-1)
                energy_flat[success_indices] = stored_energy
        
        # 返回原始顺序的成功标记
        result = torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        if indices is not None:
            result[indices] = is_empty
        else:
            result = is_empty
            
        return result
    
    def remove_matter_batch(
        self, 
        positions: torch.Tensor,
        indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        批量移除物质
        
        Args:
            positions: [N, 2] tensor of (x, y) coordinates
            indices: 可选，指定要更新的 indices
            
        Returns:
            [N] bool tensor, True 表示该位置成功移除物质
        """
        if self.matter_grid is None or positions.shape[0] == 0:
            return torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        
        gx = ((positions[:, 0] / self.matter_resolution).long() % self.matter_grid_width).clamp(0, self.matter_grid_width - 1)
        gy = ((positions[:, 1] / self.matter_resolution).long() % self.matter_grid_height).clamp(0, self.matter_grid_height - 1)
        
        flat_idx = gy * self.matter_grid_width + gx
        
        # 如果指定了 indices，先去重
        if indices is not None:
            flat_idx = flat_idx[indices]
        
        # 检查当前位置是否有物质
        grid_flat = self.matter_grid.view(-1)
        has_matter = grid_flat[flat_idx] == 1
        
        # 只移除有物质的位置
        remove_indices = flat_idx[has_matter]
        
        if remove_indices.shape[0] > 0:
            grid_flat[remove_indices] = 0
            
            # 清空能量存储
            if self.matter_energy is not None:
                energy_flat = self.matter_energy.view(-1)
                energy_flat[remove_indices] = 0.0
        
        # 返回原始顺序的成功标记
        result = torch.zeros(positions.shape[0], dtype=torch.bool, device=self.device)
        if indices is not None:
            result[indices] = has_matter
        else:
            result = has_matter
            
        return result
    
    def get_matter_energy_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """
        批量获取物质存储的能量
        
        Args:
            positions: [N, 2] tensor of (x, y) coordinates
            
        Returns:
            [N] float tensor, 有物质的位置返回存储能量，无物质返回 0
        """
        if self.matter_grid is None or self.matter_energy is None:
            return torch.zeros(positions.shape[0], device=self.device)
        
        gx = ((positions[:, 0] / self.matter_resolution).long() % self.matter_grid_width).clamp(0, self.matter_grid_width - 1)
        gy = ((positions[:, 1] / self.matter_resolution).long() % self.matter_grid_height).clamp(0, self.matter_grid_height - 1)
        
        flat_idx = gy * self.matter_grid_width + gx
        
        # 获取能量值
        energy_flat = self.matter_energy.view(-1)
        energies = energy_flat[flat_idx]
        
        # 只返回有物质位置的能量
        grid_flat = self.matter_grid.view(-1)
        has_matter = grid_flat[flat_idx] == 1
        
        return energies * has_matter.float()
    # ========================================================


def benchmark_environment_gpu(n_steps: int = 100):
    """性能基准测试"""
    import time
    
    print("\n" + "="*60)
    print("🎯 EnvironmentGPU 性能基准测试")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 创建环境
    env = EnvironmentGPU(
        width=100, height=100,
        device=device,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True
    )
    
    # 预热
    print("\n预热 (10步)...")
    for _ in range(10):
        env.step()
    
    # 基准测试
    print(f"\n运行 {n_steps} 步...")
    start = time.perf_counter()
    
    for _ in range(n_steps):
        env.step()
    
    torch.cuda.synchronize() if device.startswith('cuda') else None
    
    elapsed = time.perf_counter() - start
    
    # 统计
    stats = env.get_stats()
    
    print(f"\n📊 结果:")
    print(f"  总耗时: {elapsed:.4f}s")
    print(f"  平均每步: {elapsed/n_steps*1000:.3f}ms")
    print(f"  吞吐量: {n_steps/elapsed:.1f} steps/sec")
    
    if stats:
        print(f"\n  详细统计:")
        print(f"    平均步耗时: {stats['avg_step_time_ms']:.3f}ms")
        print(f"    最小步耗时: {stats['min_step_time_ms']:.3f}ms")
        print(f"    最大步耗时: {stats['max_step_time_ms']:.3f}ms")
    
    # 对比 CPU 版本
    print("\n" + "="*60)
    print("📈 对比 CPU 版本")
    print("="*60)
    
    from core.eoe.environment import Environment as EnvironmentCPU
    
    env_cpu = EnvironmentCPU(
        width=100, height=100,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True,
        n_food=0
    )
    
    # CPU 预热
    for _ in range(5):
        env_cpu.step()
    
    # CPU 基准
    start = time.perf_counter()
    for _ in range(n_steps):
        env_cpu.step()
    cpu_elapsed = time.perf_counter() - start
    
    print(f"  CPU 每步: {cpu_elapsed/n_steps*1000:.3f}ms")
    print(f"  GPU 每步: {elapsed/n_steps*1000:.3f}ms")
    print(f"  🚀 加速比: {cpu_elapsed/elapsed:.1f}x")
    
    return env, env_cpu


# ============================================================
# v16.1 Flickering Energy Field (Deceptive Landscape)
# ============================================================
class FlickeringEnergyFieldGPU:
    """GPU Flickering Energy Field - periodic invisibility + inertia motion"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        n_sources: int = 30,
        source_strength: float = 50.0,
        flicker_period: int = 25,
        invisible_moves: int = 75,
        source_speed: float = 0.5,
    ):
        import numpy as np
        self.np = np
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        self.flicker_period = flicker_period
        self.invisible_moves = invisible_moves
        self.source_speed = source_speed
        self.step_count = 0
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        self.field = torch.zeros(
            1, 1, self.grid_height, self.grid_width,
            device=device, dtype=torch.float32
        )
        
        # [x, y, vx, vy, energy, visible]
        self.sources = torch.zeros(n_sources, 6, device=device)
        self.n_sources = n_sources
        self.source_strength = source_strength
        
        self._init_sources()
        
        # ========== v16.16: GPU 渲染卷积核 (P0 优化) ==========
        self._render_radius = 5
        kernel_size = 2 * self._render_radius + 1
        
        y, x = torch.meshgrid(
            torch.arange(-self._render_radius, self._render_radius + 1, device=self.device, dtype=torch.float32),
            torch.arange(-self._render_radius, self._render_radius + 1, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        distance = torch.sqrt(x**2 + y**2)
        self._render_kernel = torch.exp(-distance**2 / 8.0).view(1, 1, kernel_size, kernel_size)
        # ========================================================
    
    def _init_sources(self):
        import numpy as np
        for i in range(self.n_sources):
            x = torch.rand(1, device=self.device) * (self.width - 10) + 5
            y = torch.rand(1, device=self.device) * (self.height - 10) + 5
            
            angle = torch.rand(1, device=self.device) * 2 * np.pi
            vx = torch.cos(angle) * self.source_speed
            vy = torch.sin(angle) * self.source_speed
            
            energy = torch.full((1,), self.source_strength, device=self.device)
            visible = torch.ones(1, device=self.device)
            
            self.sources[i] = torch.cat([x, y, vx, vy, energy, visible])
    
    def step(self):
        import numpy as np
        self.step_count += 1
        
        cycle_length = self.flicker_period + self.invisible_moves
        cycle_pos = self.step_count % cycle_length
        is_visible = cycle_pos < self.flicker_period
        
        # 隐身期曲线运动标志
        is_invisible = not is_visible
        curved_motion = getattr(self, '_invisible_curved', True)
        
        for i in range(self.n_sources):
            src = self.sources[i]
            
            self.sources[i, 5] = 1.0 if is_visible else 0.0
            
            # 圆形运动模式 (新!)
            if getattr(self, '_circular_motion', False):
                # 更新相位
                phase_speed = 0.05  # 旋转速度
                self._circular_phase[i] += phase_speed
                phase = self._circular_phase[i].item()
                radius = self._circular_radius[i].item()
                speed = self.source_speed * radius * 2
                
                self.sources[i, 2] = np.cos(phase) * speed
                self.sources[i, 3] = np.sin(phase) * speed
            elif is_invisible and curved_motion:
                # 隐身期曲线运动: 添加微小的角度偏转
                current_vx = src[2].item()
                current_vy = src[3].item()
                current_speed = np.sqrt(current_vx**2 + current_vy**2)
                
                if current_speed > 0.01:
                    current_angle = np.arctan2(current_vy, current_vx)
                    # 随机偏转 ±15度
                    angle_offset = (np.random.random() - 0.5) * 0.5  # ~±15 degrees
                    new_angle = current_angle + angle_offset
                    
                    self.sources[i, 2] = np.cos(new_angle) * current_speed
                    self.sources[i, 3] = np.sin(new_angle) * current_speed
            
            new_x = src[0] + src[2]
            new_y = src[1] + src[3]
            
            if new_x < 0 or new_x > self.width:
                self.sources[i, 2] *= -1
                new_x = torch.clip(new_x, 0, self.width)
            if new_y < 0 or new_y > self.height:
                self.sources[i, 3] *= -1
                new_y = torch.clip(new_y, 0, self.height)
            
            self.sources[i, 0] = new_x
            self.sources[i, 1] = new_y
        
        self._render_field()
    
    def set_circular_motion(self, enabled: bool = True):
        """启用圆形/8字轨迹运动模式"""
        self._circular_motion = enabled
        self._circular_phase = torch.zeros(self.n_sources, device=self.device)
        self._circular_radius = torch.rand(self.n_sources, device=self.device) * 0.3 + 0.2  # 0.2-0.5 speed variation
        if enabled:
            print(f"  🔄 能量源圆形轨迹模式已启用")
    
    def _apply_circular_motion(self, idx: int, src):
        """应用圆形运动"""
        if not getattr(self, '_circular_motion', False):
            return src[2].item(), src[3].item()
        
        # 每个能量源有自己的相位
        phase = self._circular_phase[idx].item()
        radius = self._circular_radius[idx].item()
        speed = self.source_speed * radius * 2
        
        # 圆形运动: vx = cos(phase), vy = sin(phase)
        # 相位随时间推进
        new_vx = np.cos(phase) * speed
        new_vy = np.sin(phase) * speed
        
        return new_vx, new_vy
    
    def _render_field(self):
        """
        v16.16: GPU 向量化渲染 (P0 优化)
        
        使用 F.conv2d 批量计算所有能量源的光环辐射，
        替代原有的 Python 嵌套循环 + .item() 同步调用。
        
        性能提升: O(n×r²) → O(1), 消除 GPU 同步开销
        """
        # 1. 获取可见源 (capacity > threshold)
        visible_mask = self.sources[:, 5] > 0.5  # [N]
        if not visible_mask.any():
            self.field.zero_()
            return
        
        # 2. 提取可见源的坐标和强度 (保持在 GPU)
        visible_sources = self.sources[visible_mask]
        src_x = visible_sources[:, 0]  # [M]
        src_y = visible_sources[:, 1]  # [M]
        src_strength = visible_sources[:, 4]  # [M]
        
        # 3. 坐标转换为网格索引
        gx = (src_x / self.resolution).long().clamp(0, self.grid_width - 1)
        gy = (src_y / self.resolution).long().clamp(0, self.grid_height - 1)
        
        # 4. 创建脉冲网格 (GPU)
        B, C, H, W = self.field.shape
        impulses = torch.zeros((1, 1, H, W), device=self.device, dtype=torch.float32)
        
        # 5. 使用 scatter_add_ 批量写入脉冲信号
        flat_indices = gy * self.grid_width + gx  # [M]
        impulses.view(-1).scatter_add_(0, flat_indices, src_strength)
        
        # 6. 单次卷积完成所有光环渲染
        rendered = F.conv2d(
            impulses, 
            self._render_kernel, 
            padding=self._render_radius
        )
        
        # 7. 叠加到场 (保持梯度流)
        self.field += rendered
    
    def consume_at(self, x: float, y: float, radius: float = 2.0) -> float:
        gained = 0.0
        
        for i in range(self.n_sources):
            src = self.sources[i]
            
            dx = src[0].item() - x
            dy = src[1].item() - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < radius and src[4].item() > 0:
                gained += src[4].item()
                self.sources[i, 4] = 0.0
                
                self.sources[i, 0] = torch.rand(1, device=self.device) * (self.width - 20) + 10
                self.sources[i, 1] = torch.rand(1, device=self.device) * (self.height - 20) + 10
                self.sources[i, 4] = self.source_strength
        
        return gained
    
    def get_energy_at(self, x: float, y: float, sensor_range: float) -> float:
        import numpy as np
        total = 0.0
        
        for i in range(self.n_sources):
            src = self.sources[i]
            
            if src[5] < 0.5:
                continue
            
            dx = src[0].item() - x
            dy = src[1].item() - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < sensor_range:
                intensity = np.exp(-dist**2 / (sensor_range**2 / 2))
                total += src[4].item() * intensity
        
        return total
    
    def get_stats(self) -> dict:
        visible_count = (self.sources[:, 5] > 0.5).sum().item()
        return {
            'visible_sources': visible_count,
            'total_sources': self.n_sources,
            'invisible_moves': self.invisible_moves,
            'cycle_pos': self.step_count % (self.flicker_period + self.invisible_moves)
        }
    
    @property
    def is_visible_cycle(self) -> bool:
        """当前是否处于可见周期"""
        cycle_length = self.flicker_period + self.invisible_moves
        cycle_pos = self.step_count % cycle_length
        return cycle_pos < self.flicker_period
    
    def sample_batch(self, positions: torch.Tensor) -> torch.Tensor:
        """
        v16.16: 批量采样可见能量 (GPU 向量化)
        
        Args:
            positions: [N, 2] 位置坐标
            
        Returns:
            [N] 每个位置的能量值
        """
        # 使用 field 的采样（已渲染的场）
        return super().sample_batch(positions)
    
    def sample_all_sources_batch(self, positions: torch.Tensor, sensor_range: float = 15.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量采样所有能量源(包括隐身源) - GPU加速版
        返回: (total_energy, invisible_energy)
        """
        N = positions.shape[0]
        n_src = self.n_sources
        
        # sources: [n_sources, 6] -> [n_sources, 1, 6]
        # positions: [N, 2] -> [1, N, 2]
        src = self.sources.unsqueeze(1)  # [n_sources, 1, 6]
        pos = positions.unsqueeze(0)     # [1, N, 2]
        
        # 计算距离 (简化: 不考虑环形世界)
        diff = src[..., :2] - pos        # [n_sources, N, 2]
        dist_sq = (diff ** 2).sum(dim=2) # [n_sources, N]
        
        # 高斯衰减
        variance = (sensor_range ** 2) / 2
        intensity = torch.exp(-dist_sq / variance)  # [n_sources, N]
        
        # 能量 = 源能量 * 衰减强度
        src_energy = src[..., 4]  # [n_sources, 1]
        energy_at_pos = src_energy * intensity  # [n_sources, N]
        
        # 可见性
        is_visible = src[..., 5] > 0.5  # [n_sources, 1]
        
        # 总能量 (所有源)
        total_energy = energy_at_pos.sum(dim=0)  # [N]
        
        # 不可见能量 - 需要正确广播
        invisible_energy = torch.zeros(N, device=self.device)
        for i in range(n_src):
            if not is_visible[i, 0].item():
                invisible_energy += energy_at_pos[i]
        
        return total_energy, invisible_energy
    
    def consume_batch(self, positions: torch.Tensor, feed_amounts: torch.Tensor, sensor_range: float = 30.0) -> torch.Tensor:
        """
        v16.30: 直接从能量源消耗 - 基于距离衰减 (与采样一致!)
        
        Args:
            positions: [N, 2] agent positions
            feed_amounts: [N] requested energy to consume
            sensor_range: perception range
            
        Returns:
            [N] actual energy consumed
        """
        N = positions.shape[0]
        n_src = self.n_sources
        
        # sources: [n_sources, 6] -> [n_sources, 1, 6]
        # positions: [N, 2] -> [1, N, 2]
        src = self.sources.unsqueeze(1)  # [n_sources, 1, 6]
        pos = positions.unsqueeze(0)     # [1, N, 2]
        
        # 计算距离
        diff = src[..., :2] - pos        # [n_sources, N, 2]
        dist_sq = (diff ** 2).sum(dim=2) # [n_sources, N]
        
        # 高斯衰减 (与采样一致!)
        variance = (sensor_range ** 2) / 2
        intensity = torch.exp(-dist_sq / variance)  # [n_sources, N]
        
        # 可见性
        is_visible = src[..., 5] > 0.5  # [n_sources, 1]
        
        # 有效强度 = 源能量 × 可见性 × 距离衰减
        src_energy = src[..., 4]  # [n_sources, 1]
        effective_intensity = src_energy * intensity * is_visible.float()  # [n_sources, N]
        
        # 对每个agent，计算从所有源消耗的能量
        # 归一化强度作为权重
        total_intensity = effective_intensity.sum(dim=0)  # [N]
        total_intensity = total_intensity.clamp(min=1e-8)  # 避免除零
        
        # 按强度比例分配请求的能量
        weights = effective_intensity / total_intensity  # [n_sources, N]
        
        # 计算每个agent实际消耗
        actual_consumed = torch.zeros(N, device=self.device)
        for i in range(n_src):
            # 每个源对每个agent的贡献
            contribution = feed_amounts * weights[i]  # [N]
            
            # 从源容量中扣除 (按比例)
            src_capacity = self.sources[i, 4].item()
            total_from_src = contribution.sum().item()
            
            if total_from_src > 0 and src_capacity > 0:
                # 按请求比例扣除
                scale = min(1.0, src_capacity / total_from_src)
                actual_contribution = contribution * scale  # v16.30: 使用缩放后的值
                
                # 扣除源容量
                self.sources[i, 4] = src_capacity - total_from_src * scale
            else:
                actual_contribution = contribution
            
            actual_consumed += actual_contribution
        
        # 更新field (重新渲染)
        self._render_field()
        
        # v16.30: 检查并重生耗尽的源 (FlickeringEnergyFieldGPU版本)
        # 使用 source_strength 作为容量参考
        respawn_threshold = 0.15
        max_capacities = torch.full((self.n_sources,), self.source_strength, device=self.device)
        min_capacities = max_capacities * respawn_threshold
        remaining = self.sources[:, 4]
        need_respawn = remaining <= min_capacities
        
        if need_respawn.any():
            n_respawn = need_respawn.sum()
            respawn_indices = torch.where(need_respawn)[0]
            
            # 新位置
            self.sources[respawn_indices, 0] = torch.rand(n_respawn, device=self.device) * (self.width - 10) + 5
            self.sources[respawn_indices, 1] = torch.rand(n_respawn, device=self.device) * (self.height - 10) + 5
            
            # 重置容量
            self.sources[respawn_indices, 4] = self.source_strength * (0.8 + torch.rand(n_respawn, device=self.device) * 0.4)
        
        return actual_consumed
    
    def set_invisible_motion_curved(self, curved: bool = True):
        """设置隐身期运动模式: True=曲线运动, False=直线运动"""
        self._invisible_curved = curved


if __name__ == "__main__":
    benchmark_environment_gpu(100)