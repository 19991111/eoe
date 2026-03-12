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
        source_strength: float = 80.0, # 增加脉冲强度
        source_capacity: float = 300.0, # 减少总容量(更快枯竭)
        decay_rate: float = 0.98,     # 更快衰减
        respawn_threshold: float = 0.15 # 更早枯竭(15%)
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
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
        
        # 1. 能量自然衰减
        self.field *= self.decay_rate
        
        # 2. 脉冲式能量注入 (每10步一个脉冲)
        if self.step_count % 10 == 0:
            self._inject_energy_pulse()
        
        # 3. 检查并处理枯竭的源
        self._check_and_respawn()
    
    def _inject_energy_pulse(self):
        """脉冲式注入能量"""
        for i in range(self.n_sources):
            if self.sources[i, 3] > 0 and self.sources[i, 4] > 0:  # active and has capacity
                # 将源位置转换为网格坐标
                gx = int(self.sources[i, 0].item() / self.resolution) % self.grid_width
                gy = int(self.sources[i, 1].item() / self.resolution) % self.grid_height
                
                # 注入能量
                inject_amount = self.sources[i, 2].item()
                self.field[0, 0, gy, gx] += inject_amount
                
                # 消耗容量
                self.sources[i, 4] -= inject_amount
    
    def _check_and_respawn(self):
        """检查能量源是否枯竭，必要时迁移"""
        for i in range(self.n_sources):
            # 检查是否枯竭 (剩余容量 < 阈值的最大值)
            min_capacity = self.sources[i, 5] * self.respawn_threshold
            if self.sources[i, 4] < min_capacity and self.sources[i, 4] > 0:
                # 源已枯竭，重生到新位置
                self._spawn_source(i)
    
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
        从场中提取能量 (Agent吸取)
        
        Args:
            positions: [N, 2] 位置
            amounts: [N] 吸取量
        """
        gx = (positions[:, 0] / self.resolution).long() % self.grid_width
        gy = (positions[:, 1] / self.resolution).long() % self.grid_height
        
        # 批量提取
        for i in range(positions.shape[0]):
            if amounts[i] > 0:
                current = self.field[0, 0, gy[i], gx[i]]
                extracted = min(current.item(), amounts[i].item())
                self.field[0, 0, gy[i], gx[i]] -= extracted
    
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
    
    def compute_gradient(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算梯度 - GPU 加速"""
        # torch.gradient 返回 (grad_y, grad_x) - 注意顺序
        grad_y, grad_x = torch.gradient(self.field[0, 0])
        return grad_x.unsqueeze(0).unsqueeze(0), grad_y.unsqueeze(0).unsqueeze(0)


class KineticImpedanceFieldGPU:
    """GPU 加速阻抗场 (KIF)"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        noise_scale: float = 1.0,
        obstacle_density: float = 0.15
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # GPU 张量 - 使用 Perlin-like 噪声初始化
        self.field = self._generate_impedance_field(
            self.grid_width, self.grid_height, noise_scale, obstacle_density, device
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 预计算梯度
        self._grad_x = None
        self._grad_y = None
    
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
    
    def step(self):
        """单步扩散"""
        # 卷积扩散 (GPU 加速)
        diffused = F.conv2d(
            self.field, 
            self._diffusion_kernel, 
            padding=1
        )
        
        # 更新场
        self.field = self.field + diffused
        
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
        stigmergy_field_enabled: bool = True
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        print(f"[EnvironmentGPU] 初始化 GPU 环境 {width}x{height} on {device}")
        
        # 场初始化
        self.energy_field_enabled = energy_field_enabled
        self.impedance_field_enabled = impedance_field_enabled
        self.stigmergy_field_enabled = stigmergy_field_enabled
        
        if energy_field_enabled:
            self.energy_field = EnergyFieldGPU(
                width, height, resolution, device
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
        
        # 预计算梯度矩阵 (GPU)
        self.epf_grad_x = None
        self.epf_grad_y = None
        self.kif_grad_x = None
        self.kif_grad_y = None
        self.isf_grad_x = None
        self.isf_grad_y = None
        
        # 性能统计
        self.step_count = 0
        self.step_times = []
    
    def step(self) -> float:
        """执行单步 - 返回耗时 (ms)"""
        import time
        start = time.perf_counter()
        
        # 1. 更新所有场
        if self.energy_field_enabled:
            self.energy_field.step()
        
        if self.impedance_field_enabled:
            self.impedance_field.step()
        
        if self.stigmergy_field_enabled:
            self.stigmergy_field.step()
        
        # 2. 预计算梯度 (每个场每步计算一次)
        self._compute_gradients()
        
        self.step_count += 1
        
        elapsed = (time.perf_counter() - start) * 1000
        self.step_times.append(elapsed)
        
        return elapsed
    
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


if __name__ == "__main__":
    benchmark_environment_gpu(100)