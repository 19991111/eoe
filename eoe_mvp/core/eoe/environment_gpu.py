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
    """GPU 加速能量场 (EPF)"""
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        device: str = 'cuda:0',
        n_sources: int = 3,
        source_strength: float = 50.0,
        decay_rate: float = 0.99
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
        
        # 能量源 (GPU)
        self.sources = torch.zeros(n_sources, 4, device=device)  # [x, y, strength, active]
        self.n_sources = n_sources
        self.source_strength = source_strength
        self.decay_rate = decay_rate
        
        # 初始化源
        self._init_sources()
    
    def _init_sources(self):
        """初始化能量源"""
        for i in range(self.n_sources):
            # 随机位置
            self.sources[i, 0] = torch.rand(1, device=self.device) * self.width
            self.sources[i, 1] = torch.rand(1, device=self.device) * self.height
            self.sources[i, 2] = self.source_strength * (0.5 + torch.rand(1, device=self.device))
            self.sources[i, 3] = 1.0  # active
    
    def step(self):
        """单步更新"""
        # 1. 能量衰减
        self.field *= self.decay_rate
        
        # 2. 能量源注入
        self._inject_energy()
    
    def _inject_energy(self):
        """GPU 批量注入能量源"""
        # 将源位置转换为网格坐标
        source_x = (self.sources[:, 0] / self.resolution).long()
        source_y = (self.sources[:, 1] / self.resolution).long()
        
        # 边界检查
        source_x = torch.clamp(source_x, 0, self.grid_width - 1)
        source_y = torch.clamp(source_y, 0, self.grid_height - 1)
        
        # 批量注入 (使用高级索引)
        for i in range(self.n_sources):
            if self.sources[i, 3] > 0:  # active
                self.field[0, 0, source_y[i], source_x[i]] += self.sources[i, 2]
    
    def sample(self, x: float, y: float) -> float:
        """采样位置的能量值 (CPU 调用时)"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return self.field[0, 0, gy, gx].item()
    
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
        """注入信号"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        self.field[0, 0, gy, gx] += amount
    
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
        批量采样场值 - F.grid_sample 实现
        
        Args:
            positions: Tensor [N, 2] (x, y) - GPU 上的坐标
        Returns:
            Tensor [N, 6] - [EPF, EPF_gx, EPF_gy, KIF, KIF_gx, KIF_gy]
        """
        N = positions.shape[0]
        
        # 归一化坐标到 [-1, 1]
        grid_x = (positions[:, 0] / self.width) * 2 - 1
        grid_y = (positions[:, 1] / self.height) * 2 - 1
        
        # 构建 grid [N, 2] -> [N, 1, 1, 2]
        grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(1).unsqueeze(1)
        
        results = []
        
        # EPF 采样
        if self.energy_field_enabled:
            epf_field = self.energy_field.field  # [1, 1, H, W]
            
            # CENTER
            sampled = F.grid_sample(epf_field, grid, align_corners=False, mode='nearest')
            epf_c = sampled.squeeze()  # [N]
            
            # GRADIENT (使用预计算的梯度)
            # 采样位置对应的梯度
            gx_idx = (positions[:, 0] / self.energy_field.resolution).long()
            gy_idx = (positions[:, 1] / self.energy_field.resolution).long()
            gx_idx = torch.clamp(gx_idx, 0, self.energy_field.grid_width - 1)
            gy_idx = torch.clamp(gy_idx, 0, self.energy_field.grid_height - 1)
            
            epf_gx = self.epf_grad_x[0, 0, gy_idx, gx_idx]
            epf_gy = self.epf_grad_y[0, 0, gy_idx, gx_idx]
            
            results.extend([epf_c, epf_gx, epf_gy])
        
        # KIF 采样
        if self.impedance_field_enabled:
            kif_field = self.impedance_field.field
            
            gx_idx = (positions[:, 0] / self.impedance_field.resolution).long()
            gy_idx = (positions[:, 1] / self.impedance_field.resolution).long()
            gx_idx = torch.clamp(gx_idx, 0, self.impedance_field.grid_width - 1)
            gy_idx = torch.clamp(gy_idx, 0, self.impedance_field.grid_height - 1)
            
            kif_c = kif_field[0, 0, gy_idx, gx_idx]
            kif_gx = self.kif_grad_x[0, 0, gy_idx, gx_idx]
            kif_gy = self.kif_grad_y[0, 0, gy_idx, gx_idx]
            
            results.extend([kif_c, kif_gx, kif_gy])
        
        # ISF 采样
        if self.stigmergy_field_enabled:
            isf_field = self.stigmergy_field.field
            
            gx_idx = (positions[:, 0] / self.stigmergy_field.resolution).long()
            gy_idx = (positions[:, 1] / self.stigmergy_field.resolution).long()
            gx_idx = torch.clamp(gx_idx, 0, self.stigmergy_field.grid_width - 1)
            gy_idx = torch.clamp(gy_idx, 0, self.stigmergy_field.grid_height - 1)
            
            isf_c = isf_field[0, 0, gy_idx, gx_idx]
            isf_gx = self.isf_grad_x[0, 0, gy_idx, gx_idx]
            isf_gy = self.isf_grad_y[0, 0, gy_idx, gx_idx]
            
            results.extend([isf_c, isf_gx, isf_gy])
        
        return torch.stack(results, dim=1)  # [N, 6] or [N, 9]
    
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