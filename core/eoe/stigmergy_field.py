"""
v13.0 信息/压痕场 (Information/Stigmergy Field, ISF)

"社会的神经系统" - 环境的记忆系统

物理模型:
- 空间扩散: ∂S/∂t = D∇²S (拉普拉斯卷积)
- 衰减: S = S × decay_rate
- 软饱和: S_actual = S × (1 - S/S_max)
- 信号代价: E_cost = signal_strength² × cost_coeff (非线性)

核心思想:
- Agent 的移动轨迹在环境中留下持久信号
- 信号可被其他 Agent 读取 (非直接通讯)
- 信号释放消耗能量 (演化博弈)

依赖:
- Agent: 智能体
"""

from typing import Tuple, Optional
import numpy as np


class StigmergyField:
    """
    信息/压痕场 - 环境的记忆系统
    
    物理本质:
    - S(x,y): 信号强度 ∈ [0, S_max]
    - 扩散: 信号向周围蔓延
    - 衰减: 信号随时间挥发
    - 软饱和: 接近最大值时注入递减
    
    实现:
    - NumPy 向量化扩散计算
    - 预计算梯度矩阵供传感器使用
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        # 扩散参数
        diffusion_rate: float = 0.1,    # 扩散系数 (< 0.25 稳定)
        decay_rate: float = 0.98,       # 衰减率 (每步)
        # 注入参数
        base_deposit: float = 0.1,      # 基础注入量
        max_value: float = 10.0,        # 最大信号值 (软饱和上限)
        # 信号代价
        signal_energy_cost: float = 0.01  # 信号能量代价系数
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.base_deposit = base_deposit
        self.max_value = max_value
        self.signal_energy_cost = signal_energy_cost
        
        # 网格尺寸
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 压痕场矩阵
        self.field = np.zeros((self.grid_width, self.grid_height), dtype=np.float64)
        
        # 预计算梯度矩阵 (优化传感器性能)
        self.grad_x = np.zeros((self.grid_width, self.grid_height), dtype=np.float64)
        self.grad_y = np.zeros((self.grid_width, self.grid_height), dtype=np.float64)
        self.gradient_valid = False
        
        # 扩散核 (Laplacian 5点)
        self._laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        
    def step(self, matter_grid: Optional[np.ndarray] = None):
        """
        单步更新: 扩散 -> 衰减
        
        Args:
            matter_grid: 可选的物质网格，用于遮挡扩散
        """
        self._diffuse(matter_grid)
        self._decay()
        
        # 标记梯度需要重新计算
        self.gradient_valid = False
        
    def _diffuse(self, matter_grid: Optional[np.ndarray] = None):
        """
        空间扩散 - 使用卷积实现拉普拉斯算子
        
        v16.0 增强: MatterGrid 遮挡 (防止量子隧穿)
        - 必须在卷积前和后都应用掩码
        - 否则3x3卷积核会透过1像素的墙
        """
        # v16.0: 应用前置掩码 (防止墙后接收墙前扩散)
        if matter_grid is not None:
            # 1 = 可通行, 0 = 墙壁
            mask = (matter_grid == 0).astype(np.float64)
            masked_field = self.field * mask
        else:
            masked_field = self.field
        
        # 扩散计算
        try:
            from scipy.ndimage import convolve
            laplacian = convolve(masked_field, self._laplacian_kernel, mode='constant', cval=0.0)
        except ImportError:
            # 回退: 手动向量化
            laplacian = np.zeros_like(self.field)
            laplacian[1:-1, 1:-1] = (
                masked_field[0:-2, 1:-1] +
                masked_field[2:, 1:-1] +
                masked_field[1:-1, 0:-2] +
                masked_field[1:-1, 2:] -
                4 * masked_field[1:-1, 1:-1]
            )
            # 边界 (环形世界)
            laplacian[0, :] = (
                masked_field[-1, :] + masked_field[1, :] +
                masked_field[0, :-1] + masked_field[0, 1:] - 4 * masked_field[0, :]
            )
            laplacian[-1, :] = (
                masked_field[-2, :] + masked_field[0, :] +
                masked_field[-1, :-1] + masked_field[-1, 1:] - 4 * masked_field[-1, :]
            )
            laplacian[:, 0] = (
                masked_field[:, -1] + masked_field[:, 1] +
                masked_field[:-1, 0] + masked_field[1:, 0] - 4 * masked_field[:, 0]
            )
            laplacian[:, -1] = (
                masked_field[:, -2] + masked_field[:, 0] +
                masked_field[:-1, -1] + masked_field[1:, -1] - 4 * masked_field[:, -1]
            )
        
        # 扩散更新
        new_field = self.field + self.diffusion_rate * laplacian
        
        # v16.0: 应用后置掩码 (防止墙前受墙后回流)
        if matter_grid is not None:
            mask = (matter_grid == 0).astype(np.float64)
            new_field = new_field * mask
        
        self.field = new_field
        
        # 裁剪负值
        np.maximum(0, self.field, out=self.field)
        
    def _decay(self):
        """信号衰减 - 模拟挥发"""
        self.field *= self.decay_rate
        
    def deposit(
        self,
        x: float,
        y: float,
        amount: float,
        agent_energy: float = None
    ) -> float:
        """
        软饱和注入
        
        参数:
            x, y: 世界坐标
            amount: 期望注入量
            agent_energy: Agent当前能量 (用于计算信号代价)
        
        返回:
            实际注入量
        """
        # 环形世界坐标
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        
        # 计算信号代价 (非线性)
        # E_cost = amount² × cost_coeff
        if agent_energy is not None:
            signal_cost = (amount ** 2) * self.signal_energy_cost
            if agent_energy < signal_cost:
                # 能量不足，减少信号量
                amount = np.sqrt(agent_energy / self.signal_energy_cost) if signal_cost > 0 else 0
        
        # 软饱和公式: actual = amount × (1 - S/S_max)
        current = self.field[gx, gy]
        saturation_factor = 1.0 - (current / self.max_value)
        actual_amount = amount * max(0, saturation_factor)
        
        # 注入
        self.field[gx, gy] += actual_amount
        
        # 返回实际消耗的能量
        return actual_amount
        
    def compute_signal_cost(self, signal_amount: float) -> float:
        """
        计算信号释放的能量代价
        
        E_cost = signal² × cost_coeff (非线性)
        """
        return (signal_amount ** 2) * self.signal_energy_cost
        
    def sample(self, x: float, y: float) -> float:
        """采样信号强度"""
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        return self.field[gx, gy]
        
    def _compute_gradient(self):
        """预计算梯度矩阵 (供传感器使用)"""
        # 使用 np.gradient 计算全图梯度
        self.grad_y, self.grad_x = np.gradient(self.field)
        self.gradient_valid = True
        
    def sample_gradient(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        采样信号梯度
        
        返回: (grad_x, grad_y, magnitude)
        """
        if not self.gradient_valid:
            self._compute_gradient()
            
        gx = int(x / self.resolution) % self.grid_width
        gy = int(y / self.resolution) % self.grid_height
        
        # 双线性插值 (简化版: 直接取最近点)
        gx = np.clip(gx, 0, self.grid_width - 1)
        gy = np.clip(gy, 0, self.grid_height - 1)
        
        dx = self.grad_x[gx, gy]
        dy = self.grad_y[gx, gy]
        mag = np.sqrt(dx*dx + dy*dy)
        
        return (dx, dy, mag)
        
    def get_heatmap_data(self) -> np.ndarray:
        """获取热力图数据"""
        return self.field.T


# ============================================================
# StigmergyLaw - 压痕场物理法则
# ============================================================

class StigmergyLaw:
    """
    压痕场物理法则
    
    负责:
    - Agent 移动时的信号注入
    - 信号代价计算
    - 与 Agent 行为的集成
    """
    
    def __init__(
        self,
        base_deposit: float = 0.1,
        signal_energy_cost: float = 0.01,
        move_threshold: float = 0.1  # 只有移动超过此距离才注入
    ):
        self.base_deposit = base_deposit
        self.signal_energy_cost = signal_energy_cost
        self.move_threshold = move_threshold
        
    def apply_to_agent(
        self,
        agent,
        stigmergy_field: StigmergyField,
        old_x: float,
        old_y: float
    ) -> dict:
        """
        将压痕法则应用到 Agent
        
        参数:
            agent: Agent实例
            stigmergy_field: 压痕场
            old_x, old_y: 上一步位置
        
        返回:
            注入信息字典
        """
        # 计算移动距离
        dx = agent.x - old_x
        dy = agent.y - old_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # 只有移动超过阈值才注入信号
        if distance < self.move_threshold:
            return {
                'deposited': 0.0,
                'energy_cost': 0.0,
                'distance': distance
            }
        
        # 注入量与移动距离成正比
        desired_amount = self.base_deposit * distance
        
        # 计算信号代价
        energy_cost = stigmergy_field.compute_signal_cost(desired_amount)
        
        # 执行软饱和注入
        actual_deposited = stigmergy_field.deposit(
            agent.x, agent.y,
            desired_amount,
            agent.internal_energy
        )
        
        # 扣除能量
        if actual_deposited > 0:
            actual_cost = stigmergy_field.compute_signal_cost(actual_deposited)
            agent.internal_energy -= actual_cost
            agent.energy_spent += actual_cost
        
        return {
            'deposited': actual_deposited,
            'energy_cost': energy_cost,
            'distance': distance
        }


def create_stigmergy_field(
    width: float = 100.0,
    height: float = 100.0,
    terrain_type: str = 'natural'
) -> StigmergyField:
    """
    创建压痕场
    
    参数:
        terrain_type: 'natural' | 'labyrinth' | 'open'
    """
    if terrain_type == 'natural':
        return StigmergyField(
            width=width, height=height,
            diffusion_rate=0.1,
            decay_rate=0.98,
            base_deposit=0.1,
            signal_energy_cost=0.01
        )
    elif terrain_type == 'labyrinth':
        return StigmergyField(
            width=width, height=height,
            diffusion_rate=0.15,  # 更快的扩散
            decay_rate=0.95,      # 更慢的衰减
            base_deposit=0.15,
            signal_energy_cost=0.015
        )
    else:  # open
        return StigmergyField(
            width=width, height=height,
            diffusion_rate=0.05,
            decay_rate=0.99,
            base_deposit=0.05,
            signal_energy_cost=0.005
        )