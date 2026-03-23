"""
v13.0 运动阻抗场 (Kinetic Impedance Field, KIF)

"世界的骨架" - 定义空间的物理约束

物理模型:
- 连续标量场 Z(x,y) ∈ [0, ∞)
- 移动能耗: E = c × |F|² × log(1+Z)
- 位移抑制: V_actual = V_thrust / (1 + Z^0.5)
- 梯度反作用: F_repulsion = -α × ∇Z

核心思想:
- 废除离散分类 (FREE/NORMAL/MUD/WALL)
- 使用连续 Perlin 噪声场
- "墙"不存在于代码中，只存在于物理代价里

依赖:
- Agent: 智能体
"""

from typing import Tuple, Optional
import numpy as np


class KineticImpedanceField:
    """
    运动阻抗场 - 连续标量场
    
    物理本质:
    - Z = 0: 自由空间
    - Z = 1: 普通地面
    - Z > 10: 高阻抗区域 (沼泽、山地)
    - Z > 100: 近似墙壁
    
    实现:
    - 基于 Perlin/FBM 噪声生成连续地形
    - 支持 log(1+Z) 非线性缩放避免数值爆炸
    - 支持阻抗梯度计算
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        seed: int = 42,
        # 地形参数
        base_impedance: float = 1.0,      # 基础阻抗
        noise_scale: float = 0.05,        # 噪声缩放 (越大越崎岖)
        obstacle_density: float = 0.15,   # 障碍物密度
        max_impedance: float = 1000.0,    # 最大阻抗 (防止数值爆炸)
        # 梯度采样参数
        gradient_step: float = 5.0        # 梯度采样步长 (覆盖Agent身体)
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.seed = seed
        self.base_impedance = base_impedance
        self.noise_scale = noise_scale
        self.obstacle_density = obstacle_density
        self.max_impedance = max_impedance
        self.gradient_step = gradient_step
        
        # 网格尺寸
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 生成阻抗场
        self.field = self._generate_impedance_field()
        
    def _generate_impedance_field(self) -> np.ndarray:
        """
        基于 Perlin/FBM 噪声生成连续阻抗场
        
        使用多层噪声叠加:
        - 低频: 大陆轮廓
        - 中频: 地形起伏
        - 高频: 局部障碍
        """
        rng = np.random.RandomState(self.seed)
        
        # 初始化
        field = np.zeros((self.grid_width, self.grid_height), dtype=np.float64)
        
        # FBM 噪声叠加
        # 低频 (大尺度地形)
        field += self._perlin_noise_2d(
            self.grid_width, self.grid_height,
            scale=0.02 * self.noise_scale,
            seed=self.seed
        ) * 10.0
        
        # 中频 (局部起伏)
        field += self._perlin_noise_2d(
            self.grid_width, self.grid_height,
            scale=0.05 * self.noise_scale,
            seed=self.seed + 100
        ) * 5.0
        
        # 高频 (细节障碍)
        field += self._perlin_noise_2d(
            self.grid_width, self.grid_height,
            scale=0.15 * self.noise_scale,
            seed=self.seed + 200
        ) * 2.0
        
        # 添加随机障碍物 (基于密度)
        obstacle_threshold = 1.0 - self.obstacle_density
        field = np.where(field > obstacle_threshold * field.max(),
                        field * 2.0,  # 障碍区域阻抗翻倍
                        field)
        
        # 归一化到 [base_impedance, max_impedance]
        field = field - field.min() + self.base_impedance
        field = np.clip(field, self.base_impedance, self.max_impedance)
        
        return field
        
    def _perlin_noise_2d(self, w: int, h: int, scale: float, seed: int) -> np.ndarray:
        """
        生成2D Perlin噪声 (简化版 - 使用多倍频正弦叠加)
        
        为了性能，使用简化的噪声生成：
        - 多层正弦波叠加模拟 FBM 效果
        - 比真正的 Perlin 噪声快，适合实时演化
        """
        rng = np.random.RandomState(seed)
        
        # 坐标网格
        x = np.linspace(0, w * scale, w)
        y = np.linspace(0, h * scale, h)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # 多层正弦叠加 (模拟 FBM)
        noise = np.zeros((w, h))
        
        # 基频
        noise += np.sin(xx * 2 + rng.uniform(0, 2*np.pi)) * np.cos(yy * 2 + rng.uniform(0, 2*np.pi))
        
        # 第一倍频
        noise += 0.5 * np.sin(xx * 4 + rng.uniform(0, 2*np.pi)) * np.cos(yy * 4 + rng.uniform(0, 2*np.pi))
        
        # 第二倍频
        noise += 0.25 * np.sin(xx * 8 + rng.uniform(0, 2*np.pi)) * np.cos(yy * 8 + rng.uniform(0, 2*np.pi))
        
        # 归一化到 [0, 1]
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
    def sample(self, x: float, y: float) -> float:
        """
        采样指定位置的阻抗值
        
        参数:
            x, y: 世界坐标
        返回:
            阻抗值 Z (已应用 log(1+Z) 缩放)
        """
        # 环形世界坐标
        gx = (x / self.resolution) % self.grid_width
        gy = (y / self.resolution) % self.grid_height
        
        ix = int(gx)
        iy = int(gy)
        
        # 边界处理
        ix = np.clip(ix, 0, self.grid_width - 1)
        iy = np.clip(iy, 0, self.grid_height - 1)
        
        # 双线性插值
        fx = gx - ix
        fy = gy - iy
        
        Z = (1-fx)*(1-fy)*self.field[ix, iy] + \
            fx*(1-fy)*self.field[ix+1, iy] + \
            (1-fx)*fy*self.field[ix, iy+1] + \
            fx*fy*self.field[ix+1, iy+1]
            
        return Z
        
    def sample_log(self, x: float, y: float) -> float:
        """
        采样阻抗值并应用 log(1+Z) 非线性缩放
        
        用于能耗计算，避免数值爆炸
        """
        Z = self.sample(x, y)
        return np.log(1.0 + Z)
        
    def sample_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        采样阻抗梯度 ∇Z
        
        使用有限差分法，采样步长覆盖Agent身体
        方向: 指向阻抗增加方向
        
        参数:
            x, y: 世界坐标
        返回:
            (dZ/dx, dZ/dy) 梯度向量
        """
        step = self.gradient_step
        
        # 四个方向采样 (环形世界)
        x_plus = (x + step) % self.width
        x_minus = (x - step) % self.width
        y_plus = (y + step) % self.height
        y_minus = (y - step) % self.height
        
        Z_center = self.sample(x, y)
        Z_x_plus = self.sample(x_plus, y)
        Z_x_minus = self.sample(x_minus, y)
        Z_y_plus = self.sample(x, y_plus)
        Z_y_minus = self.sample(x, y_minus)
        
        # 有限差分 (考虑环形世界)
        dx = (Z_x_plus - Z_x_minus) / (2 * step)
        dy = (Z_y_plus - Z_y_minus) / (2 * step)
        
        return (dx, dy)
        
    def sample_gradient_magnitude(self, x: float, y: float) -> float:
        """采样梯度幅值 |∇Z|"""
        dx, dy = self.sample_gradient(x, y)
        return np.sqrt(dx*dx + dy*dy)
        
    def get_heatmap_data(self) -> np.ndarray:
        """获取热力图数据"""
        return self.field.T


# ============================================================
# KineticImpedanceLaw - 运动阻抗物理法则
# ============================================================

class KineticImpedanceLaw:
    """
    运动阻抗物理法则
    
    核心功能:
    1. 位移抑制: 高阻抗区域降低实际位移
    2. 能耗计算: 能耗与 log(1+Z) 成正比
    3. 梯度反作用: 阻抗梯度产生排斥力
    """
    
    def __init__(
        self,
        move_cost_coeff: float = 0.1,
        repulsion_coeff: float = 0.5,
        velocity_suppression: float = 0.1
    ):
        """
        参数:
            move_cost_coeff: 移动做功能量系数 c
            repulsion_coeff: 梯度反作用力系数 α
            velocity_suppression: 速度抑制系数
        """
        self.move_cost_coeff = move_cost_coeff
        self.repulsion_coeff = repulsion_coeff
        self.velocity_suppression = velocity_suppression
        
    def compute_displacement_and_cost(
        self,
        agent,
        impedance_field: KineticImpedanceField,
        left_force: float,
        right_force: float
    ) -> dict:
        """
        计算位移和能耗
        
        参数:
            agent: Agent实例
            impedance_field: 阻抗场
            left_force, right_force: 左右推进力
        
        返回:
            dict: {
                'thrust': 原始推力,
                'velocity': 实际位移,
                'impedance': 当前阻抗Z,
                'log_impedance': log(1+Z) 缩放值,
                'move_cost': 能耗
            }
        """
        # 原始推力
        thrust = (abs(left_force) + abs(right_force)) / 2.0
        
        # 采样阻抗 (使用log缩放)
        Z = impedance_field.sample(agent.x, agent.y)
        log_Z = np.log(1.0 + Z)  # 非线性缩放
        
        # 位移抑制: V_actual = V_thrust / (1 + Z^0.5)
        # Z越大，实际位移越小
        velocity = thrust / (1.0 + np.sqrt(Z) * self.velocity_suppression)
        
        # 能耗: E = c × |F|² × log(1+Z)
        move_cost = self.move_cost_coeff * (thrust ** 2) * log_Z
        
        return {
            'thrust': thrust,
            'velocity': velocity,
            'impedance': Z,
            'log_impedance': log_Z,
            'move_cost': move_cost
        }
        
    def compute_repulsion_force(
        self,
        agent,
        impedance_field: KineticImpedanceField
    ) -> Tuple[float, float]:
        """
        计算梯度反作用力
        
        F_repulsion = -α × ∇Z
        
        参数:
            agent: Agent实例
            impedance_field: 阻抗场
        
        返回:
            (fx, fy) 反作用力向量
        """
        dx, dy = impedance_field.sample_gradient(agent.x, agent.y)
        
        # 梯度指向阻抗增加方向，反作用力反向
        fx = -self.repulsion_coeff * dx
        fy = -self.repulsion_coeff * dy
        
        return (fx, fy)
        
    def apply_to_agent(
        self,
        agent,
        impedance_field: KineticImpedanceField,
        left_force: float,
        right_force: float
    ) -> dict:
        """
        将运动阻抗法则应用到Agent
        
        参数:
            agent: Agent实例
            impedance_field: 阻抗场
            left_force, right_force: 左右推进力
        
        返回:
            完整的物理状态字典
        """
        # 1. 计算位移和能耗
        move_info = self.compute_displacement_and_cost(
            agent, impedance_field, left_force, right_force
        )
        
        # 2. 计算梯度反作用力
        repulsion_x, repulsion_y = self.compute_repulsion_force(
            agent, impedance_field
        )
        
        # 3. 记录到Agent
        agent.impedance = move_info['impedance']
        agent.impedance_log = move_info['log_impedance']
        agent.repulsion_force = (repulsion_x, repulsion_y)
        
        return {
            **move_info,
            'repulsion_x': repulsion_x,
            'repulsion_y': repulsion_y
        }


def create_kinetic_impedance_field(
    width: float = 100.0,
    height: float = 100.0,
    terrain_type: str = 'natural'
) -> KineticImpedanceField:
    """
    创建运动阻抗场
    
    参数:
        terrain_type: 'natural' | 'labyrinth' | 'open'
    """
    if terrain_type == 'natural':
        return KineticImpedanceField(
            width=width, height=height,
            resolution=1.0,
            noise_scale=1.0,
            obstacle_density=0.15,
            gradient_step=5.0
        )
    elif terrain_type == 'labyrinth':
        return KineticImpedanceField(
            width=width, height=height,
            resolution=1.0,
            noise_scale=2.0,  # 更崎岖
            obstacle_density=0.30,  # 更多障碍
            gradient_step=3.0
        )
    else:  # open
        return KineticImpedanceField(
            width=width, height=height,
            resolution=1.0,
            noise_scale=0.3,
            obstacle_density=0.05,
            gradient_step=8.0
        )