from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from .node import NodeType
from .thermodynamic_law import ThermodynamicLaw
from .kinetic_impedance import KineticImpedanceField, KineticImpedanceLaw
from .stigmergy_field import StigmergyField, StigmergyLaw
from .stress_field import StressField, StressLaw

# ============================================================
# Numba JIT 加速 (v10.0)
# ============================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    
    @njit(cache=True)
    def _fast_toroidal_distance_scalar(x1, y1, x2, y2, width, height):
        """Numba加速的标量环形距离"""
        dx = x2 - x1
        dy = y2 - y1
        dx = dx - width * np.floor(dx / width + 0.5)
        dy = dy - height * np.floor(dy / height + 0.5)
        return np.sqrt(dx*dx + dy*dy)
    
    @njit(cache=True)
    def _fast_distance_matrix(agent_x, agent_y, food_x, food_y, width, height):
        """Numba加速的距离矩阵计算"""
        n_food = len(food_x)
        n_agent = len(agent_x)
        result = np.zeros((n_food, n_agent), dtype=np.float64)
        for i in range(n_food):
            for j in range(n_agent):
                dx = food_x[i] - agent_x[j]
                dy = food_y[i] - agent_y[j]
                dx = dx - width * np.floor(dx / width + 0.5)
                dy = dy - height * np.floor(dy / height + 0.5)
                result[i, j] = np.sqrt(dx*dx + dy*dy)
        return result
    
    @njit(cache=True)
    def _fast_clip(val, min_val, max_val):
        """Numba加速的裁剪"""
        if val < min_val:
            return min_val
        if val > max_val:
            return max_val
        return val
    
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: Numba not available, using NumPy fallback")

# 尝试导入其他性能优化函数
try:
    from ..core import compute_sensor_vectorized, update_positions_vectorized, compute_distances_vectorized
except ImportError:

    def update_positions_vectorized(x, y, theta, left_force, right_force, max_speed, turn_rate, width, height):
        diff = right_force - left_force
        new_theta = (theta + diff * turn_rate) % (2 * np.pi)
        # 优化: 用原生Python代替np.clip
        speed = (left_force + right_force) / 2.0
        speed = max(-max_speed, min(max_speed, speed))
        return (x + np.cos(new_theta) * speed) % width, (y + np.sin(new_theta) * speed) % height, new_theta

    def compute_sensor_vectorized(agent_x, agent_y, agent_theta, target_x, target_y, sensor_range, width=100.0, height=100.0):
        """环形世界传感器计算"""
        # 环形世界最短距离
        dx = target_x - agent_x
        dy = target_y - agent_y
        dx = dx - width * np.floor(dx / width + 0.5)
        dy = dy - height * np.floor(dy / height + 0.5)
        dist = np.hypot(dx, dy)
        if dist < 0.1:
            return np.array([1.0, 1.0])
        angle = np.arctan2(dy, dx) - agent_theta
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        sigma = np.radians(45)
        # 优化: 用原生Python代替np.clip
        decay = sensor_range / (dist + 1.0)
        decay = max(0.0, min(1.0, decay))
        return np.array([
            np.exp(-0.5 * ((angle + np.radians(30)) / sigma) ** 2),
            np.exp(-0.5 * ((angle - np.radians(30)) / sigma) ** 2)
        ]) * decay

# Forward declaration
class Agent:
    pass

class ChunkManager:
    """
    无限世界分块管理器 (v8.0)

    功能:
    - 基于柏林噪声的程序化生成
    - 按需加载/卸载区块
    - 探索代价机制
    - 确定性环境 (种子可重现)
    """

    CHUNK_SIZE = 50.0  # 每个区块的大小
    UNLOAD_THRESHOLD = 1000  # 无Agent时的卸载帧数
    EXPLORATION_TAX = 1.2  # 新区块探索代谢加成 (20%)

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 区块数据: {(chunk_x, chunk_y): chunk_data}
        self.chunks: Dict[Tuple[int, int], Dict] = {}

        # 访问记录: {(chunk_x, chunk_y): last_access_frame}
        self.chunk_access: Dict[Tuple[int, int], int] = {}

        # 已访问区块集合
        self.visited_chunks: Set[Tuple[int, int]] = set()

        # 当前帧数 (用于清理判定)
        self._current_frame = 0

        # 清理间隔 (每100帧尝试清理)
        self._cleanup_interval = 100
        self._last_cleanup_frame = 0

        # Perlin噪声生成器
        self._init_noise()

    def _init_noise(self):
        """初始化Perlin噪声参数"""
        self.noise_scale = 0.1  # 噪声缩放
        self.noise_offset = self.rng.uniform(0, 1000)

        # 初始化Perlin噪声梯度表
        self._init_perlin_gradients()

    def _init_perlin_gradients(self):
        """初始化Perlin噪声梯度表"""
        # 梯度向量 (8个方向)
        self.gradients = {}
        self.permutation = self.rng.permutation(256).tolist()
        # 扩展到512以避免模运算边界问题
        self.permutation = self.permutation + self.permutation

    def _fade(self, t: float) -> float:
        """Perlin fade函数: 6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """线性插值"""
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        """计算梯度"""
        # 将hash转为0-7
        h = hash_val & 7
        # 8个方向的梯度向量
        if h == 0:
            return x + y
        elif h == 1:
            return -x + y
        elif h == 2:
            return x - y
        elif h == 3:
            return -x - y
        elif h == 4:
            return x
        elif h == 5:
            return -x
        elif h == 6:
            return y
        else:
            return -y

    def _perlin_noise_2d(self, x: float, y: float) -> float:
        """
        真正的2D Perlin噪声 (梯度噪声)

        避免假正弦波产生的网格伪影
        """
        # 坐标偏移
        x += self.noise_offset
        y += self.noise_offset

        # 整数坐标
        xi = int(np.floor(x)) & 255
        yi = int(np.floor(y)) & 255

        # 小数坐标
        xf = x - np.floor(x)
        yf = y - np.floor(y)

        # 淡入淡出
        u = self._fade(xf)
        v = self._fade(yf)

        # 哈希角点
        aa = self.permutation[self.permutation[xi] + yi]
        ab = self.permutation[self.permutation[xi] + yi + 1]
        ba = self.permutation[self.permutation[xi + 1] + yi]
        bb = self.permutation[self.permutation[xi + 1] + yi + 1]

        # 计算梯度
        x1 = self._lerp(self._grad(aa, xf, yf), self._grad(ba, xf - 1, yf), u)
        x2 = self._lerp(self._grad(ab, xf, yf - 1), self._grad(bb, xf - 1, yf - 1), u)

        # 组合并归一化
        val = self._lerp(x1, x2, v)

        # 归一化到 [0, 1]
        return (val + 1.0) / 2.0

    def _fbm_noise(self, x: float, y: float, octaves: int = 4) -> float:
        """
        分形布朗运动 (FBM) - 多层Perlin噪声叠加

        参数:
            x, y: 坐标
            octaves: 叠加的八度音阶数

        返回:
            噪声值 [0, 1]
        """
        val = 0.0
        freq = self.noise_scale
        amp = 1.0
        max_val = 0.0

        for _ in range(octaves):
            val += amp * self._perlin_noise_2d(x * freq, y * freq)
            max_val += amp
            freq *= 2.0
            amp *= 0.5

        return (val / max_val + 1.0) / 2.0

    def get_chunk_coords(self, x: float, y: float) -> Tuple[int, int]:
        """获取坐标所在的区块坐标"""
        return (int(x // self.CHUNK_SIZE), int(y // self.CHUNK_SIZE))

    def generate_chunk(self, chunk_x: int, chunk_y: int) -> Dict:
        """生成单个区块"""
        chunk_key = (chunk_x, chunk_y)

        if chunk_key in self.chunks:
            # 更新访问时间
            self.chunk_access[chunk_key] = self._current_frame
            return self.chunks[chunk_key]

        # 基于区块坐标生成确定性数据
        chunk_seed = self.seed + hash(chunk_key) % 100000
        chunk_rng = np.random.RandomState(chunk_seed)

        # 食物分布 (基于真实Perlin噪声的FBM)
        foods = []
        # 使用FBM产生更自然的分布，避免网格伪影
        noise_val = self._fbm_noise(chunk_x * 0.5, chunk_y * 0.5, octaves=4)
        n_food_approx = 3 + int(noise_val * 4)

        # 使用泊松盘采样思想：避免食物过于密集
        min_dist = 8.0
        attempts = 0
        max_attempts = 20

        while len(foods) < n_food_approx and attempts < max_attempts:
            fx = chunk_x * self.CHUNK_SIZE + chunk_rng.uniform(5, self.CHUNK_SIZE - 5)
            fy = chunk_y * self.CHUNK_SIZE + chunk_rng.uniform(5, self.CHUNK_SIZE - 5)

            # 检查与现有食物的距离
            too_close = False
            for ex, ey in foods:
                if np.sqrt((fx - ex)**2 + (fy - ey)**2) < min_dist:
                    too_close = True
                    break

            if not too_close:
                foods.append((fx, fy))
            attempts += 1

        # 障碍物 (稀少的墙壁)
        walls = []
        wall_noise = self._fbm_noise(chunk_x * 0.3 + 100, chunk_y * 0.3 + 100)
        if wall_noise > 0.7:  # 约20%概率有墙
            n_walls = chunk_rng.randint(0, 2)
            for _ in range(n_walls):
                wx1 = chunk_x * self.CHUNK_SIZE + chunk_rng.uniform(10, self.CHUNK_SIZE - 20)
                wy1 = chunk_y * self.CHUNK_SIZE + chunk_rng.uniform(10, self.CHUNK_SIZE - 20)
                wx2 = wx1 + chunk_rng.uniform(5, 15)
                wy2 = wy1 + chunk_rng.uniform(5, 15)
                walls.append((wx1, wy1, wx2, wy2))

        # 地形难度 (离原点越远越难)
        dist_from_origin = np.sqrt(chunk_x**2 + chunk_y**2)
        difficulty = min(1.0, dist_from_origin * 0.1)

        chunk_data = {
            'foods': foods,
            'walls': walls,
            'difficulty': difficulty,
            'seed': chunk_seed
        }

        self.chunks[chunk_key] = chunk_data
        self.chunk_access[chunk_key] = self._current_frame
        return chunk_data

    def get_foods_in_range(self, x: float, y: float, sensor_range: float) -> List[Tuple[float, float]]:
        """获取指定范围内的食物"""
        chunk_x, chunk_y = self.get_chunk_coords(x, y)

        all_foods = []

        # 加载当前及相邻区块
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cx, cy = chunk_x + dx, chunk_y + dy
                chunk = self.generate_chunk(cx, cy)

                for fx, fy in chunk['foods']:
                    dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                    if dist <= sensor_range:
                        all_foods.append((fx, fy, dist))

        return all_foods

    def get_walls_in_range(self, x: float, y: float, sensor_range: float) -> List[Tuple]:
        """获取指定范围内的墙壁"""
        chunk_x, chunk_y = self.get_chunk_coords(x, y)

        all_walls = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cx, cy = chunk_x + dx, chunk_y + dy
                if (cx, cy) in self.chunks:
                    for wall in self.chunks[(cx, cy)]['walls']:
                        # 检查墙壁是否在范围内
                        wx1, wy1, wx2, wy2 = wall
                        wx_center, wy_center = (wx1 + wx2) / 2, (wy1 + wy2) / 2
                        dist = np.sqrt((x - wx_center)**2 + (y - wy_center)**2)
                        if dist <= sensor_range:
                            all_walls.append(wall)

        return all_walls

    def ensure_chunks_loaded(self, agents: List, sensor_range: float, current_step: int):
        """确保Agent感知范围内的区块已加载"""
        # 更新当前帧
        self._current_frame = current_step

        # 定期清理不活跃区块 (每_cleanup_interval帧)
        if current_step - self._last_cleanup_frame > self._cleanup_interval:
            self._last_cleanup_frame = current_step
            self.cleanup_inactive_chunks(current_step)

        for agent in agents:
            if not hasattr(agent, 'is_alive') or not agent.is_alive:
                continue

            chunk_x, chunk_y = self.get_chunk_coords(agent.x, agent.y)

            # 加载感知范围2倍内的区块
            load_radius = int(np.ceil(sensor_range * 2 / self.CHUNK_SIZE)) + 1

            for dx in range(-load_radius, load_radius + 1):
                for dy in range(-load_radius, load_radius + 1):
                    cx, cy = chunk_x + dx, chunk_y + dy
                    self.generate_chunk(cx, cy)

                    # 记录访问
                    key = (cx, cy)
                    if key not in self.visited_chunks:
                        self.visited_chunks.add(key)
                        agent.is_in_new_chunk = True  # 标记新区块探索

    def is_chunk_visited(self, x: float, y: float) -> bool:
        """检查坐标所在的区块是否被访问过"""
        chunk_key = self.get_chunk_coords(x, y)
        return chunk_key in self.visited_chunks

    def get_difficulty(self, x: float, y: float) -> float:
        """获取坐标所在区块的难度"""
        chunk_x, chunk_y = self.get_chunk_coords(x, y)
        chunk = self.generate_chunk(chunk_x, chunk_y)
        return chunk['difficulty']

    def cleanup_inactive_chunks(self, current_step: int):
        """清理不活跃的区块"""
        to_remove = []

        for chunk_key, last_access in self.chunk_access.items():
            if current_step - last_access > self.UNLOAD_THRESHOLD:
                to_remove.append(chunk_key)

        for key in to_remove:
            if key in self.chunks:
                del self.chunks[key]
            del self.chunk_access[key]

        return len(to_remove)

    def record_access(self, x: float, y: float, step: int):
        """记录区块访问"""
        chunk_key = self.get_chunk_coords(x, y)
        self.chunk_access[chunk_key] = step


# ============================================================
# v13.0: 能量场物理系统 (Energy Field Physics)
# ============================================================

class EnergySource:
    """
    能量泉眼 - 持续向环境注入能量的固定源
    
    属性:
        position: (x, y) 坐标
        injection_rate: 能量注入速率 (单位/帧)
        radius: 影响半径
    """
    
    def __init__(self, x: float, y: float, injection_rate: float = 0.5, radius: float = 15.0):
        self.position = (x, y)
        self.injection_rate = injection_rate
        self.radius = radius
        
    def __repr__(self):
        return f"EnergySource{self.position} rate={self.injection_rate:.2f} r={self.radius}"


class EnergyField:
    """
    连续能量场 E(x, y) - 替代离散食物系统
    
    物理模型:
    - 扩散方程: ∂E/∂t = D∇²E - kE
    - 能量源: 固定位置持续注入
    - 环形世界: Toroidal边界条件
    
    v13.0 设计要点:
    - NumPy向量化实现扩散 (避免Python循环)
    - 支持与Agent的能量交换
    - 热力图可视化支持
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 1.0,
        diffusion_rate: float = 0.05,
        decay_rate: float = 0.001,
        sources: Optional[List[EnergySource]] = None
    ):
        self.width = width
        self.height = height
        self.resolution = resolution  # 网格分辨率
        
        # 网格尺寸
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 能量场矩阵 (初始化为低能量背景)
        self.field = np.zeros((self.grid_width, self.grid_height), dtype=np.float64)
        
        # 物理参数
        self.diffusion_rate = diffusion_rate  # 扩散系数 D
        self.decay_rate = decay_rate           # 衰减系数 k
        
        # 能量源
        self.sources = sources or []
        
        # 预计算扩散核 (Laplacian 5-point stencil)
        # 使用向量化操作，无需显式循环
        self._laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        
    def add_source(self, source: EnergySource):
        """添加能量源"""
        self.sources.append(source)
        
    def add_default_sources(self):
        """添加默认能量源 (3个泉眼)"""
        self.sources = [
            EnergySource(25, 25, injection_rate=0.5, radius=15),
            EnergySource(75, 75, injection_rate=0.5, radius=15),
            EnergySource(75, 25, injection_rate=0.3, radius=10),
        ]
        
    def _inject_energy(self):
        """向能量场注入能量 (向量化)"""
        for source in self.sources:
            sx, sy = source.position
            sx_idx = int(sx / self.resolution)
            sy_idx = int(sy / self.resolution)
            radius_idx = int(source.radius / self.resolution)
            
            # 创建局部坐标网格 (向量化计算距离)
            x = np.arange(max(0, sx_idx - radius_idx), min(self.grid_width, sx_idx + radius_idx + 1))
            y = np.arange(max(0, sy_idx - radius_idx), min(self.grid_height, sy_idx + radius_idx + 1))
            
            if len(x) == 0 or len(y) == 0:
                continue
                
            xx, yy = np.meshgrid(x, y, indexing='ij')
            dist = np.sqrt((xx - sx_idx)**2 + (yy - sy_idx)**2)
            
            # 距离衰减注入
            mask = dist <= radius_idx
            injection = source.injection_rate * np.exp(-dist / (radius_idx + 1e-6))
            injection = injection * mask
            
            # 应用注入 (边界检查)
            for ix in x:
                for iy in y:
                    if 0 <= ix < self.grid_width and 0 <= iy < self.grid_height:
                        self.field[ix, iy] += injection[ix - x[0], iy - y[0]]
                        
    def _diffuse(self):
        """
        能量扩散 (向量化实现)
        
        使用SciPy或NumPy的卷积实现Laplacian
        ∂E/∂t = D∇²E
        """
        try:
            from scipy.ndimage import convolve
            # 使用卷积计算Laplacian
            laplacian = convolve(self.field, self._laplacian_kernel, mode='constant', cval=0.0)
        except ImportError:
            # 回退方案: 手动向量化计算 (略慢但兼容)
            laplacian = np.zeros_like(self.field)
            
            # 向量化5点Laplacian
            laplacian[1:-1, 1:-1] = (
                self.field[0:-2, 1:-1] +
                self.field[2:, 1:-1] +
                self.field[1:-1, 0:-2] +
                self.field[1:-1, 2:] -
                4 * self.field[1:-1, 1:-1]
            )
            
            # 边界处理 (环形世界)
            laplacian[0, :] = (
                self.field[-1, :] + self.field[1, :] +
                self.field[0, :-1] + self.field[0, 1:] - 4 * self.field[0, :]
            )
            laplacian[-1, :] = (
                self.field[-2, :] + self.field[0, :] +
                self.field[-1, :-1] + self.field[-1, 1:] - 4 * self.field[-1, :]
            )
            laplacian[:, 0] = (
                self.field[:, -1] + self.field[:, 1] +
                self.field[:-1, 0] + self.field[1:, 0] - 4 * self.field[:, 0]
            )
            laplacian[:, -1] = (
                self.field[:, -2] + self.field[:, 0] +
                self.field[:-1, -1] + self.field[1:, -1] - 4 * self.field[:, -1]
            )
        
        # 扩散更新
        self.field += self.diffusion_rate * laplacian
        
    def _decay(self):
        """能量衰减 (熵增) ∂E/∂t = -kE"""
        self.field *= (1.0 - self.decay_rate)
        
    def step(self):
        """单步更新: 注入 -> 扩散 -> 衰减"""
        self._inject_energy()
        self._diffuse()
        self._decay()
        
        # 边界裁剪 (防止数值爆炸)
        np.clip(self.field, 0, 1000, out=self.field)
        
    def sample(self, x: float, y: float) -> float:
        """
        采样指定位置的场能量 (双线性插值)
        
        参数:
            x, y: 世界坐标
        返回:
            能量值
        """
        gx = x / self.resolution
        gy = y / self.resolution
        
        # 环形世界坐标
        gx = gx % self.grid_width
        gy = gy % self.grid_height
        
        ix = int(gx)
        iy = int(gy)
        
        # 边界情况
        if ix >= self.grid_width - 1 or iy >= self.grid_height - 1:
            ix = min(ix, self.grid_width - 1)
            iy = min(iy, self.grid_height - 1)
            return self.field[ix, iy]
        
        # 双线性插值
        fx = gx - ix
        fy = gy - iy
        
        return (1-fx)*(1-fy)*self.field[ix, iy] + \
               fx*(1-fy)*self.field[ix+1, iy] + \
               (1-fx)*fy*self.field[ix, iy+1] + \
               fx*fy*self.field[ix+1, iy+1]
    
    def sample_gradient(self, x: float, y: float) -> Tuple[float, float, float, float, float]:
        """
        采样周围4个方向的能量梯度
        
        返回: (front, right, back, left, center)
        用于Agent的传感器输入
        """
        # 4个方向偏移 (基于环形世界)
        offsets = [
            (0, -5),   # 前
            (5, 0),    # 右
            (0, 5),    # 后  
            (-5, 0),   # 左
        ]
        
        results = []
        for dx, dy in offsets:
            # 环形世界坐标
            nx = (x + dx) % self.width
            ny = (y + dy) % self.height
            results.append(self.sample(nx, ny))
            
        results.append(self.sample(x, y))  # 中心
        
        return tuple(results)
        
    def get_heatmap_data(self) -> np.ndarray:
        """获取热力图数据 (转置以便正确显示)"""
        return self.field.T


# ============================================================
# 4. 2D 趋化性环境 (Environment)
# ============================================================

class Environment:
    """
    2D 趋化性沙盒环境 - EOE v3.0 (含障碍物、昼夜循环、社会感知)

    属性:
        width, height: 空间尺寸 (100x100)
        food_positions: 多个食物位置
        agents: 智能体列表
        walls: 障碍物列表

    传感器模型:
        - 基于距离的反比例衰减
        - 左右传感器基于 agent 当前的朝向计算
        - 越接近食物，气味浓度越高

    多智能体竞争:
        - 有限食物 (默认 5 个)
        - Agent 吃到食物获得能量并分裂
        - 能量耗尽立即死亡
        - 生命周期结束时没吃到食物的淘汰

    v3.0 新增:
        - walls: 不可逾越的障碍物
        - day/night cycle: 代谢税率波动
        - ENTITY_RADAR: 社会传感器
        - energy theft: 能量夺取
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        target_pos: Optional[Tuple[float, float]] = None,
        metabolic_alpha: float = 0.003,  # v10.4: 默认0.003 (低代谢，允许大脑复杂度)
        metabolic_beta: float = 0.05,
        surprise_penalty: float = 0.5,
        n_food: int = 5,
        food_energy: float = 30.0,
        respawn_food: bool = True,
        n_walls: int = 0,
        day_night_cycle: bool = True,
        pure_survival_mode: bool = False,  # v0.74: 纯生存适应度
        # v0.78: 季节系统
        seasonal_cycle: bool = False,
        season_length: int = 50,
        winter_food_multiplier: float = 0.0,
        winter_metabolic_multiplier: float = 2.0,
        # v0.99: 即时进食模式（降维打击）
        # 当开启时，拾取食物 = 直接进食 + 能量恢复
        # 这样可以让 Agent 建立"寻找食物=活下去"的基础反射
        immediate_eating: bool = False,
        # ============================================================
        # v11.0: 三大突破机制 (2026-03-12)
        # ============================================================
        # 1. 代谢熵增: 体内能量随时间挥发，流失率 ∝ E²
        energy_decay_k: float = 0.0001,  # 能量衰减系数
        # 2. 端口干涉: 多端口同时激活时代谢成本指数上升
        port_interference_gamma: float = 2.0,  # 干涉gamma值
        # 3. 季节波动率: 每代环境参数 ±X% 扰动
        season_jitter: float = 0.10,  # 10% 扰动幅度
        # 入库税: 贮粮入库时一次性扣除
        nest_tax: float = 0.10,  # 10% 入库税
        # ============================================================
        # v13.0: 能量场物理系统 (Energy Field Physics)
        # ============================================================
        energy_field_enabled: bool = False,  # 是否启用能量场 (替代离散食物)
        field_resolution: float = 1.0,       # 能量场网格分辨率
        field_diffusion_rate: float = 0.05,  # 扩散系数 D
        field_decay_rate: float = 0.001,     # 衰减系数 k
        field_initial_energy: float = 0.1,   # 初始场能量 (背景值)
        # 渗透膜参数
        permeability_cost: float = 0.01,     # 维持渗透膜的代价
        waste_heat_ratio: float = 0.3,       # 代谢转化为废热的比例
        move_cost_coeff: float = 0.1,        # 移动做功能量系数
        # ============================================================
        # v13.0: 运动阻抗场 (KIF)
        # ============================================================
        impedance_field_enabled: bool = False,  # 是否启用运动阻抗场
        impedance_resolution: float = 1.0,      # 阻抗场分辨率
        impedance_noise_scale: float = 1.0,     # 噪声缩放 (越大越崎岖)
        impedance_obstacle_density: float = 0.15,  # 障碍物密度
        impedance_repulsion_coeff: float = 0.5,  # 梯度反作用力系数
        # ============================================================
        # v13.0: 信息/压痕场 (ISF)
        # ============================================================
        stigmergy_field_enabled: bool = False,  # 是否启用压痕场
        stigmergy_resolution: float = 1.0,      # 压痕场分辨率
        stigmergy_diffusion_rate: float = 0.1,  # 信号扩散率
        stigmergy_decay_rate: float = 0.98,     # 信号衰减率
        stigmergy_signal_cost: float = 0.01,     # 信号能量代价系数
        # ============================================================
        # v13.0: 环境应力场 (ESF)
        # ============================================================
        stress_field_enabled: bool = False,       # 是否启用应力场
        stress_resolution: float = 2.0,           # 应力场分辨率
        stress_temp_period: int = 200,            # 温度周期
        stress_season_length: int = 200           # 应力周期
    ):
        self.width = width
        self.height = height
        self.pure_survival_mode = pure_survival_mode  # v0.74
        self.agents: List[Agent] = []
        
        # v0.99: 即时进食模式
        self.immediate_eating = immediate_eating
        
        # v1.3: 可配置的物理常量 (解耦硬编码)
        self.CHUNK_SIZE = getattr(self, 'CHUNK_SIZE', 50.0)  # 默认值，保持兼容

        # v5.2: 步数计数器
        self.step_count = 0  # 用于预测偏差计算

        # ============================================================
        # v13.0: 能量场物理系统
        # ============================================================
        self.energy_field_enabled = energy_field_enabled
        self.permeability_cost = permeability_cost
        self.waste_heat_ratio = waste_heat_ratio
        self.move_cost_coeff = move_cost_coeff
        
        if energy_field_enabled:
            # 初始化能量场
            self.energy_field = EnergyField(
                width=width,
                height=height,
                resolution=field_resolution,
                diffusion_rate=field_diffusion_rate,
                decay_rate=field_decay_rate
            )
            # 添加默认能量源
            self.energy_field.add_default_sources()
            # 设置初始背景能量
            self.energy_field.field.fill(field_initial_energy)
            print(f"  [EnergyField] Enabled: {self.energy_field.grid_width}x{self.energy_field.grid_height} grid")
            print(f"    Sources: {len(self.energy_field.sources)}")
        else:
            self.energy_field = None
            
        # v13.0: 热力学物理法则
        self.thermodynamic_law = ThermodynamicLaw(
            permeability_cost=permeability_cost,
            waste_heat_ratio=waste_heat_ratio,
            move_cost_coeff=move_cost_coeff
        )

        # ============================================================
        # v13.0: 运动阻抗场 (Kinetic Impedance Field)
        # ============================================================
        self.impedance_field_enabled = impedance_field_enabled
        
        if impedance_field_enabled:
            # 初始化阻抗场
            self.impedance_field = KineticImpedanceField(
                width=width,
                height=height,
                resolution=impedance_resolution,
                noise_scale=impedance_noise_scale,
                obstacle_density=impedance_obstacle_density
            )
            # 初始化阻抗法则
            self.kinetic_impedance_law = KineticImpedanceLaw(
                move_cost_coeff=move_cost_coeff,
                repulsion_coeff=impedance_repulsion_coeff
            )
            print(f"  [KineticImpedanceField] Enabled: {self.impedance_field.grid_width}x{self.impedance_field.grid_height} grid")
            print(f"    Noise scale: {impedance_noise_scale}, Obstacle density: {impedance_obstacle_density}")
        else:
            self.impedance_field = None
            self.kinetic_impedance_law = None

        # ============================================================
        # v13.0: 信息/压痕场 (Stigmergy Field)
        # ============================================================
        self.stigmergy_field_enabled = stigmergy_field_enabled
        
        if stigmergy_field_enabled:
            # 初始化压痕场
            self.stigmergy_field = StigmergyField(
                width=width,
                height=height,
                resolution=stigmergy_resolution,
                diffusion_rate=stigmergy_diffusion_rate,
                decay_rate=stigmergy_decay_rate,
                signal_energy_cost=stigmergy_signal_cost
            )
            # 初始化压痕法则
            self.stigmergy_law = StigmergyLaw(
                base_deposit=0.1,
                signal_energy_cost=stigmergy_signal_cost
            )
            print(f"  [StigmergyField] Enabled: {self.stigmergy_field.grid_width}x{self.stigmergy_field.grid_height} grid")
            print(f"    Diffusion: {stigmergy_diffusion_rate}, Decay: {stigmergy_decay_rate}, Cost: {stigmergy_signal_cost}")
        else:
            self.stigmergy_field = None
            self.stigmergy_law = None

        # ============================================================
        # v13.0: 环境应力场 (Stress Field)
        # ============================================================
        self.stress_field_enabled = stress_field_enabled
        self.base_diffusion_rate = field_diffusion_rate  # 保存基准值
        self.base_decay_rate = field_decay_rate          # 保存基准值
        self.stress_metabolic_multiplier = 1.0           # ESF 代谢调制
        
        if stress_field_enabled:
            # 初始化应力场
            self.stress_field = StressField(
                width=width,
                height=height,
                resolution=stress_resolution,
                temp_period=stress_temp_period,
                metabolic_period=int(stress_temp_period * 0.75),
                diffusion_period=int(stress_temp_period * 0.5),
                impedance_period=int(stress_temp_period * 0.9)
            )
            # 初始化应力法则
            self.stress_law = StressLaw()
            print(f"  [StressField] Enabled: {self.stress_field.grid_width}x{self.stress_field.grid_height} grid")
            print(f"    Temperature period: {stress_temp_period}")
        else:
            self.stress_field = None
            self.stress_law = None

        # ============================================================
        # v13.0: 预计算梯度矩阵 (性能优化)
        # 在Environment层级全局计算一次，Agent只做O(1)索引
        # ============================================================
        self.epf_grad_x: Optional[np.ndarray] = None
        self.epf_grad_y: Optional[np.ndarray] = None
        self.kif_grad_x: Optional[np.ndarray] = None
        self.kif_grad_y: Optional[np.ndarray] = None
        self.isf_grad_x: Optional[np.ndarray] = None
        self.isf_grad_y: Optional[np.ndarray] = None

        # ============================================================
        # 多食物系统 (马尔萨斯竞速)
        # ============================================================
        self.n_food = n_food
        self.food_energy = food_energy  # v9.0: 已减半
        self.respawn_food = respawn_food
        self.food_positions: List[Tuple[float, float]] = []

        # ============================================================
        # v9.0 逃逸食物 (Non-convex Resource)
        # 食物在被发现后会逃逸，速度略低于Agent最大速度
        # ============================================================
        self.food_velocities: List[Tuple[float, float]] = []  # 食物速度
        self.food_escape_enabled = True
        self.food_escape_speed = 0.6  # v0.99: 降低50% (1.2→0.6)，让Agent更容易抓到食物
        self.food_escape_range = 25.0  # 触发逃逸的距离
        self.food_escape_cooldown = 0  # 逃逸冷却期

        self._init_food()

        # 兼容: 单目标接口
        self.target_pos = self.food_positions[0] if self.food_positions else target_pos

        # ============================================================
        # v3.0: 障碍物系统
        # ============================================================
        self.n_walls = n_walls
        self.walls: List[Tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)
        if n_walls > 0:
            self._init_walls()

        # ============================================================
        # v3.0: 昼夜循环
        # ============================================================
        self.day_night_cycle = day_night_cycle
        self.current_time = 0  # 当前时间步
        self.day_length = 50   # 一天多少步
        self.is_day = True     # 当前是否为白天

        # ============================================================
        # v0.78: 季节系统
        # ============================================================
        self.seasonal_cycle = seasonal_cycle
        self.season_length = season_length
        self.base_season_length = season_length  # v11.0: 基准季节长度
        self.winter_food_multiplier = winter_food_multiplier
        self.winter_metabolic_multiplier = winter_metabolic_multiplier
        self.base_winter_metabolic_multiplier = winter_metabolic_multiplier  # v11.0: 基准冬天代谢
        self.current_season = "summer"  # 开始于夏天
        self.season_frame = 0
        self.summer_food_multiplier = 1.0  # 夏天的食物倍率

        # v0.78: 巢穴系统 (贮粮)
        self.nest_enabled = seasonal_cycle  # 有季节才需要巢穴
        self.nest_position = (width * 0.15, height * 0.15)  # 左上角
        self.nest_radius = 10.0
        self.nest_stored_food = 0  # 巢穴中存储的食物

        # ============================================================
        # v4.0: 诱饵系统 (时空不一致性挑战)
        # ============================================================
        self.n_bait = 0        # 诱饵数量
        self.bait_positions: List[Tuple[float, float]] = []
        self.bait_frequencies: List[float] = []  # 移动频率 (Hz)
        self.bait_angles: List[float] = []  # 当前角度

        # 物理参数
        self.max_speed = 2.0          # 最大速度
        self.turn_rate = 0.1          # 转向速率
        self.sensor_range = 200.0     # 传感器感知范围 (用于衰减)

        # 代谢惩罚参数
        self.metabolic_alpha = metabolic_alpha  # 每个节点的能耗
        self.metabolic_beta = metabolic_beta    # 每条边的能耗
        self.surprise_penalty = surprise_penalty  # 预测误差惩罚

        # 动态环境参数 (Malthusian Trap)
        self.base_metabolic_alpha = metabolic_alpha
        self.base_metabolic_beta = metabolic_beta
        self.base_sensor_range = 200.0
        self.generation = 0

        # ============================================================
        # v11.0: 三大突破机制参数 (2026-03-12)
        # ============================================================
        # 1. 代谢熵增: 体内能量随时间挥发
        self.energy_decay_k = energy_decay_k
        self.base_energy_decay_k = energy_decay_k
        # 2. 端口干涉: 多端口同时激活代谢成本
        self.port_interference_gamma = port_interference_gamma
        self.base_port_interference_gamma = port_interference_gamma
        # 3. 季节波动率: 环境参数扰动
        self.season_jitter = season_jitter
        self.base_season_jitter = season_jitter
        # 入库税: 贮粮入库时一次性扣除
        self.nest_tax = nest_tax
        self.base_nest_tax = nest_tax

        # ============================================================
        # v4.1: 幽灵猎物 (Blinking Prey) - 记忆测试
        # ============================================================
        self.blink_enabled = False        # 是否启用闪烁
        self.blink_on_duration = 20       # 信号持续帧数
        self.blink_off_duration = 40      # 信号消失帧数
        self.blink_signal_on = True       # 当前信号状态
        self.blink_cycle = 0              # 当前周期计数

        # v4.1: 课程学习参数
        self.blink_curriculum = True      # 是否启用课程学习
        self.blink_phase1_thresh = 50     # 阶段1阈值 (代)
        self.blink_phase2_thresh = 100    # 阶段2阈值 (代)
        self.blink_initial_easy = (50, 5) # 阶段1: 开50, 关5
        self.blink_medium = (40, 20)      # 阶段2: 开40, 关20
        self.blink_hard = (20, 40)        # 阶段3: 开20, 关40

        # ============================================================
        # v7.0: 不透明障碍物 (Opaque Obstacles)
        # ============================================================
        self.opaque_walls: List[Tuple[float, float, float, float]] = []  # 阻挡光线的墙壁
        self.n_opaque_walls = 0
        self._init_opaque_walls()

        # ============================================================
        # v7.0: 移动光源 (Moving Light Source)
        # ============================================================
        self.light_source_enabled = True
        self.light_pos = (50.0, 50.0)  # 初始光源位置
        self.light_speed = 0.5  # 光源移动速度
        self.light_direction = np.random.uniform(0, 2*np.pi)  # 移动方向
        self.light_period = 60  # 光源移动周期 (步)

        # ============================================================
        # v7.0: 信号干扰与识别反馈
        # ============================================================
        self.signal_channel_enabled = True  # 启用信号通道
        self.signal_detection_range = 30.0  # 信号感知范围

        # ============================================================
        # v7.1: 随机环境事件 (Chaos Events)
        # ============================================================
        self.chaos_enabled = True
        self.chaos_probability = 0.02  # 每步2%概率触发随机事件

        # 事件类型
        self.EVENT_FOOD_RELOCATE = "food_relocate"
        self.EVENT_WALL_APPEAR = "wall_appear"
        self.EVENT_WALL_DISAPPEAR = "wall_disappear"
        self.EVENT_LIGHT_JUMP = "light_jump"
        self.EVENT_SENSOR_NOISE = "sensor_noise"
        self.EVENT_SIGNAL_JAM = "signal_jam"

        # 当前活跃的随机事件
        self.active_events: List[str] = []
        self.sensor_noise_level = 0.0  # 传感器噪声级别
        self.signal_jam_active = False  # 信号干扰

        # ============================================================
        # v8.0 无尽边疆 - 分块世界管理器
        # ============================================================
        self.infinite_mode = False  # 是否启用无限模式
        self.chunk_manager: Optional[ChunkManager] = None
        self.sensor_range = 40.0  # v0.99: 增加33% (30→40)，更容易发现远处的食物

        # GPS坐标追踪
        self.origin_x = 0.0  # 世界原点
        self.origin_y = 0.0

        # ============================================================
        # v0.97: 三大突破机制 (专家建议)
        # ============================================================

        # 方案1: 代谢疲劳 + 安全掩体 (降维破解6步长链路)
        self.fatigue_system_enabled = False
        self.max_fatigue = 100.0        # 最大疲劳值
        self.fatigue_recovery_rate = 0.5  # 睡眠恢复速度
        self.fatigue_build_rate = 0.3     # 每步疲劳积累
        self.sleep_energy_cost = 0.02     # 睡眠时代谢消耗
        self.sleep_danger_zone = True     # 开阔地睡眠危险
        self.safe_zone_near_walls = True  # 墙拐角是安全区
        self.danger_kill_probability = 0.99  # 危险区睡眠死亡率

        # Stage 1 控制: 禁用起床饥饿和物理掉落
        self.enable_wakeup_hunger = True   # 起床饥饿: 睡眠消耗能量
        self.enable_sleep_drop = True      # 物理掉落: 睡眠时物品掉落

        # 方案2: 无聊信息素 (破解气味盲区与贴墙死循环)
        self.pheromone_enabled = False
        self.pheromone_deposit_rate = 0.1  # 停留时分泌速率
        self.pheromone_decay = 0.95        # 气味消散率
        self.pheromone_sensor_range = 5.0  # 自身气味感知范围
        self.pheromone_max = 10.0          # 最大气味浓度

        # 方案3: 夏日食物热力学 (赋予"放下"即时收益)
        self.food_thermodynamics_enabled = False
        self.food_freshness_decay = 0.995  # 平原食物新鲜度衰减
        self.shadow_zone_decay = 1.0       # 阴影区(巢穴/墙后)不衰减
        self.food_poison_threshold = 0.5   # 食物毒害阈值
        self.food_poison_damage = 20.0     # 吃腐败食物的伤害

        # 食物新鲜度追踪 (每个食物)
        self.food_freshness: List[float] = []

        # ============================================================
        # v0.98: 热力学庇护所 (Thermodynamic Sanctuary)
        # 机制: 温度场 + 食物产热 + 巢穴保温
        # 目的: 将"为冬天贮粮"降维为对"热力学梯度"的趋性
        # ============================================================
        self.thermal_sanctuary_enabled = False
        self.summer_temperature = 25.0      # 夏天环境温度 (°C)
        self.winter_temperature = -10.0     # 冬天环境温度 (°C)
        self.global_temperature = 25.0      # 当前环境温度
        self.food_heat_output = 12.0        # 食物产热能力 (°C) - 模拟发酵热
        self.food_heat_radius = 15.0        # 食物热影响半径
        self.nest_insulation = 0.02         # 巢穴保温系数 (极低散热)
        self.nest_temperature = 20.0        # 巢穴内部温度 (热岛)
        self.agent_base_temperature = 25.0  # Agent基础体温
        self.cold_damage_threshold = -5.0   # 冻伤阈值

        # 温度场矩阵 (向量化计算) - 预分配
        self._temperature_grid_resolution = 5.0  # 网格分辨率
        self._temp_grid: Optional[np.ndarray] = None
        self._temp_grid_built = False

        # ============================================================
        # v0.98: 形态计算 - 被动物理吸附
        # 机制: Agent表皮吸附特性，撞击食物自动附着
        # 目的: 剥离"抓取"和"携带"的复杂神经控制
        # ============================================================
        self.morphological_computation_enabled = False
        self.adhesion_range = 2.5           # 吸附距离阈值
        self.carry_speed_penalty = 0.7      # 携带时的速度系数 (70%速度)

        # 放电/卸货信号阈值
        self.discharge_threshold = 0.75     # 激活卸货的信号强度
        
        # ============================================================
        # v0.98: 压痕系统 - 环境记忆 (Stigmergic Friction)
        # ============================================================
        self.stigmergic_friction_enabled = False
        self.friction_grid_resolution = 2.0      # 压痕网格分辨率
        self.friction_grid: Optional[np.ndarray] = None
        self.trail_deposit_rate = 0.1            # 每次经过增加的压痕
        self.trail_decay = 0.995                 # 压痕自然衰减
        self.load_bonus = 2.0                    # 携带食物时压痕加深度数
        self.friction_bonus_max = 0.7            # 最大速度加成
        
        # ============================================================
        # v0.98: 发育相变 - 幼体保护期 (Ontogenetic Phase Transition)
        # ============================================================
        self.ontogenetic_phase_enabled = False
        self.juvenile_duration = 30              # 幼体期帧数
        self.juvenile_metabolic_rate = 0.3       # 幼体代谢率 (30%)
        self.juvenile_cold_immunity = True       # 幼体免疫寒冷
        self.phase_transition_bonus = 1.5        # 相变后的代谢跳跃

    def enable_thermal_sanctuary(self, enabled: bool = True,
                                  summer_temp: float = 25.0,
                                  winter_temp: float = -10.0,
                                  food_heat: float = 12.0,
                                  nest_insulation: float = 0.02):
        """
        启用热力学庇护所系统

        物理模型:
        - 环境温度随季节变化 (夏天25°C → 冬天-10°C)
        - 食物成为热源 (周围形成温度梯度)
        - 巢穴是保温层 (热量散失极慢)

        涌现行为:
        - Agent演化出"趋暖"本能
        - 冬天携带食物回巢 = 获取热源+食物双重奖励
        """
        self.thermal_sanctuary_enabled = enabled
        self.summer_temperature = summer_temp
        self.winter_temperature = winter_temp
        self.food_heat_output = food_heat
        self.nest_insulation = nest_insulation

        # 初始化温度网格
        self._init_temperature_grid()

        print(f"  [Thermal Sanctuary] Enabled: summer={summer_temp}°C, winter={winter_temp}°C, food_heat={food_heat}")

    def _init_temperature_grid(self):
        """初始化温度场网格"""
        nx = int(self.width / self._temperature_grid_resolution) + 1
        ny = int(self.height / self._temperature_grid_resolution) + 1
        self._temp_grid = np.full((nx, ny), self.global_temperature)
        self._temp_grid_built = False

    def _build_temperature_field(self):
        """
        构建温度场 (向量化计算)

        T(x,y) = T_env + Σ(food_heat / dist) + nest_insulation_bonus
        """
        if not self.thermal_sanctuary_enabled:
            return

        nx = int(self.width / self._temperature_grid_resolution) + 1
        ny = int(self.height / self._temperature_grid_resolution) + 1

        # 创建坐标网格
        x_coords = np.linspace(0, self.width, nx)
        y_coords = np.linspace(0, self.height, ny)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

        # 基础环境温度
        temp_field = np.full((nx, ny), self.global_temperature)

        # 食物热源叠加 (向量化)
        if self.food_positions:
            for fx, fy in self.food_positions:
                # 计算每个网格点到食物的距离
                dx = xx - fx
                dy = yy - fy
                dist = np.sqrt(dx**2 + dy**2) + 1e-6  # 避免除零
                # 温度贡献: 距离衰减
                temp_field += self.food_heat_output * np.exp(-dist / self.food_heat_radius)

        # 巢穴热岛 (保温效果)
        if self.nest_enabled:
            nx_pos, ny_pos = self.nest_position
            dx = xx - nx_pos
            dy = yy - ny_pos
            dist = np.sqrt(dx**2 + dy**2)
            # 巢穴附近温度提升
            nest_bonus = self.nest_temperature * np.exp(-dist / self.nest_radius)
            temp_field += nest_bonus * (1.0 - self.nest_insulation)

        self._temp_grid = temp_field
        self._temp_grid_built = True

    def get_temperature_at(self, x: float, y: float) -> float:
        """获取指定位置的温度 (双线性插值)"""
        if not self.thermal_sanctuary_enabled or self._temp_grid is None:
            return self.global_temperature

        # 网格坐标
        gx = x / self._temperature_grid_resolution
        gy = y / self._temperature_grid_resolution

        ix = int(gx)
        iy = int(gy)

        nx, ny = self._temp_grid.shape
        if ix >= nx - 1 or iy >= ny - 1:
            # 边界情况
            ix = min(ix, nx - 1)
            iy = min(iy, ny - 1)
            return self._temp_grid[ix, iy]

        # 双线性插值
        fx = gx - ix
        fy = gy - iy

        temp = (1-fx)*(1-fy)*self._temp_grid[ix, iy] + \
               fx*(1-fy)*self._temp_grid[ix+1, iy] + \
               (1-fx)*fy*self._temp_grid[ix, iy+1] + \
               fx*fy*self._temp_grid[ix+1, iy+1]

        return temp

    def enable_morphological_computation(self, enabled: bool = True,
                                          adhesion_range: float = 2.5,
                                          carry_speed_penalty: float = 0.7,
                                          discharge_threshold: float = 0.75):
        """
        启用形态计算 - 被动物理吸附系统

        物理模型:
        - Agent表皮有魔术贴/静电特性
        - 撞击食物自动附着 (无需神经控制)
        - 携带时增加移动阻力

        神经简化:
        - 只需一个"放电/卸货"通道
        - 当 actuator[1] > threshold 时触发卸货
        """
        self.morphological_computation_enabled = enabled
        self.adhesion_range = adhesion_range
        self.carry_speed_penalty = carry_speed_penalty
        self.discharge_threshold = discharge_threshold

        print(f"  [Morphological] Enabled: adhesion={adhesion_range}, carry_penalty={carry_speed_penalty}")

    def enable_stigmergic_friction(self, enabled: bool = True,
                                    resolution: float = 2.0,
                                    deposit_rate: float = 0.1,
                                    trail_decay: float = 0.995,
                                    load_bonus: float = 2.0):
        """
        启用压痕系统 - 环境记忆

        物理模型:
        - Agent移动时在地面留下压痕
        - 携带食物时压痕更深 (负载标记)
        - 后续Agent经过压痕时移动阻力降低
        - 形成"高速公路"效应

        涌现行为:
        - 从食物到巢穴的路径被自然踩踏出来
        - Agent只需演化"走低阻力路线"的低级反射
        """
        self.stigmergic_friction_enabled = enabled
        self.friction_grid_resolution = resolution
        self.trail_deposit_rate = deposit_rate
        self.trail_decay = trail_decay
        self.load_bonus = load_bonus

        # 初始化压痕网格
        self._init_friction_grid()

        print(f"  [Stigmergic Friction] Enabled: resolution={resolution}, deposit={deposit_rate}")

    def _init_friction_grid(self):
        """初始化压痕网格"""
        nx = int(self.width / self.friction_grid_resolution) + 1
        ny = int(self.height / self.friction_grid_resolution) + 1
        self.friction_grid = np.zeros((nx, ny))

    def _update_friction_grid(self, agent: Agent, old_x: float, old_y: float):
        """
        更新压痕网格

        1. 衰减旧压痕
        2. 添加新压痕 (基于是否携带食物)
        """
        if not self.stigmergic_friction_enabled or self.friction_grid is None:
            return

        # 1. 全局衰减
        self.friction_grid *= self.trail_decay

        # 2. 计算路径上的压痕
        # 从旧位置到新位置添加压痕
        old_gx = int(old_x / self.friction_grid_resolution)
        old_gy = int(old_y / self.friction_grid_resolution)
        new_gx = int(agent.x / self.friction_grid_resolution)
        new_gy = int(agent.y / self.friction_grid_resolution)

        nx, ny = self.friction_grid.shape
        old_gx = np.clip(old_gx, 0, nx-1)
        old_gy = np.clip(old_gy, 0, ny-1)
        new_gx = np.clip(new_gx, 0, nx-1)
        new_gy = np.clip(new_gy, 0, ny-1)

        # 线性插值添加压痕
        steps = max(abs(new_gx - old_gx), abs(new_gy - old_gy), 1)

        # 携带食物时压痕更深
        load_factor = 1.0 + agent.food_carried * self.load_bonus

        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            gx = int(old_gx + (new_gx - old_gx) * t)
            gy = int(old_gy + (new_gy - old_gy) * t)
            gx = np.clip(gx, 0, nx-1)
            gy = np.clip(gy, 0, ny-1)

            # 添加压痕 (有上限)
            self.friction_grid[gx, gy] = min(
                1.0,
                self.friction_grid[gx, gy] + self.trail_deposit_rate * load_factor
            )

    def get_friction_bonus(self, x: float, y: float) -> float:
        """
        获取位置的速度加成

        压痕越深，阻力越小，速度越快
        返回: 速度乘数 (0.3 ~ 1.0)
        """
        if not self.stigmergic_friction_enabled or self.friction_grid is None:
            return 1.0

        gx = int(x / self.friction_grid_resolution)
        gy = int(y / self.friction_grid_resolution)

        nx, ny = self.friction_grid.shape
        gx = np.clip(gx, 0, nx-1)
        gy = np.clip(gy, 0, ny-1)

        friction = self.friction_grid[gx, gy]

        # 压痕从0到1，速度加成从30%到100%
        bonus = 0.3 + friction * 0.7
        return bonus

    def enable_ontogenetic_phase(self, enabled: bool = True,
                                  juvenile_duration: int = 30,
                                  metabolic_rate: float = 0.3,
                                  cold_immunity: bool = True):
        """
        启用发育相变系统

        物理模型:
        - Agent出生时是"幼体态"
        - 幼体有物理硬壳保护:
          - 代谢率降低 (30%)
          - 免疫寒冷伤害
        - 经过juvenile_duration帧后发生"相变"
        - 相变后: 硬壳脱落，代谢飙升到150%

        涌现行为:
        - 幼体期允许大胆探索
        - 相变后面临真实选择压力
        - 给网络时间在高压环境前收敛
        """
        self.ontogenetic_phase_enabled = enabled
        self.juvenile_duration = juvenile_duration
        self.juvenile_metabolic_rate = metabolic_rate
        self.juvenile_cold_immunity = cold_immunity

        print(f"  [Ontogenetic Phase] Enabled: duration={juvenile_duration} frames, metabolic={metabolic_rate*100}%")

    def enable_fatigue_system(self, enabled: bool = True,
                              max_fatigue: float = 100.0,
                              fatigue_build_rate: float = 0.3,
                              sleep_danger_prob: float = 0.99,
                              enable_wakeup_hunger: bool = True,
                              enable_sleep_drop: bool = True):
        """启用代谢疲劳+安全掩体系统"""
        self.fatigue_system_enabled = enabled
        self.max_fatigue = max_fatigue
        self.fatigue_build_rate = fatigue_build_rate
        self.danger_kill_probability = sleep_danger_prob
        self.enable_wakeup_hunger = enable_wakeup_hunger
        self.enable_sleep_drop = enable_sleep_drop

    def enable_pheromone_system(self, enabled: bool = True,
                                deposit_rate: float = 0.1,
                                decay: float = 0.95):
        """启用无聊信息素系统"""
        self.pheromone_enabled = enabled
        self.pheromone_deposit_rate = deposit_rate
        self.pheromone_decay = decay

    def enable_food_thermodynamics(self, enabled: bool = True,
                                   freshness_decay: float = 0.995,
                                   poison_damage: float = 20.0):
        """启用夏日食物热力学系统"""
        self.food_thermodynamics_enabled = enabled
        self.food_freshness_decay = freshness_decay
        self.food_poison_damage = poison_damage
        # 初始化食物新鲜度
        self.food_freshness = [1.0] * len(self.food_positions)

    def enable_infinite_mode(self, seed: int = 42, sensor_range: float = 30.0):
        """
        启用无尽边疆模式 (v8.0)

        参数:
            seed: 世界生成种子
            sensor_range: Agent传感器范围
        """
        self.infinite_mode = True
        self.sensor_range = sensor_range
        self.chunk_manager = ChunkManager(seed=seed)

        # 禁用旧的固定食物系统
        self.food_positions = []
        self.n_food = 0
        self.respawn_food = False

        print(f"  [Infinite World] Enabled with seed={seed}, chunk_size={ChunkManager.CHUNK_SIZE}")

    def get_chunk_manager(self) -> Optional[ChunkManager]:
        """获取区块管理器"""
        return self.chunk_manager

    def set_bait(self, n_bait: int, freq_range: Tuple[float, float] = (0.1, 0.5)) -> None:
        """
        设置诱饵 (v4.0)

        诱饵特性:
        - 气味/外观与真实食物完全一样
        - 只能通过观察移动频率来区分
        - 低频移动 = 食物, 高频移动 = 诱饵
        - 强迫"慢思考模块"的发展
        """
        self.n_bait = n_bait
        self.bait_positions = []
        self.bait_frequencies = []
        self.bait_angles = []

        margin = 15.0
        for _ in range(n_bait):
            pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )
            self.bait_positions.append(pos)
            # 随机频率: 低频(~0.1Hz) = 可能食物, 高频(~0.5Hz) = 诱饵
            self.bait_frequencies.append(
                np.random.uniform(freq_range[0], freq_range[1])
            )
            self.bait_angles.append(np.random.uniform(0, 2 * np.pi))

    def set_blink(
        self,
        enabled: bool = True,
        on_duration: int = 20,
        off_duration: int = 40
    ) -> None:
        """
        设置幽灵猎物 (Blinking Prey) - v4.1 记忆测试

        物理机制:
        - 食物信号周期性地"闪烁": 发信号20帧, 静默40帧
        - 食物物理位置不变, 但传感器信号周期消失

        演化压力:
        - 无记忆个体: 信号消失时停止或打转
        - 有DELAY记忆: 在静默期继续"盲开"

        参数:
            enabled: 是否启用闪烁
            on_duration: 信号持续帧数
            off_duration: 信号消失帧数
        """
        self.blink_enabled = enabled
        self.blink_on_duration = on_duration
        self.blink_off_duration = off_duration
        self.blink_signal_on = True
        self.blink_cycle = 0

    def update_blink_difficulty(self, generation: int) -> None:
        """
        根据代数更新闪烁难度 (课程学习)

        阶段1 (0-50代): on=50, off=5   - 简单适应
        阶段2 (50-100代): on=40, off=20 - 中等难度
        阶段3 (100+代): on=20, off=40  - 终极测试
        """
        if not self.blink_curriculum:
            return

        if generation < self.blink_phase1_thresh:
            # 阶段1: 轻微闪烁，只需短暂记忆
            self.blink_on_duration, self.blink_off_duration = self.blink_initial_easy
            phase = 1
        elif generation < self.blink_phase2_thresh:
            # 阶段2: 中等闪烁
            self.blink_on_duration, self.blink_off_duration = self.blink_medium
            phase = 2
        else:
            # 阶段3: 终极挑战
            self.blink_on_duration, self.blink_off_duration = self.blink_hard
            phase = 3

        # 重置闪烁周期以平滑过渡
        self.blink_cycle = 0
        self.blink_signal_on = True

    def _update_blink(self) -> None:
        """更新闪烁状态"""
        if not self.blink_enabled:
            return

        self.blink_cycle += 1
        cycle_position = self.blink_cycle % (self.blink_on_duration + self.blink_off_duration)

        if cycle_position < self.blink_on_duration:
            self.blink_signal_on = True
        else:
            self.blink_signal_on = False

    def _compute_niche_preference(self, agent: Agent) -> None:
        """
        v6.0 GAIA: 计算Agent的生态位偏好

        根据5个端口的激活程度确定生态位:
        - predator: 高OFFENSE
        - defender: 高DEFENSE
        - repairer: 高REPAIR
        - deceiver: 高SIGNAL
        - general: 平衡
        """
        total = (agent.port_offense + agent.port_defense +
                agent.port_repair + agent.port_signal + 0.001)

        off_pct = agent.port_offense / total
        def_pct = agent.port_defense / total
        rep_pct = agent.port_repair / total
        sig_pct = agent.port_signal / total

        if off_pct > 0.4:
            agent.niche_type = "predator"
        elif def_pct > 0.4:
            agent.niche_type = "defender"
        elif rep_pct > 0.4:
            agent.niche_type = "repairer"
        elif sig_pct > 0.4:
            agent.niche_type = "deceiver"
        else:
            agent.niche_type = "general"

        # 代谢效率 (ROI)
        if agent.energy_spent > 0:
            agent.metabolic_efficiency = agent.energy_gained / agent.energy_spent
        else:
            agent.metabolic_efficiency = 0.0

        # 预测深度
        genome = agent.genome
        predictor_count = sum(1 for n in genome.nodes.values() if n.node_type == NodeType.PREDICTOR)
        delay_count = sum(1 for n in genome.nodes.values() if n.node_type == NodeType.DELAY)
        agent.prediction_horizon = predictor_count * 1.5 + delay_count * 1.0

    def _update_bait(self) -> None:
        """更新诱饵位置 (模拟移动)"""
        for i in range(self.n_bait):
            # 基于频率移动
            freq = self.bait_frequencies[i]
            # 每 freq 步移动一次
            if self.current_time % int(1.0 / max(freq, 0.01)) == 0:
                self.bait_angles[i] += np.random.uniform(-0.5, 0.5)
                speed = 0.5  # 缓慢移动

                new_x = self.bait_positions[i][0] + np.cos(self.bait_angles[i]) * speed
                new_y = self.bait_positions[i][1] + np.sin(self.bait_angles[i]) * speed

                # 边界处理 (环绕)
                new_x = new_x % self.width
                new_y = new_y % self.height

                self.bait_positions[i] = (new_x, new_y)

    def get_bait_sensor_value(self, agent_x: float, agent_y: float, agent_theta: float) -> float:
        """
        计算诱饵传感器值
        - 气味与食物完全一样
        - 位置是动态的
        - Agent 需要通过观察移动模式来判断
        """
        if self.n_bait == 0:
            return 0.0

        # 找到最近的诱饵
        min_dist = float('inf')
        for bait_pos in self.bait_positions:
            dist = np.sqrt((agent_x - bait_pos[0])**2 + (agent_y - bait_pos[1])**2)
            min_dist = min(min_dist, dist)

        if min_dist > self.sensor_range:
            return 0.0

        # 气味强度与距离成反比
        return self.sensor_range / (min_dist + 1.0)

    def add_agent(self, agent: Agent) -> None:
        """添加智能体到环境"""
        self.agents.append(agent)

    def _update_food_escape(self, active_agent: Agent) -> None:
        """
        v9.1 (优化版): 逃逸食物逻辑
        - 快速退出路径
        - 距离阈值过滤
        - NumPy向量化
        """
        # 快速退出条件
        if self.food_escape_cooldown > 0:
            self.food_escape_cooldown -= 1
            return
        
        if len(self.food_positions) == 0:
            return
        
        alive_agents = [a for a in self.agents if a.is_alive]
        if len(alive_agents) == 0:
            return
        
        # 优化: 只有当有活跃Agent在食物附近时才计算
        # 使用向量化距离计算
        agent_x = np.array([a.x for a in alive_agents], dtype=np.float64)
        agent_y = np.array([a.y for a in alive_agents], dtype=np.float64)
        
        food_x = np.array([p[0] for p in self.food_positions], dtype=np.float64)
        food_y = np.array([p[1] for p in self.food_positions], dtype=np.float64)
        
        # 向量化距离矩阵
        diff_x = food_x[:, np.newaxis] - agent_x
        diff_y = food_y[:, np.newaxis] - agent_y
        dists = np.sqrt(diff_x**2 + diff_y**2)
        
        # 快速路径: 找最小距离
        min_dists = np.min(dists, axis=1)
        
        # 只处理范围内的食物
        need_escape = min_dists < self.food_escape_range
        
        if not np.any(need_escape):
            # 减速所有食物
            vx = np.array([v[0] for v in self.food_velocities])
            vy = np.array([v[1] for v in self.food_velocities])
            vx *= 0.95
            vy *= 0.95
            self.food_velocities = list(zip(vx, vy))
            return
        
        # 需要逃逸的食物
        escape_indices = np.where(need_escape)[0]
        
        for i in escape_indices:
            nearest_idx = np.argmin(dists[i])
            nearest_x = agent_x[nearest_idx]
            nearest_y = agent_y[nearest_idx]
            
            fx, fy = food_x[i], food_y[i]
            
            # 逃逸方向 (远离最近Agent)
            dx = fx - nearest_x
            dy = fy - nearest_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx, dy = dx / dist, dy / dist
            
            # 更新速度
            current_vx, current_vy = self.food_velocities[i]
            new_vx = current_vx * 0.8 + dx * self.food_escape_speed * 0.2
            new_vy = current_vy * 0.8 + dy * self.food_escape_speed * 0.2
            
            # 移动食物
            new_fx = fx + new_vx
            new_fy = fy + new_vy
            
            # 边界反弹
            if new_fx < 5 or new_fx > self.width - 5:
                new_vx *= -1
                new_fx = np.clip(new_fx, 5, self.width - 5)
            if new_fy < 5 or new_fy > self.height - 5:
                new_vy *= -1
                new_fy = np.clip(new_fy, 5, self.height - 5)
            
            self.food_positions[i] = (new_fx, new_fy)
            self.food_velocities[i] = (new_vx, new_vy)

    def _init_food(self) -> None:
        """初始化食物位置 (随机分布，避开边缘和障碍物)"""
        self.food_positions = []
        self.food_velocities = []  # v9.0: 初始化速度
        margin = 15.0
        for _ in range(self.n_food):
            pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )
            self.food_positions.append(pos)
            self.food_velocities.append((0.0, 0.0))  # 初始静止

    def _init_walls(self) -> None:
        """初始化障碍物 (随机线段)"""
        self.walls = []
        for _ in range(self.n_walls):
            # 随机起点和终点
            x1 = np.random.uniform(10, self.width - 10)
            y1 = np.random.uniform(10, self.height - 10)
            length = np.random.uniform(10, 30)
            angle = np.random.uniform(0, 2 * np.pi)
            x2 = x1 + length * np.cos(angle)
            y2 = y1 + length * np.sin(angle)

            # 确保在边界内
            x2 = max(5, min(self.width - 5, x2))
            y2 = max(5, min(self.height - 5, y2))

            self.walls.append((x1, y1, x2, y2))

    def _init_opaque_walls(self) -> None:
        """
        v7.0: 初始化不透明障碍物

        这些墙壁会阻挡:
        - 食物传感器信号
        - 光源传感器信号
        迫使Agent演化空间记忆来追踪物体位置
        """
        self.opaque_walls = []
        # 默认创建3个不透明墙壁，形成复杂的迷宫结构
        self.n_opaque_walls = 3

        # 固定结构: 3条互相遮挡的墙
        # 墙1: 垂直墙，左侧
        self.opaque_walls.append((30.0, 10.0, 30.0, 50.0))
        # 墙2: 水平墙，上方
        self.opaque_walls.append((30.0, 50.0, 70.0, 50.0))
        # 墙3: 垂直墙，右侧
        self.opaque_walls.append((70.0, 30.0, 70.0, 70.0))

    def _raycast_to_target(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> bool:
        """
        v7.0: 射线投射检测 - 检查起点到终点是否被不透明墙阻挡

        参数:
            start: 起点 (x, y)
            end: 目标点 (x, y)

        返回:
            True if 视线被阻挡, False if 可见
        """
        x0, y0 = start
        x1, y1 = end

        # 采样检测点 (简化版: 检测线段中点)
        dx = x1 - x0
        dy = y1 - y0
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1.0:
            return False  # 非常近，直接可见

        # 检测线段是否与墙壁相交
        for (wx1, wy1, wx2, wy2) in self.opaque_walls:
            if self._line_intersects(x0, y0, x1, y1, wx1, wy1, wx2, wy2):
                return True  # 被阻挡

        return False  # 畅通

    def _line_intersects(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
        x4: float, y4: float
    ) -> bool:
        """
        线段相交检测
        """
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        A = (x1, y1)
        B = (x2, y2)
        C = (x3, y3)
        D = (x4, y4)

        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def _update_light_source(self) -> None:
        """
        v7.0: 更新移动光源位置

        光源周期性移动，迫使Agent:
        1. 记住光源的历史位置
        2. 推断光源的运动趋势
        3. 使用DELAY回路来预测光源位置
        """
        if not self.light_source_enabled:
            return

        self.current_time += 1

        # 每light_period步改变方向
        if self.current_time % self.light_period == 0:
            self.light_direction = np.random.uniform(0, 2 * np.pi)

        # 移动光源
        dx = np.cos(self.light_direction) * self.light_speed
        dy = np.sin(self.light_direction) * self.light_speed

        new_x = self.light_pos[0] + dx
        new_y = self.light_pos[1] + dy

        # 边界反弹
        if new_x < 10 or new_x > self.width - 10:
            self.light_direction = np.pi - self.light_direction
            new_x = np.clip(new_x, 10, self.width - 10)

        if new_y < 10 or new_y > self.height - 10:
            self.light_direction = -self.light_direction
            new_y = np.clip(new_y, 10, self.height - 10)

        self.light_pos = (new_x, new_y)

    def _trigger_chaos_event(self) -> None:
        """
        v7.1: 随机环境事件 - 迫使Agent适应变化

        事件类型:
        - food_relocate: 食物随机移动到新位置
        - wall_appear: 随机生成新墙壁
        - wall_disappear: 移除一个现有墙壁
        - light_jump: 光源瞬移到随机位置
        - sensor_noise: 传感器添加高斯噪声
        - signal_jam: 信号通道暂时失效

        演化压力:
        - Agent需要快速适应新环境
        - 长期记忆(DELAY)在事件后仍有用
        - 学会忽略虚假传感器信号
        """
        if not self.chaos_enabled:
            return

        # 2%概率触发事件
        if np.random.random() > self.chaos_probability:
            return

        # 选择随机事件
        event_type = np.random.choice([
            self.EVENT_FOOD_RELOCATE,
            self.EVENT_WALL_APPEAR,
            self.EVENT_WALL_DISAPPEAR,
            self.EVENT_LIGHT_JUMP,
            self.EVENT_SENSOR_NOISE,
            self.EVENT_SIGNAL_JAM
        ])

        if event_type == self.EVENT_FOOD_RELOCATE:
            # 随机移动一个食物
            if self.food_positions:
                idx = np.random.randint(len(self.food_positions))
                margin = 15.0
                self.food_positions[idx] = (
                    np.random.uniform(margin, self.width - margin),
                    np.random.uniform(margin, self.height - margin)
                )
                # 更新target_pos为最近的食物
                self.target_pos = self.food_positions[0]

        elif event_type == self.EVENT_WALL_APPEAR:
            # 添加新墙壁 (最多5个不透明墙)
            if len(self.opaque_walls) < 5:
                x1 = np.random.uniform(20, self.width - 20)
                y1 = np.random.uniform(20, self.height - 20)
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.uniform(15, 25)
                x2 = x1 + length * np.cos(angle)
                y2 = y1 + length * np.sin(angle)
                x2 = np.clip(x2, 10, self.width - 10)
                y2 = np.clip(y2, 10, self.height - 10)
                self.opaque_walls.append((x1, y1, x2, y2))

        elif event_type == self.EVENT_WALL_DISAPPEAR:
            # 移除一个墙壁
            if len(self.opaque_walls) > 1:
                idx = np.random.randint(len(self.opaque_walls))
                self.opaque_walls.pop(idx)

        elif event_type == self.EVENT_LIGHT_JUMP:
            # 光源瞬移
            margin = 20.0
            self.light_pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )
            self.light_direction = np.random.uniform(0, 2 * np.pi)

        elif event_type == self.EVENT_SENSOR_NOISE:
            # 传感器噪声 (持续10步)
            self.sensor_noise_level = 0.5  # 50%噪声

        elif event_type == self.EVENT_SIGNAL_JAM:
            # 信号干扰 (持续15步)
            self.signal_jam_active = True

        # 记录事件
        self.active_events.append(event_type)

    def _update_chaos_effects(self) -> None:
        """
        v7.1: 更新随机事件的后续效果
        """
        # 衰减传感器噪声
        if self.sensor_noise_level > 0:
            self.sensor_noise_level *= 0.9  # 每步衰减
            if self.sensor_noise_level < 0.01:
                self.sensor_noise_level = 0.0

        # 结束信号干扰
        if self.signal_jam_active:
            if np.random.random() < 0.1:  # 10%概率结束
                self.signal_jam_active = False

    # ============================================================
    # v0.97: 三大突破机制
    # ============================================================

    def _update_fatigue_system(self, agent: Agent) -> None:
        """
        核心改造一：渐进式虚弱

        - 疲劳度只影响速度，不强制睡眠
        - 静止3帧后开始恢复疲劳
        - 不硬编码行为，只改变物理规则
        """
        if not self.fatigue_system_enabled:
            return

        # 获取Agent当前速度（用于检测是否静止）
        speed = getattr(agent, 'speed', 0.0)

        # 活动时积累疲劳
        if speed > 0.1:  # 移动中
            agent.fatigue = min(self.max_fatigue, agent.fatigue + self.fatigue_build_rate)
            agent.stationary_frames = 0  # 重置静止计数
            agent._in_sleep_recovery = False  # 重置睡眠恢复标志
        else:  # 静止
            agent.stationary_frames = getattr(agent, 'stationary_frames', 0) + 1

            # 静止3帧后开始恢复疲劳 (主动睡眠)
            if agent.stationary_frames >= 3:
                # 记录恢复前的疲劳值
                old_fatigue = agent.fatigue

                # 记录睡眠尝试 (刚开始恢复疲劳时)
                if not getattr(agent, '_in_sleep_recovery', False):
                    agent.sleep_cycles = getattr(agent, 'sleep_cycles', 0) + 1
                    agent._in_sleep_recovery = True

                agent.fatigue = max(0, agent.fatigue - self.fatigue_recovery_rate * 3)  # 恢复更快

                # 检测睡眠周期完成 (从高疲劳恢复到0)
                if old_fatigue > self.max_fatigue * 0.5 and agent.fatigue <= 0:
                    agent._in_sleep_recovery = False  # 重置标志

                # 核心改造二：起床饥饿 (5帧缓冲期)
                # 睡觉时不扣能量，醒来后5帧内逐渐流失
                if self.enable_wakeup_hunger:
                    # 检测是否刚醒来（疲劳刚降到0）
                    if old_fatigue > 5 and agent.fatigue <= 0:
                        # 激活起床低血糖缓冲：5帧内逐渐扣能量
                        agent._wakeup_buffer = 5  # 5帧缓冲
                        agent._just_woke_up = True

                    # 缓冲期内每帧扣能量
                    if hasattr(agent, '_wakeup_buffer') and agent._wakeup_buffer > 0:
                        agent.internal_energy -= 8.0  # 每帧扣8点
                        agent._wakeup_buffer -= 1
                        if agent._wakeup_buffer == 0:
                            # 缓冲结束，确保不死透
                            agent.internal_energy = max(agent.internal_energy, 15.0)

                # 核心改造三：物理掉落 (零距离+气味爆发)
                if self.enable_sleep_drop and agent.food_carried > 0:
                    # 食物掉在Agent当前坐标（零距离，确保醒来就能吃）
                    ax, ay = agent.x, agent.y

                    # 真正在环境中添加一个食物（贮粮）
                    self.food_positions.append((ax, ay))
                    self.food_velocities.append((0.0, 0.0))
                    self.food_freshness.append(1.0)

                    # v11.0: 入库税 - 贮粮时一次性扣除
                    stored = agent.food_carried
                    tax = int(stored * self.nest_tax) if self.nest_tax > 0 else 0
                    stored_after_tax = stored - tax
                    
                    # 增加贮粮计数
                    agent.food_stored += stored_after_tax

                    # 标记行为耦合：睡觉时携带食物
                    if agent.food_carried > 0:
                        agent.has_slept_with_food = True

                    # 释放高浓度气味标记（局部气味爆发）
                    # 这会在小范围内提供强烈的嗅觉信号
                    if not hasattr(self, 'scent_bursts'):
                        self.scent_bursts = []
                    self.scent_bursts.append({
                        'x': ax, 'y': ay,
                        'intensity': 50.0,  # 极高浓度
                        'frames': 30        # 持续30帧
                    })

                    agent.food_carried = 0

        # 计算疲劳对速度的影响 (渐进式虚弱)
        # 疲劳度越高，移动速度越慢
        fatigue_ratio = agent.fatigue / self.max_fatigue if self.max_fatigue > 0 else 0
        agent.fatigue_ratio = fatigue_ratio  # 供其他地方使用

    def _is_in_safe_zone(self, x: float, y: float) -> bool:
        """
        检查坐标是否在安全区 (墙拐角附近)
        """
        if not self.walls:
            return False

        for wx1, wy1, wx2, wy2 in self.walls:
            # L型墙拐角检测
            if wx1 == wx2:  # 垂直墙
                corner_x, corner_y = wx1, wy1
                if abs(x - corner_x) < 8 and abs(y - corner_y) < 8:
                    return True
            else:  # 水平墙
                corner_x, corner_y = wx1, wy1
                if abs(x - corner_x) < 8 and abs(y - corner_y) < 8:
                    return True

        # 巢穴区域也是安全的
        if self.nest_enabled:
            nx, ny = self.nest_position
            if np.sqrt((x - nx)**2 + (y - ny)**2) < self.nest_radius:
                return True

        return False

    def _update_pheromone_system(self, agent: Agent) -> None:
        """
        方案2: 无聊信息素

        机制:
        - Agent原地不动时分泌"废气"
        - 气味会随时间消散
        - Agent演化出厌恶自身气味的神经连接后会后退
        """
        if not self.pheromone_enabled:
            return

        # 计算速度
        speed = getattr(agent, 'speed', 0.0)

        if speed < 0.1:  # 几乎静止
            agent.stationary_frames += 1
            # 分泌信息素
            agent.pheromone_level = min(
                self.pheromone_max,
                agent.pheromone_level + self.pheromone_deposit_rate
            )
        else:
            agent.stationary_frames = 0
            # 移动时气味衰减
            agent.pheromone_level *= self.pheromone_decay

        # 感知自身气味 (传感器输入)
        agent.pheromone_sensor = agent.pheromone_level

    def _update_food_thermodynamics(self) -> None:
        """
        方案3: 夏日食物热力学

        机制:
        - 平原上的食物暴露在阳光下，新鲜度衰减
        - 阴影区(墙后/巢穴)食物保鲜
        - 食物腐败后有毒
        """
        if not self.food_thermodynamics_enabled:
            return

        # 确保新鲜度列表长度匹配
        if len(self.food_freshness) != len(self.food_positions):
            self.food_freshness = [1.0] * len(self.food_positions)

        for i, (fx, fy) in enumerate(self.food_positions):
            # 检查是否在阴影区
            in_shadow = self._is_in_shadow_zone(fx, fy)

            if in_shadow:
                # 阴影区不腐败
                self.food_freshness[i] = 1.0
            else:
                # 平原上腐败
                self.food_freshness[i] *= self.food_freshness_decay

    def _is_in_shadow_zone(self, x: float, y: float) -> bool:
        """
        检查坐标是否在阴影区 (墙后或巢穴内)
        """
        # 巢穴内
        if self.nest_enabled:
            nx, ny = self.nest_position
            if np.sqrt((x - nx)**2 + (y - ny)**2) < self.nest_radius:
                return True

        # 墙后区域
        if self.walls:
            for wx1, wy1, wx2, wy2 in self.walls:
                if wx1 == wx2:  # 垂直墙
                    # 墙的左侧是阴影区
                    if x < wx1 and min(wy1, wy2) - 10 < y < max(wy1, wy2) + 10:
                        return True
                else:  # 水平墙
                    # 墙的上方是阴影区
                    if y < wy1 and min(wx1, wx2) - 10 < x < max(wx1, wx2) + 10:
                        return True

        return False

    def _apply_food_poison_damage(self, agent: Agent) -> None:
        """
        检查并应用食物毒害伤害
        """
        if not self.food_thermodynamics_enabled:
            return

        # 检查正在吃的食物是否腐败
        if hasattr(agent, 'just_ate') and agent.just_ate:
            # 找到最近的食物
            min_dist = float('inf')
            eaten_idx = -1
            for i, (fx, fy) in enumerate(self.food_positions):
                dist = np.sqrt((agent.x - fx)**2 + (agent.y - fy)**2)
                if dist < min_dist:
                    min_dist = dist
                    eaten_idx = i

            if eaten_idx >= 0 and self.food_freshness[eaten_idx] < self.food_poison_threshold:
                # 食物已腐败: 伤害
                agent.internal_energy -= self.food_poison_damage
                agent.ate_poisoned = True

    # ============================================================
    # v0.98: 热力学庇护所 - 温度场更新
    # ============================================================

    def _update_thermal_system(self) -> None:
        """
        更新温度场系统

        流程:
        1. 根据季节更新全局环境温度
        2. 重建温度场 (含食物热源和巢穴热岛)
        """
        if not self.thermal_sanctuary_enabled:
            return

        # 1. 更新环境温度 (基于季节)
        if self.seasonal_cycle:
            if self.current_season == "summer":
                # 夏天: 平滑过渡到高温
                self.global_temperature += (self.summer_temperature - self.global_temperature) * 0.05
            else:
                # 冬天: 平滑过渡到低温
                self.global_temperature += (self.winter_temperature - self.global_temperature) * 0.05
        else:
            # 无季节系统时，使用固定温度
            self.global_temperature = self.summer_temperature

        # 2. 重建温度场 (每10帧更新一次以节省计算)
        if self.step_count % 10 == 0:
            self._build_temperature_field()

    def _apply_temperature_effects(self, agent: Agent) -> None:
        """
        对Agent应用温度效果

        物理效果:
        - 寒冷惩罚: 体温低于阈值时额外代谢消耗
        - 温暖奖励: 在食物/巢穴附近时代谢效率提升
        """
        if not self.thermal_sanctuary_enabled:
            return

        # 获取Agent位置的温度
        local_temp = self.get_temperature_at(agent.x, agent.y)

        # 更新Agent体温 (向环境温度趋近)
        agent.body_temperature = agent.body_temperature * 0.95 + local_temp * 0.05

        # 计算温度舒适度 (-1 到 1)
        ideal_temp = 25.0
        comfort = 1.0 - abs(agent.body_temperature - ideal_temp) / 30.0
        # 优化: 用原生Python
        comfort = max(-1.0, min(1.0, comfort))

        # 更新温度传感器 [左温度, 右温度, 舒适度]
        # 简单实现: 使用全局温度 + 局部偏差
        left_temp = local_temp + np.random.uniform(-2, 2)
        right_temp = local_temp + np.random.uniform(-2, 2)

        # 优化: 用原生Python代替循环内的np.clip
        agent.temperature_sensors = [
            max(0.0, min(1.0, (left_temp + 30) / 60)),
            max(0.0, min(1.0, (right_temp + 30) / 60)),
            max(0.0, min(1.0, (comfort + 1) / 2))
        ]
        
        # 寒冷惩罚 (幼体可能免疫)
        is_juvenile_immune = (
            self.ontogenetic_phase_enabled and 
            agent.developmental_stage == "juvenile" and 
            self.juvenile_cold_immunity
        )
        
        if local_temp < self.cold_damage_threshold and not is_juvenile_immune:
            # 额外代谢消耗: 寒冷逼迫寻找热源
            cold_penalty = (self.cold_damage_threshold - local_temp) * 0.5
            agent.internal_energy -= cold_penalty

        # 温暖奖励 (在食物或巢穴附近)
        # 这里的奖励是隐式的: 温暖区域 = 食物/巢穴附近 = 更可能存活

    # ============================================================
    # v0.98: 形态计算 - 物理吸附检测
    # ============================================================

    def _check_adhesion_collision(self, agent: Agent) -> bool:
        """
        形态计算: 检查物理吸附碰撞

        机制:
        - Agent接近食物时自动吸附 (无需神经控制)
        - 食物被"粘"在Agent身上
        - Agent可以继续移动，但速度降低
        """
        if not self.morphological_computation_enabled:
            return False

        for i, food_pos in enumerate(self.food_positions):
            dx = agent.x - food_pos[0]
            dy = agent.y - food_pos[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < self.adhesion_range:
                # 物理吸附发生！
                agent.food_carried += 1

                # 移除环境中的食物 (被吸附走)
                self.food_positions.pop(i)
                if i < len(self.food_freshness):
                    self.food_freshness.pop(i)
                if i < len(self.food_velocities):
                    self.food_velocities.pop(i)

                return True

        return False

    def _check_discharge_and_store(self, agent: Agent, actuator_outputs: np.ndarray) -> None:
        """
        形态计算: 检查放电/卸货信号

        机制:
        - 只需一个简单的神经通道
        - 当 actuator[1] > threshold 时触发卸货
        - 只有在巢穴附近才能成功存储
        """
        if not self.morphological_computation_enabled:
            return

        # 检查是否有携带食物
        if agent.food_carried == 0:
            return

        # 检查放电信号
        if len(actuator_outputs) > 1 and actuator_outputs[1] > self.discharge_threshold:
            # 检查是否在巢穴附近 - 使用环形世界距离
            if self.nest_enabled:
                dist = self._toroidal_distance(
                    agent.x, agent.y, 
                    self.nest_position[0], self.nest_position[1]
                )

                if dist < self.nest_radius:
                    # 成功存储！
                    stored = agent.food_carried
                    agent.food_stored += stored
                    agent.food_carried = 0

                    # 添加能量奖励 (存储后的即时收益)
                    bonus = stored * self.food_energy * 0.3
                    agent.internal_energy = min(
                        agent.max_energy,
                        agent.internal_energy + bonus
                    )

    def _apply_sensor_noise(self, sensor_values: np.ndarray) -> np.ndarray:
        """
        v7.1: 对传感器添加高斯噪声 (优化版)
        """
        if self.sensor_noise_level > 0:
            noise = np.random.normal(0, self.sensor_noise_level, sensor_values.shape)
            sensor_values = sensor_values + noise
            # 优化: 批量clip比循环快
            sensor_values = np.clip(sensor_values, 0.0, 1.0)
        return sensor_values

    def _get_agent_signal(self, agent: Agent) -> float:
        """
        v7.0: 获取Agent的信号值 (PORT_SIGNAL)

        返回Agent的可视化信号强度，用于其他Agent的社会感知
        """
        return getattr(agent, 'port_signal', 0.0)

    def _detect_nearby_signals(self, agent: Agent) -> np.ndarray:
        """
        v7.0: 检测附近Agent的信号

        社会传感器返回:
        - 最近3个Agent的信号强度
        - 如果没有邻居，返回[0, 0, 0]

        这允许演化出:
        - 捕食者伪装: 模仿食物的信号
        - 亲族识别: 识别同类的信号模式
        """
        # v7.1: 信号干扰时返回全零
        if self.signal_jam_active:
            return np.array([0.0, 0.0, 0.0])

        signals = []

        for other in self.agents:
            if other is agent or not other.is_alive:
                continue

            # 计算距离
            dx = other.x - agent.x
            dy = other.y - agent.y
            dist = np.sqrt(dx**2 + dy**2)

            if dist < self.signal_detection_range:
                # 获取该Agent的信号强度
                signal_strength = self._get_agent_signal(other)
                # 距离衰减
                signal_strength *= (1.0 - dist / self.signal_detection_range)
                signals.append(signal_strength)

        # 填充到3个槽位
        while len(signals) < 3:
            signals.append(0.0)

        return np.array(signals[:3])

    def _check_wall_collision(self, x: float, y: float, new_x: float, new_y: float) -> Tuple[float, float]:
        """检查并处理与墙壁的碰撞"""
        for (x1, y1, x2, y2) in self.walls:
            # 线段与移动路径的交点检测 (简化版: 矩形检测)
            margin = 2.0  # 墙壁厚度

            # 检查是否穿过墙壁
            min_x = min(x1, x2) - margin
            max_x = max(x1, x2) + margin
            min_y = min(y1, y2) - margin
            max_y = max(y1, y2) + margin

            if min_x < new_x < max_x and min_y < new_y < max_y:
                # 发生碰撞，原地不动
                return x, y

        return new_x, new_y

    def _respawn_food(self) -> None:
        """食物被吃掉后重生 (v10.1 空间风险梯度版)"""
        if not self.respawn_food:
            return

        # v0.78: 冬天不生成食物
        if self.seasonal_cycle and self.current_season == "winter":
            # 冬天没有新食物
            return

        # v10.1: 空间风险梯度 - 巢穴附近减少食物，野外增加
        if self.nest_enabled and hasattr(self, 'nest_position'):
            nx, ny = self.nest_position
            nest_radius = self.nest_radius
            
            # 在巢穴安全区外生成
            margin = 15.0
            max_attempts = 10
            
            for _ in range(max_attempts):
                x = np.random.uniform(margin, self.width - margin)
                y = np.random.uniform(margin, self.height - margin)
                
                dist_to_nest = np.sqrt((x - nx)**2 + (y - ny)**2)
                
                # 巢穴附近生成概率降低 (危险区)
                if dist_to_nest < nest_radius:
                    if np.random.random() < 0.3:  # 70%概率不在巢穴附近生成
                        continue
                
                # 找到被吃掉的最近食物位置替换
                if self.food_positions:
                    idx = np.random.randint(len(self.food_positions))
                    self.food_positions[idx] = (x, y)
                return
            
            # 如果上面的逻辑没返回，最后随机生成
            pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )
        else:
            # 原始逻辑
            margin = 15.0
            pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )
        
        # 找到被吃掉的最近食物位置替换
        if self.food_positions:
            idx = np.random.randint(len(self.food_positions))
            self.food_positions[idx] = pos

    def _check_food_collision(self, agent: Agent) -> bool:
        """
        检查 Agent 是否吃到食物

        v5.4: 应用能量吸收率 η_plant (80%)
        v1.3: 使用环形世界距离
        """
        eat_distance = 3.0  # 吃到的距离阈值

        for i, food_pos in enumerate(self.food_positions):
            dist = self._toroidal_distance(agent.x, agent.y, food_pos[0], food_pos[1])

            if dist < eat_distance:
                # 吃到食物!
                # v10.4: 去中心化携带决策
                # 废除硬编码阈值，让神经网络通过演化决定
                # 默认行为：采集到食物就携带回巢
                if self.immediate_eating:
                    energy_ratio = agent.internal_energy / agent.max_energy if agent.max_energy > 0 else 0
                    
                    # v10.4: 极简逻辑 - 只有能量极低时才吃
                    # 否则一律携带回巢贮粮
                    if energy_ratio < 0.20:  # 能量低于20%才直接吃
                        # 能量紧急：直接吃掉
                        agent.food_eaten += 1
                        food_e = self.food_energy
                        
                        absorbed_energy = food_e * agent.eta_plant
                        wasted_energy = food_e * (1 - agent.eta_plant)
                        
                        agent.internal_energy = min(
                            agent.max_energy,
                            agent.internal_energy + absorbed_energy
                        )
                        agent.energy_gained += absorbed_energy
                        agent.energy_wasted += wasted_energy
                    else:
                        # 能量充足：携带回巢贮粮
                        if self.nest_enabled:
                            agent.food_carried += 1
                        else:
                            agent.food_eaten += 1
                elif self.nest_enabled:
                    # 原有巢穴模式 - 携带食物回巢
                    agent.food_carried += 1
                else:
                    # 原有模式 - 直接吃掉
                    agent.food_eaten += 1
                    
                    # v5.4/v5.7 能量守恒: 应用吸收率 η_plant
                    absorbed_energy = self.food_energy * agent.eta_plant
                    wasted_energy = self.food_energy * (1 - agent.eta_plant)
                    
                    agent.internal_energy = min(
                        agent.max_energy,
                        agent.internal_energy + absorbed_energy
                    )
                    agent.energy_gained += absorbed_energy
                    agent.energy_wasted += wasted_energy

                # 食物重生
                self._respawn_food()
                return True

        return False

    def _attempt_energy_theft(self, agent: Agent, actuator_outputs: np.ndarray) -> None:
        """
        v3.0: 能量夺取逻辑 (v10.0 自然化版本)
        
        改动:
        - 巢穴范围内免疫抢劫
        - 攻击者需要付出更高代价
        """
        my_force = np.sum(np.abs(actuator_outputs))
        collision_dist = 5.0  # 碰撞距离
        
        # v10.0: 检查自己是否在巢穴安全区
        in_my_nest = False
        if self.nest_enabled:
            dist_to_nest = self._toroidal_distance(
                agent.x, agent.y, self.nest_position[0], self.nest_position[1]
            )
            in_my_nest = dist_to_nest < self.nest_radius

        for other in self.agents:
            if other is agent or not other.is_alive:
                continue
            
            # v10.0: 检查对方是否在巢穴安全区
            in_other_nest = False
            if self.nest_enabled:
                dist_to_nest = self._toroidal_distance(
                    other.x, other.y, self.nest_position[0], self.nest_position[1]
                )
                in_other_nest = dist_to_nest < self.nest_radius
            
            # 巢穴安全区: 双方都免疫抢劫
            if in_my_nest or in_other_nest:
                continue

            dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)

            if dist < collision_dist:
                other_force = 0.0  # 简化: 假设其他 Agent 推力为0

                # v10.0: 暴力代价 - 抢劫成本提高
                # 如果自己的推力更大，夺取能量
                if my_force > other_force and other.internal_energy > 10:
                    steal_amount = other.internal_energy * 0.20  # 夺取20%
                    agent.internal_energy = min(
                        agent.max_energy,
                        agent.internal_energy + steal_amount
                    )
                    other.internal_energy -= steal_amount
                    
                    # v10.0: 攻击者也要付出代价 (反作用力)
                    recoil_loss = steal_amount * 0.10  # 10%反作用力
                    agent.internal_energy -= recoil_loss
                    
                    # v10.0: 攻击增加疲劳度
                    if hasattr(agent, 'fatigue'):
                        agent.fatigue = min(agent.max_fatigue, agent.fatigue + 0.2)
                    
                    # v5.7 追踪战斗能量转移
                    agent.energy_gained += steal_amount * agent.eta_meat
                    other.energy_wasted += steal_amount * (1 - agent.eta_meat)

    def _compute_sensor_values(self, agent: Agent) -> np.ndarray:
        """
        计算智能体的传感器值 (v7.2 优化版, v0.81 可学习感知单元)
        """
        # v0.76: 红皇后buff - 敌对Agent获得传感器增强
        sensor_range = self.sensor_range
        if getattr(agent, 'is_rival', False):
            buff = getattr(agent, 'rival_buff', 1.0)
            sensor_range *= buff  # 增强感知范围

        # v4.1: 幽灵猎物 - 信号消失时返回0
        if self.blink_enabled and not self.blink_signal_on:
            return np.array([0.0, 0.0])

        # ============================================================
        # v0.81: 使用可学习感知单元
        # ============================================================
        from .node import NodeType

        # 检查Agent是否有可学习传感器的基因组
        has_learnable = False
        if hasattr(agent, 'genome') and agent.genome:
            sensor_nodes = [n for n in agent.genome.nodes.values()
                           if n.node_type == NodeType.SENSOR]
            if sensor_nodes and sensor_nodes[0].angle_weights is not None:
                has_learnable = True

        if has_learnable:
            # 收集所有目标信息
            targets = []

            # 1. 食物 (type=0) - 使用环形世界距离
            for fx, fy in self.food_positions[:5]:  # 最多5个
                dx, dy = self._toroidal_distance_components(agent.x, agent.y, fx, fy)
                dist = np.sqrt(dx**2 + dy**2)
                if dist < sensor_range:
                    angle = np.degrees(np.arctan2(dy, dx) - agent.theta) % 360
                    targets.append({
                        'angle': angle,
                        'distance': dist,
                        'type': 0  # food
                    })

            # 2. 巢穴 (type=1) - 使用环形世界距离
            if getattr(self, 'nest_enabled', False):
                nx, ny = self.nest_position
                dx, dy = self._toroidal_distance_components(agent.x, agent.y, nx, ny)
                dist = np.sqrt(dx**2 + dy**2)
                if dist < sensor_range * 1.5:
                    angle = np.degrees(np.arctan2(dy, dx) - agent.theta) % 360
                    targets.append({
                        'angle': angle,
                        'distance': dist,
                        'type': 1  # nest
                    })

            # 3. 敌对Agent (type=2)
            if hasattr(self, 'rivals'):
                for rival in getattr(self, 'rivals', []):
                    if not rival.is_alive:
                        continue
                    dx, dy = rival.x - agent.x, rival.y - agent.y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < sensor_range:
                        angle = np.degrees(np.arctan2(dy, dx) - agent.theta) % 360
                        targets.append({
                            'angle': angle,
                            'distance': dist,
                            'type': 2  # rival
                        })

            # 4. 墙壁 (type=3)
            for wx1, wy1, wx2, wy2 in self.walls:
                cx, cy = (wx1 + wx2) / 2, (wy1 + wy2) / 2
                dx, dy = cx - agent.x, cy - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist < sensor_range:
                    angle = np.degrees(np.arctan2(dy, dx) - agent.theta) % 360
                    targets.append({
                        'angle': angle,
                        'distance': dist,
                        'type': 3  # wall
                    })

            # 为每个传感器节点计算感知
            sensor_values = []
            for sensor_node in sensor_nodes:
                # 使用学习的注意力权重
                activation = sensor_node.get_sensor_attention(targets)
                sensor_values.append(activation)

            # 标准化到 [0, 1]
            if sensor_values:
                max_val = max(abs(v) for v in sensor_values) + 1e-8
                sensor_values = [v / max_val for v in sensor_values]

            # 基础传感器值 (2维)
            base_sensors = np.array(sensor_values[:2] if len(sensor_values) >= 2 else sensor_values + [0.0])
            
            # ============================================================
            # v13.0: 统一场物理传感器 (追加到基础传感器)
            # 格式: [EPF×3, KIF×3, ISF×3, ESF×1, ENERGY×1]
            # ============================================================
            gx = int(agent.x / self.energy_field.resolution) % self.energy_field.grid_width if self.energy_field else 0
            gy = int(agent.y / self.energy_field.resolution) % self.energy_field.grid_height if self.energy_field else 0
            
            # EPF: 能量场感知 (3维)
            if self.energy_field_enabled and self.energy_field:
                epf_center = self.energy_field.sample(agent.x, agent.y)
                epf_grad_x = self.epf_grad_x[gx, gy] if self.epf_grad_x is not None else 0.0
                epf_grad_y = self.epf_grad_y[gx, gy] if self.epf_grad_y is not None else 0.0
                epf_c = max(0.0, min(1.0, epf_center / 100.0))
                epf_gx = max(-1.0, min(1.0, epf_grad_x / 10.0))
                epf_gy = max(-1.0, min(1.0, epf_grad_y / 10.0))
                v13_sensors = np.array([epf_c, epf_gx, epf_gy])
            else:
                v13_sensors = np.zeros(3)
            
            # KIF: 阻抗场感知 (3维)
            if self.impedance_field_enabled and self.impedance_field:
                kif_center = self.impedance_field.sample(agent.x, agent.y)
                kif_grad_x = self.kif_grad_x[gx, gy] if self.kif_grad_x is not None else 0.0
                kif_grad_y = self.kif_grad_y[gx, gy] if self.kif_grad_y is not None else 0.0
                kif_c = max(0.0, min(1.0, kif_center / 100.0))
                kif_gx = max(-1.0, min(1.0, kif_grad_x / 10.0))
                kif_gy = max(-1.0, min(1.0, kif_grad_y / 10.0))
                v13_sensors = np.append(v13_sensors, [kif_c, kif_gx, kif_gy])
            else:
                v13_sensors = np.append(v13_sensors, [0.0, 0.0, 0.0])
            
            # ISF: 压痕场感知 (3维)
            if self.stigmergy_field_enabled and self.stigmergy_field:
                isf_center = self.stigmergy_field.sample(agent.x, agent.y)
                isf_grad_x = self.isf_grad_x[gx, gy] if self.isf_grad_x is not None else 0.0
                isf_grad_y = self.isf_grad_y[gx, gy] if self.isf_grad_y is not None else 0.0
                isf_c = max(0.0, min(1.0, isf_center / 10.0))
                isf_gx = max(-1.0, min(1.0, isf_grad_x))
                isf_gy = max(-1.0, min(1.0, isf_grad_y))
                v13_sensors = np.append(v13_sensors, [isf_c, isf_gx, isf_gy])
            else:
                v13_sensors = np.append(v13_sensors, [0.0, 0.0, 0.0])
            
            # ESF: 应力场感知 (1维)
            if self.stress_field_enabled and self.stress_field:
                stress = getattr(agent, 'current_stress', 0.0)
                stress_norm = max(-1.0, min(1.0, stress))
                v13_sensors = np.append(v13_sensors, stress_norm)
            else:
                v13_sensors = np.append(v13_sensors, 0.0)
            
            # INTERNAL_ENERGY: 体内能量感知 (1维)
            agent_energy = getattr(agent, 'internal_energy', 150.0)
            energy_norm = max(0.0, min(1.0, agent_energy / 200.0))
            v13_sensors = np.append(v13_sensors, energy_norm)
            
            return np.concatenate([base_sensors, v13_sensors])

        # ============================================================
        # 原有逻辑 (非可学习传感器)
        # ============================================================

        # v8.0: 无尽边疆模式 - 从区块获取食物
        if self.infinite_mode and self.chunk_manager:
            foods = self.chunk_manager.get_foods_in_range(
                agent.x, agent.y, self.sensor_range
            )

            if foods:
                # 找到最近的食物
                nearest = min(foods, key=lambda f: f[2])
                target_x, target_y, dist = nearest

                # 计算传感器值 (传递环形世界尺寸)
                sensor_values = compute_sensor_vectorized(
                    agent.x, agent.y, agent.theta,
                    target_x, target_y,
                    sensor_range, self.width, self.height
                )
            else:
                # 无食物时返回低信号
                sensor_values = np.array([0.1, 0.1])
        else:
            # 标准模式 (传递环形世界尺寸)
            sensor_values = compute_sensor_vectorized(
                agent.x, agent.y, agent.theta,
                self.target_pos[0], self.target_pos[1],
                sensor_range, self.width, self.height
            )

        # v7.0: 不透明障碍物 - 射线检测
        if self.opaque_walls:
            left_blocked = self._raycast_to_target(
                (agent.x, agent.y), self.target_pos
            )
            if left_blocked:
                sensor_values *= 0.1

        # v0.97: 添加疲劳传感器 (内部状态感知)
        # 格式扩展: [左食物, 右食物, 疲劳度]
        if self.fatigue_system_enabled:
            fatigue_sensor = getattr(agent, 'fatigue', 0.0) / getattr(agent, 'max_fatigue', 50.0)
            # 优化: 用原生Python代替np.clip (标量操作)
            fatigue_sensor = max(0.0, min(1.0, fatigue_sensor))
            sensor_values = np.append(sensor_values, fatigue_sensor)

        # ============================================================
        # v13.0: 统一场物理传感器 (11维)
        # 使用预计算的梯度矩阵 (O(1) 索引) 实现性能优化
        # 格式: [EPF×3, KIF×3, ISF×3, ESF×1, ENERGY×1]
        # ============================================================
        
        # 计算网格坐标
        gx = int(agent.x / self.energy_field.resolution) % self.energy_field.grid_width if self.energy_field else 0
        gy = int(agent.y / self.energy_field.resolution) % self.energy_field.grid_height if self.energy_field else 0
        
        # EPF: 能量场感知 (3维: CENTER, GRAD_X, GRAD_Y)
        if self.energy_field_enabled and self.energy_field:
            epf_center = self.energy_field.sample(agent.x, agent.y)
            # 使用预计算的梯度矩阵
            epf_grad_x = self.epf_grad_x[gx, gy] if self.epf_grad_x is not None else 0.0
            epf_grad_y = self.epf_grad_y[gx, gy] if self.epf_grad_y is not None else 0.0
            # 归一化
            epf_c = max(0.0, min(1.0, epf_center / 100.0))
            epf_gx = max(-1.0, min(1.0, epf_grad_x / 10.0))
            epf_gy = max(-1.0, min(1.0, epf_grad_y / 10.0))
            sensor_values = np.append(sensor_values, [epf_c, epf_gx, epf_gy])
        else:
            sensor_values = np.append(sensor_values, [0.0, 0.0, 0.0])
        
        # KIF: 阻抗场感知 (3维: CENTER, GRAD_X, GRAD_Y)
        if self.impedance_field_enabled and self.impedance_field:
            kif_center = self.impedance_field.sample(agent.x, agent.y)
            kif_grad_x = self.kif_grad_x[gx, gy] if self.kif_grad_x is not None else 0.0
            kif_grad_y = self.kif_grad_y[gx, gy] if self.kif_grad_y is not None else 0.0
            # 归一化
            kif_c = max(0.0, min(1.0, kif_center / 100.0))
            kif_gx = max(-1.0, min(1.0, kif_grad_x / 10.0))
            kif_gy = max(-1.0, min(1.0, kif_grad_y / 10.0))
            sensor_values = np.append(sensor_values, [kif_c, kif_gx, kif_gy])
        else:
            sensor_values = np.append(sensor_values, [0.0, 0.0, 0.0])
        
        # ISF: 压痕场感知 (3维: CENTER, GRAD_X, GRAD_Y)
        if self.stigmergy_field_enabled and self.stigmergy_field:
            isf_center = self.stigmergy_field.sample(agent.x, agent.y)
            isf_grad_x = self.isf_grad_x[gx, gy] if self.isf_grad_x is not None else 0.0
            isf_grad_y = self.isf_grad_y[gx, gy] if self.isf_grad_y is not None else 0.0
            # 归一化
            isf_c = max(0.0, min(1.0, isf_center / 10.0))
            isf_gx = max(-1.0, min(1.0, isf_grad_x))
            isf_gy = max(-1.0, min(1.0, isf_grad_y))
            sensor_values = np.append(sensor_values, [isf_c, isf_gx, isf_gy])
        else:
            sensor_values = np.append(sensor_values, [0.0, 0.0, 0.0])
        
        # ESF: 应力场感知 (1维: VAL)
        if self.stress_field_enabled and self.stress_field:
            stress = getattr(agent, 'current_stress', 0.0)
            stress_norm = max(-1.0, min(1.0, stress))
            sensor_values = np.append(sensor_values, stress_norm)
        else:
            sensor_values = np.append(sensor_values, 0.0)
        
        # INTERNAL_ENERGY: 体内能量感知 (1维: 饱腹感)
        agent_energy = getattr(agent, 'internal_energy', 150.0)
        energy_norm = max(0.0, min(1.0, agent_energy / 200.0))  # 归一化到0-200范围
        sensor_values = np.append(sensor_values, energy_norm)

        # ============================================================
        # v13.0: 阻抗场传感器 (Impedance Field)
        # 采样当前阻抗 + 梯度方向
        # 格式扩展: [当前阻抗, 梯度dx, 梯度dy]
        # ============================================================
        if self.impedance_field_enabled and self.impedance_field:
            Z = self.impedance_field.sample(agent.x, agent.y)
            grad_x, grad_y = self.impedance_field.sample_gradient(agent.x, agent.y)
            # 归一化: 假设最大阻抗 100
            impedance_sensor = max(0.0, min(1.0, Z / 100.0))
            grad_x_norm = max(-1.0, min(1.0, grad_x / 10.0))
            grad_y_norm = max(-1.0, min(1.0, grad_y / 10.0))
            sensor_values = np.append(sensor_values, [impedance_sensor, grad_x_norm, grad_y_norm])

        # ============================================================
        # v13.0: 压痕场传感器 (Stigmergy Field)
        # 采样信号强度 + 梯度方向
        # 格式扩展: [信号强度, 梯度dx, 梯度dy, 梯度幅值]
        # ============================================================
        if self.stigmergy_field_enabled and self.stigmergy_field:
            S = self.stigmergy_field.sample(agent.x, agent.y)
            grad_x, grad_y, grad_mag = self.stigmergy_field.sample_gradient(agent.x, agent.y)
            # 归一化: 假设最大信号 10
            signal_norm = max(0.0, min(1.0, S / 10.0))
            grad_x_norm = max(-1.0, min(1.0, grad_x))
            grad_y_norm = max(-1.0, min(1.0, grad_y))
            grad_mag_norm = max(0.0, min(1.0, grad_mag))
            sensor_values = np.append(sensor_values, [signal_norm, grad_x_norm, grad_y_norm, grad_mag_norm])

        # ============================================================
        return sensor_values

    def _compute_light_sensor(self, agent: Agent) -> np.ndarray:
        """
        v7.0: 移动光源传感器

        返回 [左光源传感器, 右光源传感器]

        光源位置动态变化，迫使Agent:
        - 记住光源的历史位置
        - 使用DELAY回路来预测光源轨迹
        """
        # 目标向量 (光源) - 使用环形世界距离
        dx, dy = self._toroidal_distance_components(agent.x, agent.y, 
                                                     self.light_pos[0], self.light_pos[1])
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.1:
            return np.array([1.0, 1.0])

        # 目标方向
        target_angle = np.arctan2(dy, dx)

        # 相对角度
        relative_angle = target_angle - agent.theta
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))

        # 传感器偏移
        sensor_offset = np.radians(30)
        left_angle = relative_angle + sensor_offset
        right_angle = relative_angle - sensor_offset

        def gaussian_attention(angle, sigma=np.radians(45)):
            return np.exp(-0.5 * (angle / sigma) ** 2)

        # 距离衰减 - 优化: 用原生Python代替np.clip
        distance_decay = self.sensor_range / (distance + 1.0)
        distance_decay = max(0.0, min(1.0, distance_decay))

        left_sensor = distance_decay * gaussian_attention(left_angle)
        right_sensor = distance_decay * gaussian_attention(right_angle)

        # v7.0: 不透明障碍物也阻挡光源
        if self.opaque_walls:
            left_blocked = self._raycast_to_target(
                (agent.x, agent.y), self.light_pos
            )
            right_blocked = self._raycast_to_target(
                (agent.x, agent.y), self.light_pos
            )
            if left_blocked:
                left_sensor *= 0.1
            if right_blocked:
                right_sensor *= 0.1

        return np.array([left_sensor, right_sensor])

    def _compute_agent_radar_sensor(self, agent: Agent) -> np.ndarray:
        """
        v7.0: 社会信号传感器

        检测附近Agent的PORT_SIGNAL值

        返回 [最近Agent信号, 第二近, 第三近]

        预期涌现行为:
        - 捕食者模仿食物信号 (低信号伪装)
        - 亲族识别 (特定信号模式)
        - 群体协调 (同步信号)
        """
        return self._detect_nearby_signals(agent)

    def _compute_gps_sensor(self, agent: Agent) -> np.ndarray:
        """
        v8.0: GPS坐标传感器

        返回 [相对X位移, 相对Y位移, 离原点距离, 朝向原点的角度]

        这些信号让Agent能够:
        - 记住自己走了多远
        - 演化出"回巢"本能
        - 计算探索vs返回的权衡
        """
        # 相对于原点的位移 - 使用环形世界距离
        rel_x, rel_y = self._toroidal_distance_components(
            agent.x, agent.y, self.origin_x, self.origin_y
        )

        # 距离
        distance = np.sqrt(rel_x**2 + rel_y**2)

        # 朝向原点的角度
        angle_to_origin = np.arctan2(-rel_y, -rel_x)

        # 当前朝向与原点方向的相对角度
        relative_bearing = angle_to_origin - agent.theta
        relative_bearing = np.arctan2(np.sin(relative_bearing), np.cos(relative_bearing))

        # 归一化到 [0, 1]
        normalized_dist = min(distance / 200.0, 1.0)  # 200单位为最大感知距离
        normalized_bearing = (relative_bearing + np.pi) / (2 * np.pi)  # [0, 1]

        return np.array([
            (rel_x + 200) / 400,  # 归一化X [-200, 200] -> [0, 1]
            (rel_y + 200) / 400,  # 归一化Y [-200, 200] -> [0, 1]
            normalized_dist,       # 归一化距离
            normalized_bearing     # 归一化朝向
        ])

    # ============================================================
    # v1.3: 环形世界距离计算 (修复边界Bug)
    # ============================================================
    
    def _toroidal_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """计算环形世界中的最短距离"""
        dx = x2 - x1
        dy = y2 - y1
        # 取最短路径
        dx = dx - self.width * np.floor(dx / self.width + 0.5)
        dy = dy - self.height * np.floor(dy / self.height + 0.5)
        return np.sqrt(dx**2 + dy**2)
    
    def _toroidal_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """计算环形世界中的最短距离 (v10.0 Numba优化)"""
        if NUMBA_AVAILABLE:
            return _fast_toroidal_distance_scalar(x1, y1, x2, y2, self.width, self.height)
        else:
            # NumPy回退
            dx = x2 - x1
            dy = y2 - y1
            dx = dx - self.width * np.floor(dx / self.width + 0.5)
            dy = dy - self.height * np.floor(dy / self.height + 0.5)
            return np.sqrt(dx**2 + dy**2)
    
    def _toroidal_distance_components(self, x1: float, y1: float, x2: float, y2: float):
        """计算环形世界中的距离分量(返回最短路径的dx, dy)"""
        dx = x2 - x1
        dy = y2 - y1
        dx = dx - self.width * np.floor(dx / self.width + 0.5)
        dy = dy - self.height * np.floor(dy / self.height + 0.5)
        return dx, dy

    def _compute_compass_sensor(self, agent: Agent) -> np.ndarray:
        """
        v8.0: 指南针传感器

        返回 [北向偏差, 东向偏差]

        让Agent知道自己的绝对朝向
        """
        # 绝对方向 (假设北为-Y方向)
        north_angle = -np.pi / 2

        # 相对角度
        relative = agent.theta - north_angle
        relative = np.arctan2(np.sin(relative), np.cos(relative))

        return np.array([
            (relative + np.pi) / (2 * np.pi),  # 归一化
            (np.sin(agent.theta) + 1) / 2,     # sin分量
            (np.cos(agent.theta) + 1) / 2      # cos分量
        ])

    def _update_agent_physics(
        self,
        agent: Agent,
        actuator_outputs: np.ndarray
    ) -> None:
        """
        根据执行器输出更新智能体物理状态 (v7.2 优化版)

        差速驱动逻辑:
            - 左推进力 > 右推进力 → 右转
            - 右推进力 > 左推进力 → 左转
            - 相等 → 直线前进
        """
        left_force = actuator_outputs[0]
        right_force = actuator_outputs[1]

        # 记录旧位置 (用于压痕更新)
        old_x, old_y = agent.x, agent.y

        # v0.76: 红皇后buff - 敌对Agent获得速度增强
        max_speed = self.max_speed
        if getattr(agent, 'is_rival', False):
            buff = getattr(agent, 'rival_buff', 1.0)
            max_speed *= buff  # 增强速度

        # v0.97: 渐进式虚弱 - 疲劳度越高，速度越慢
        if self.fatigue_system_enabled and hasattr(agent, 'fatigue'):
            fatigue_ratio = agent.fatigue / agent.max_fatigue if agent.max_fatigue > 0 else 0
            # 速度从100%衰减到20%
            max_speed *= (1.0 - fatigue_ratio * 0.8)

        # v0.98: 形态计算 - 携带食物时的物理阻力
        if self.morphological_computation_enabled and agent.food_carried > 0:
            # 携带食物时移动速度降低 (物理负载)
            load_penalty = 1.0 - (agent.food_carried * (1.0 - self.carry_speed_penalty))
            max_speed *= max(0.3, load_penalty)  # 最低30%速度

        # v0.98: 压痕系统 - 低阻力高速公路加成
        if self.stigmergic_friction_enabled:
            friction_bonus = self.get_friction_bonus(agent.x, agent.y)
            max_speed *= friction_bonus

        # v7.2: 使用向量化函数
        new_x, new_y, new_theta = update_positions_vectorized(
            agent.x, agent.y, agent.theta,
            left_force, right_force,
            max_speed, self.turn_rate,
            self.width, self.height
        )

        # v3.0: 墙壁碰撞检测
        if self.walls:
            new_x, new_y = self._check_wall_collision(agent.x, agent.y, new_x, new_y)

        # 计算移动距离作为速度 (在更新位置之前)
        dx = new_x - agent.x
        dy = new_y - agent.y
        agent.speed = np.sqrt(dx**2 + dy**2)

        # 更新状态
        agent.x = new_x
        agent.y = new_y
        agent.theta = new_theta

        # 边界处理 (环绕)
        agent.x = agent.x % self.width
        agent.y = agent.y % self.height

        # v0.98: 压痕系统 - 更新路径压痕
        if self.stigmergic_friction_enabled:
            self._update_friction_grid(agent, old_x, old_y)

    def _compute_battle_signals(self, agent: Agent) -> None:
        """
        v6.0 GAIA: 计算Agent的物理端口信号

        5个物理端口:
        - PORT_MOTION: [速度, 转向] 基础移动
        - PORT_OFFENSE: 攻击强度 (捕食能力)
        - PORT_DEFENSE: 防御硬度 (减少被掠夺)
        - PORT_REPAIR: 修复 (能量→寿命)
        - PORT_SIGNAL: 信号 (诱导/伪装)

        v6.0 GAIA 代价公式: E_cost = Σ(Port_Output² × Weight)
        """
        genome = agent.genome

        # 初始化各端口
        motion = 0.0
        offense = 0.0
        defense = 0.0
        repair = 0.0
        signal = 0.0

        # 计算物理输出端口激活
        for node_id, node in genome.nodes.items():
            if node.node_type == NodeType.PORT_MOTION:
                motion = max(motion, abs(node.activation))
            elif node.node_type == NodeType.PORT_OFFENSE:
                offense = max(offense, node.activation)
            elif node.node_type == NodeType.PORT_DEFENSE:
                defense = max(defense, node.activation)
            elif node.node_type == NodeType.PORT_REPAIR:
                repair = max(repair, node.activation)
            elif node.node_type == NodeType.PORT_SIGNAL:
                signal = max(signal, node.activation)

        # v6.0 GAIA: 标准化到 [0, 1] - 优化: 用原生Python
        agent.port_motion = max(0.0, min(1.0, motion))
        agent.port_offense = max(0.0, min(1.0, offense))
        agent.port_defense = max(0.0, min(1.0, defense))
        agent.port_repair = max(0.0, min(1.0, repair))
        agent.port_signal = max(0.0, min(1.0, signal))

        # 兼容旧接口
        agent.attack_power = agent.port_offense
        agent.defense_power = agent.port_defense

        # v6.0 GAIA: 端口能耗 (二次方代价)
        # E_cost = Σ(Port_Output² × Weight)
        port_weights = {
            'motion': 0.3,
            'offense': 0.5,
            'defense': 0.4,
            'repair': 0.6,
            'signal': 0.2
        }

        port_cost = (
            (agent.port_motion ** 2) * port_weights['motion'] +
            (agent.port_offense ** 2) * port_weights['offense'] +
            (agent.port_defense ** 2) * port_weights['defense'] +
            (agent.port_repair ** 2) * port_weights['repair'] +
            (agent.port_signal ** 2) * port_weights['signal']
        )

        # 扣除端口能耗
        if agent.internal_energy > port_cost:
            agent.internal_energy -= port_cost
            agent.energy_spent += port_cost

        # v6.0 GAIA REPAIR: 能量→寿命转换 (消耗能量减少年龄)
        if agent.port_repair > 0.1 and agent.internal_energy > 10:
            repair_cost = agent.port_repair * 2.0  # 每单位修复消耗2能量
            if agent.internal_energy > repair_cost:
                agent.internal_energy -= repair_cost
                agent.energy_spent += repair_cost  # v5.7 追踪
                agent.age = max(0, agent.age - agent.port_repair * 0.5)  # 减少年龄

        # ============================================================
        # v13.0 神经拓扑对接 - 更新物理状态
        # 将大脑输出映射到 κ, F, λ, S 四个物理参数
        # ============================================================
        # 构建 5 维大脑输出向量
        brain_output = np.array([
            agent.port_signal,          # κ (复用 port_signal 作为渗透率)
            agent.port_motion,          # Fx (复用 port_motion)
            agent.port_motion * 0.5,    # Fy (简化为单向推力)
            agent.port_signal * 0.8,    # λ (独立信号，略微降低)
            agent.port_defense          # S (防御刚性)
        ])
        
        # 更新物理状态
        agent.update_physics_states(brain_output)

        # v6.0 GAIA: 计算生态位偏好
        self._compute_niche_preference(agent)

    def _handle_agent_collisions(self) -> None:
        """
        v5.3 Battle Royale: Agent间碰撞与能量掠夺

        胜负判定: ATTACK vs DEFENSE
        能量掠夺: 胜者吸收败者70%能量
        淘汰: 能量归零的Agent死亡
        """
        collision_dist = 8.0  # 碰撞距离阈值 (增大以促进战斗)

        alive_agents = [a for a in self.agents if a.is_alive]

        for i, agent_a in enumerate(alive_agents):
            for agent_b in alive_agents[i+1:]:
                # 计算距离
                dist = np.sqrt((agent_a.x - agent_b.x)**2 + (agent_a.y - agent_b.y)**2)

                if dist < collision_dist:
                    # 战斗判定
                    self._resolve_battle(agent_a, agent_b)

        # 清理死亡Agent
        self.agents = [a for a in self.agents if a.is_alive]

    def _resolve_battle(self, agent_a: Agent, agent_b: Agent) -> None:
        """
        v6.0 GAIA: 解决单次战斗 (v10.1 生态平衡版)

        - 攻击方: PORT_OFFENSE 高频时触发
        - 防御方: PORT_DEFENSE 激活时减伤80%
        - 胜者判定: 攻击方攻击 > 防御方防御 → 攻击方胜
        - 防御成功: 攻击者受到双倍疲劳惩罚
        """
        # 基础攻击/防御值 (来自端口)
        attack_a = agent_a.port_offense
        attack_b = agent_b.port_offense
        defense_a = agent_a.port_defense
        defense_b = agent_b.port_defense

        # 攻击修正: PORT_OFFENSE 高频激活时攻击增强
        if attack_a > 0.7:
            attack_a *= 1.5
        if attack_b > 0.7:
            attack_b *= 1.5

        # v10.1: 防御修正 - 80%减伤 (更硬的外壳)
        # v10.2: 携带buff - 携带食物时防御+30%
        if agent_a.food_carried > 0:
            defense_a *= 1.3  # 携带时防御增强30%
        if agent_b.food_carried > 0:
            defense_b *= 1.3
            
        effective_defense_a = min(defense_a * 0.8, 0.95)  # 最多减免95%
        effective_defense_b = min(defense_b * 0.8, 0.95)

        # 胜者判定: 攻击方突破防御 = 攻击 > (1 - 防御)
        a_attacks_b = attack_a > (1.0 - effective_defense_b)
        b_attacks_a = attack_b > (1.0 - effective_defense_a)

        if a_attacks_b and not b_attacks_a:
            winner, loser = agent_a, agent_b
        elif b_attacks_a and not a_attacks_b:
            winner, loser = agent_b, agent_a
        elif a_attacks_b and b_attacks_a:
            # 双方都突破: 能量交换
            exchange = min(3.0, min(agent_a.internal_energy, agent_b.internal_energy) / 2)
            agent_a.internal_energy -= exchange
            agent_b.internal_energy -= exchange
            # v5.7 追踪能量交换消耗
            agent_a.energy_spent += exchange
            agent_b.energy_spent += exchange
            return
        else:
            # 无人突破: 平局
            return

        # v5.4/v5.7 能量守恒: 应用肉食吸收率 η_meat (70%)
        steal_base = loser.internal_energy * 0.7
        absorbed = steal_base * winner.eta_meat
        wasted = steal_base * (1 - winner.eta_meat)

        winner.internal_energy += absorbed
        loser.internal_energy -= steal_base
        # v5.7 追踪能量流动
        winner.energy_gained += absorbed  # 获胜者获得能量
        winner.energy_wasted += wasted   # 30%散失到环境
        loser.energy_wasted += (steal_base - absorbed)  # 失败者损失

        # 记录统计
        winner.energy_stolen += absorbed
        winner.battle_wins += 1
        loser.battle_losses += 1
        
        # v10.1: 防御成功惩罚 - 攻击者被防御挡住时额外疲劳
        # 检查A攻击B是否被防御挡住
        defender = agent_b if a_attacks_b else agent_a
        defender_defense = defender.port_defense
        if defender_defense > 0.3:  # 如果防御方有防御
            attacker = agent_b if a_attacks_b else agent_a
            if hasattr(attacker, 'fatigue'):
                attacker.fatigue = min(attacker.max_fatigue, attacker.fatigue + 0.4)  # 双倍疲劳惩罚

        # 检查Epic Motif: 3连胜
        if winner.battle_wins >= 3:
            winner.is_epic_motif = True

        # 淘汰: 能量归零
        if loser.internal_energy <= 0:
            loser.is_alive = False
            loser.internal_energy = 0
            
            # v10.1: 死亡掉落高能尸体 (5倍普通食物能量)
            corpse_energy = self.food_energy * 5.0  # 尸体能量密度 = 5倍食物
            self.food_positions.append((loser.x, loser.y))
            self.food_velocities.append((0.0, 0.0))
            self.food_freshness.append(2.0)  # 尸体更新鲜，保质期更长

    def step(self) -> None:
        """
        环境单步更新 (EOE v4.0)

        流程:
            1. 昼夜循环更新
            2. 诱饵移动更新 (v4.0)
            3. 跳过死亡 Agent
            4. 计算传感器值 (多食物 + ENTITY_RADAR + BAIT)
            5. 计算 Surprise
            6. 前向传播
            7. 检查是否吃到食物
            8. 检查是否吃到诱饵 (v4.0 - 负奖励)
            9. 能量夺取 (社会交互)
            10. 能量消耗 (内稳态 + 昼夜税率)
            11. 赫布学习更新
            12. 更新物理状态 (含墙壁碰撞)
            13. 更新适应度
        """
        # v5.2: 步数计数
        self.step_count += 1

        # ============================================================
        # v13.0: 能量场更新
        # ============================================================
        if self.energy_field_enabled and self.energy_field:
            self.energy_field.step()

        # ============================================================
        # v13.0: 压痕场更新 (扩散 + 衰减)
        # ============================================================
        if self.stigmergy_field_enabled and self.stigmergy_field:
            self.stigmergy_field.step()

        # ============================================================
        # v13.0: 应力场 - 场间耦合 (调制物理常数)
        # ============================================================
        if self.stress_field_enabled and self.stress_field:
            self.stress_field.apply_coupling(self, self.step_count)

        # ============================================================
        # v13.0: 预计算梯度矩阵 (性能优化)
        # 在环境层级计算一次梯度，Agent只做O(1)索引
        # ============================================================
        if self.energy_field_enabled and self.energy_field:
            self.epf_grad_y, self.epf_grad_x = np.gradient(self.energy_field.field)
        
        if self.impedance_field_enabled and self.impedance_field:
            self.kif_grad_y, self.kif_grad_x = np.gradient(self.impedance_field.field)
        
        if self.stigmergy_field_enabled and self.stigmergy_field:
            self.isf_grad_y, self.isf_grad_x = np.gradient(self.stigmergy_field.field)

        # ============================================================
        # v8.0: 无尽边疆 - 区块加载
        # ============================================================
        if self.infinite_mode and self.chunk_manager:
            # 加载Agent感知范围内的区块
            self.chunk_manager.ensure_chunks_loaded(
                self.agents, self.sensor_range, self.step_count
            )

            # 记录每个Agent的访问
            for agent in self.agents:
                if hasattr(agent, 'is_alive') and agent.is_alive:
                    self.chunk_manager.record_access(agent.x, agent.y, self.step_count)

            # 定期清理不活跃区块
            if self.step_count % 500 == 0:
                cleaned = self.chunk_manager.cleanup_inactive_chunks(self.step_count)
                if cleaned > 0:
                    print(f"  [Infinite] Cleaned {cleaned} inactive chunks")

        # ============================================================
        # v3.0: 昼夜循环
        # ============================================================
        self.current_time += 1
        if self.day_night_cycle and self.current_time % self.day_length == 0:
            self.is_day = not self.is_day

        # ============================================================
        # v0.78: 季节循环 (夏天→冬天→夏天...)
        # ============================================================
        if self.seasonal_cycle:
            self.season_frame += 1
            if self.season_frame >= self.season_length:
                self.season_frame = 0
                # 切换季节
                if self.current_season == "summer":
                    self.current_season = "winter"
                    # 冬天开始时，Agent可以用存储的食物补充能量
                    if self.nest_enabled:
                        for agent in self.agents:
                            if agent.food_stored > 0:
                                bonus = agent.food_stored * self.food_energy * 0.5
                                agent.internal_energy = min(
                                    agent.max_energy,
                                    agent.internal_energy + bonus
                                )
                else:
                    self.current_season = "summer"

                    # 春天: 重置食物(如果需要)
                    if self.respawn_food and len(self.food_positions) < self.n_food:
                        self._init_food()

        # ============================================================
        # v4.1: 幽灵猎物 (Blinking Prey)
        # ============================================================
        self._update_blink()

        # 代谢税率: 黑夜更高 + 冬天更高
        metabolic_multiplier = 0.6 if self.is_day else 1.5
        # v0.78: 冬天代谢加成
        if self.seasonal_cycle and self.current_season == "winter":
            metabolic_multiplier *= self.winter_metabolic_multiplier
        # v13.0: ESF 应力场调制
        if self.stress_field_enabled:
            metabolic_multiplier *= self.stress_metabolic_multiplier

        # ============================================================
        # v4.0: 更新诱饵位置
        # ============================================================
        if self.n_bait > 0:
            self._update_bait()

        # ============================================================
        # v0.98: 热力学庇护所 - 温度场更新
        # ============================================================
        self._update_thermal_system()

        # ============================================================
        # v0.98: 压痕系统 - 全局衰减 (每步衰减)
        # ============================================================
        if self.stigmergic_friction_enabled and self.friction_grid is not None:
            self.friction_grid *= self.trail_decay

        # 收集本步所有节点的激活值 (用于赫布学习)
        all_node_activations = {}

        # ============================================================
        # v3.0: 更新 ENTITY_RADAR 节点数据
        # ============================================================
        radar_range = 30.0  # 雷达感知范围

        for agent in self.agents:
            if not agent.is_alive:
                continue

            # 为每个 ENTITY_RADAR 节点设置雷达数据
            for node in agent.genome.nodes.values():
                if node.node_type == NodeType.ENTITY_RADAR:
                    # 寻找最近的3个竞争者
                    competitors = []
                    for other in self.agents:
                        if other is not agent and other.is_alive:
                            dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
                            if dist < radar_range:
                                competitors.append({
                                    'distance': dist,
                                    'energy': other.internal_energy,
                                    'x': other.x - agent.x,
                                    'y': other.y - agent.y
                                })

                    # 排序并取前3个
                    competitors.sort(key=lambda c: c['distance'])
                    competitors = competitors[:3]

                    # 编码: [最近距离/范围, 最近能量, 次近距离/范围, 次近能量, ...]
                    if competitors:
                        radar_data = []
                        for c in competitors:
                            radar_data.append(c['distance'] / radar_range)  # 归一化距离
                            radar_data.append(c['energy'] / 100.0)           # 归一化能量

                        # 不足6维则填充
                        while len(radar_data) < 6:
                            radar_data.append(0.0)

                        node.radar_data = np.mean(radar_data[:6])  # 简化为单一值
                    else:
                        node.radar_data = 0.0

        # 主循环
        for agent in self.agents:
            if not agent.is_alive:
                continue

            agent.steps_alive += 1

            # v7.0: 更新移动光源
            self._update_light_source()

            # v7.1: 随机环境事件
            self._trigger_chaos_event()
            self._update_chaos_effects()

            # ============================================================
            # v0.97: 三大突破机制
            # ============================================================

            # 方案1: 代谢疲劳 + 安全掩体
            self._update_fatigue_system(agent)

            # 方案2: 无聊信息素
            self._update_pheromone_system(agent)

            # 如果在睡眠，跳过后续传感器计算，但需要设置基本代谢
            if getattr(agent, 'is_sleeping', False):
                # 睡眠时代谢很低
                genome_info = agent.genome.get_info()
                n_nodes = genome_info['total_nodes']
                n_edges = genome_info['enabled_edges']
                metabolic_cost = n_nodes * self.metabolic_alpha * 0.1 + n_edges * self.metabolic_beta * 0.1
                agent.internal_energy -= metabolic_cost
                if agent.internal_energy <= 0:
                    agent.is_alive = False
                continue

            # 方案3: 夏日食物热力学 (全局更新)
            if self.food_thermodynamics_enabled:
                self._update_food_thermodynamics()

            # ============================================================
            # v0.98: 热力学庇护所 - 温度效果应用
            # ============================================================
            if self.thermal_sanctuary_enabled:
                self._apply_temperature_effects(agent)

            # ============================================================
            # v0.98: 形态计算 - 物理吸附检测
            # 在传感器计算之前检测吸附 (食物被移走)
            # ============================================================
            if self.morphological_computation_enabled:
                self._check_adhesion_collision(agent)
                # 检查放电/卸货信号
                self._check_discharge_and_store(agent, actuator_outputs if 'actuator_outputs' in dir() else np.array([0, 0]))

            # 1. 计算当前传感器输入 (多食物)
            current_sensor_values = self._compute_sensor_values(agent)

            # v9.0: 感官延迟墙 - 将当前传感器值加入历史
            agent.sensor_history.append(current_sensor_values.copy())

            # v9.0: 使用延迟后的传感器值 (30帧前的感知)
            if len(agent.sensor_history) >= agent.sensor_delay_frames:
                sensor_values = agent.sensor_history[0]  # 最旧的记录
            else:
                sensor_values = current_sensor_values  # 不足30帧时使用当前值

            # v7.1: 应用传感器噪声
            sensor_values = self._apply_sensor_noise(sensor_values)

            # v7.0: 计算光源传感器
            light_sensor_values = self._compute_light_sensor(agent)

            # v7.0: 计算社会信号传感器
            agent_radar_values = self._compute_agent_radar_sensor(agent)

            # 2. 计算 Surprise (预测误差)
            if agent.predicted_next_sensor is not None and agent.last_sensor_inputs is not None:
                surprise = np.mean((sensor_values - agent.predicted_next_sensor) ** 2)
                agent.prediction_error = surprise
                agent.surprise_accumulated += surprise
            else:
                agent.prediction_error = 0.0

            agent.last_sensor_inputs = sensor_values.copy()

            # v8.0: 计算GPS和指南针传感器
            gps_sensor_values = self._compute_gps_sensor(agent)  # [4]
            compass_sensor_values = self._compute_compass_sensor(agent)  # [3]

            # v0.98: 温度传感器
            if self.thermal_sanctuary_enabled:
                temp_sensor_values = np.array(agent.temperature_sensors)  # [左, 右, 舒适]
            else:
                temp_sensor_values = np.array([0.5, 0.5, 0.5])  # 默认值

            # 3. 前向传播 - 拼接所有传感器输入
            extended_inputs = np.concatenate([
                sensor_values,           # 食物传感器 [0,1]
                light_sensor_values,     # 光源传感器 [2,3]
                agent_radar_values,      # 社会传感器 [4,5,6]
                gps_sensor_values,       # GPS传感器 [7,8,9,10]
                compass_sensor_values,   # 指南针传感器 [11,12,13]
                temp_sensor_values       # 温度传感器 [14,15,16]
            ])

            # 也保留历史传感器输入用于预测
            if agent.last_sensor_inputs is not None:
                full_inputs = np.concatenate([extended_inputs, agent.last_sensor_inputs])
            else:
                full_inputs = extended_inputs

            all_outputs = agent.genome.forward(full_inputs)

            # ============================================================
            # v7.3: 熵增与腐蚀 - 功能腐蚀
            # ============================================================
            corrosion_stats = agent.genome.apply_functional_corrosion()

            # 确保输出至少有 4 维
            if len(all_outputs) < 4:
                all_outputs = np.pad(
                    all_outputs,
                    (0, 4 - len(all_outputs)),
                    constant_values=0.0
                )

            actuator_outputs = all_outputs[:2]
            agent.predicted_next_sensor = all_outputs[2:4]

            if len(actuator_outputs) < 2:
                actuator_outputs = np.pad(
                    actuator_outputs,
                    (0, 2 - len(actuator_outputs)),
                    constant_values=0.0
                )

            # 收集节点激活值用于赫布学习
            for node_id, node in agent.genome.nodes.items():
                all_node_activations[(agent.id, node_id)] = node.activation

            # 4. 检查是否吃到食物 (多智能体竞争)
            self._check_food_collision(agent)

            # ============================================================
            # v0.78: 巢穴贮粮
            # 如果Agent带着食物回到巢穴，存储起来
            # ============================================================
            if self.nest_enabled and agent.food_carried > 0:
                dist = self._toroidal_distance(
                    agent.x, agent.y,
                    self.nest_position[0], self.nest_position[1]
                )

                if dist < self.nest_radius:
                    # 存储食物到巢穴
                    stored = agent.food_carried
                    # v11.0: 入库税 - 贮粮时一次性扣除
                    tax = int(stored * self.nest_tax) if self.nest_tax > 0 else 0
                    stored_after_tax = stored - tax
                    
                    agent.food_carried = 0
                    agent.food_stored += stored_after_tax
                    self.nest_stored_food += stored_after_tax

            # ============================================================
            # v3.0: 能量夺取 (社会交互)
            # ============================================================
            # 当 Agent A 的推进力大于 B 且发生碰撞时，A 夺取 B 20% 的能量
            self._attempt_energy_theft(agent, actuator_outputs)

            # 5. 能量消耗 (内稳态 + 昼夜税率)
            genome_info = agent.genome.get_info()
            n_nodes = genome_info['total_nodes']
            n_edges = genome_info['enabled_edges']

            # ============================================================
            # v5.2: 预测偏差惩罚 (Prediction Loss Integration)
            # Metabolic_new = Metabolic_base + γ * Σ|Pred(t-1) - Sensor(t)|
            # 强迫PREDICTOR必须真实反映环境变化
            # ============================================================
            gamma = 2.0  # 预测偏差权重
            prediction_penalty = agent.prediction_error * gamma if self.step_count > 1 else 0.0

            # ============================================================
            # v5.4 表型特化: 节点特异性代谢 (非线性能耗)
            # ============================================================
            node_specific_cost = 0.0
            for node_id, node in agent.genome.nodes.items():
                base_cost = self.metabolic_alpha

                # v6.0 GAIA: 基础算子特化能耗
                # 所有特殊能力通过端口输出,代谢与基础算子类型相关
                if node.node_type == NodeType.DELAY:
                    base_cost *= 1.5  # 记忆: 1.5倍
                elif node.node_type == NodeType.PREDICTOR:
                    base_cost *= 1.2  # 预测: 1.2倍
                elif node.node_type == NodeType.PORT_MOTION:
                    base_cost *= 1.3  # 运动端口
                elif node.node_type == NodeType.PORT_OFFENSE:
                    base_cost *= 3.0  # v10.1: 攻击端口能耗提升3倍 (暴力代价，高风险高回报)
                elif node.node_type == NodeType.PORT_DEFENSE:
                    base_cost *= 0.65  # v10.1: 防御端口能耗降低50% (从1.3降到0.65)
                elif node.node_type == NodeType.PORT_REPAIR:
                    base_cost *= 1.5  # 修复端口
                elif node.node_type == NodeType.PORT_SIGNAL:
                    base_cost *= 1.2  # 信号端口

                node_specific_cost += base_cost

            # 基础代谢消耗 + 节点特异性 + 动作消耗 + 预测偏差
            # v5.7 物理规律: 代谢 cost = 结构成本 + 计算成本
            # v10.4: 非线性代谢缩放 - sqrt(节点数) * alpha (异速生长定律)
            # 大脑功耗增长率远低于神经元的增加速度
            node_cost = np.sqrt(n_nodes) * self.metabolic_alpha if n_nodes > 0 else 0
            edge_cost = np.sqrt(n_edges) * self.metabolic_beta if n_edges > 0 else 0
            metabolic_cost = node_cost + edge_cost
            metabolic_cost += node_specific_cost  # 表型特化能耗
            metabolic_cost += prediction_penalty  # 加入预测偏差
            
            # v10.0: 巢穴安全区 - 代谢降至30%
            if self.nest_enabled:
                dist_to_nest = self._toroidal_distance(
                    agent.x, agent.y, self.nest_position[0], self.nest_position[1]
                )
                if dist_to_nest < self.nest_radius:
                    metabolic_cost *= 0.3  # 巢穴内低耗

            # v10.3: 生态避难所 (Refugia) - 地图边缘的稳定区域
            # 在地图左下角(5%区域)和右上角(5%区域)创建稳定区
            refugia_margin = self.width * 0.05  # 5%边缘
            in_refugia = False
            
            # 检查是否在避难所区域 (左下或右上)
            if (agent.x < refugia_margin and agent.y < refugia_margin) or \
               (agent.x > self.width - refugia_margin and agent.y > self.height - refugia_margin):
                in_refugia = True
                metabolic_cost *= 0.5  # 避难所代谢减半

            # 动作越强烈，能量消耗越多 (v0.74: 降低系数以允许更长寿命)
            action_intensity = np.mean(np.abs(actuator_outputs))
            metabolic_cost += action_intensity * 0.05  # 降低50倍

            # ============================================================
            # v0.98: 发育相变 - 幼体保护期代谢调整
            # ============================================================
            if self.ontogenetic_phase_enabled:
                # 更新发育阶段
                agent.age_steps += 1

                # 检查是否需要相变
                if not agent.phase_transition_done and agent.age_steps >= self.juvenile_duration:
                    agent.phase_transition_done = True
                    agent.developmental_stage = "adult"

                # 应用代谢调整
                if agent.developmental_stage == "juvenile":
                    # 幼体: 低代谢
                    metabolic_cost *= self.juvenile_metabolic_rate
                elif agent.phase_transition_done and agent.age_steps == self.juvenile_duration:
                    # 相变瞬间: 代谢飙升 (适应新压力)
                    metabolic_cost *= self.phase_transition_bonus

            # ============================================================
            # v0.97 Stage 1 修复: 疲劳指数级耗能 (Exhaustion Tax)
            # 移动所消耗的能量与疲劳度成非线性正比
            # 公式: 移动耗能 = 基础耗能 * (1 + 疲劳度^2 * 5)
            # 当疲劳度=0.8时，耗能是平时的4倍 (更温和)
            # ============================================================
            speed = getattr(agent, 'speed', 0.0)
            if speed > 0.1 and self.fatigue_system_enabled:
                fatigue_ratio = agent.fatigue / agent.max_fatigue if agent.max_fatigue > 0 else 0
                exhaustion_multiplier = 1.0 + (fatigue_ratio ** 2) * 5.0
                
                # v0.99: 阶段三 - 疲劳惩罚
                # 当疲劳 > 80% 时，基础代谢额外增加30%
                if fatigue_ratio > 0.8:
                    exhaustion_multiplier *= 1.3  # +30% 代谢惩罚
                
                metabolic_cost *= exhaustion_multiplier

            # ============================================================
            # v0.97 Stage 1 修复: 低功耗休眠模式 (Deep Sleep Metabolism)
            # 当Agent静止且正在恢复疲劳(睡觉)时，代谢降到10%
            # ============================================================
            if self.fatigue_system_enabled:
                in_sleep_recovery = getattr(agent, '_in_sleep_recovery', False)
                if in_sleep_recovery and speed < 0.1:
                    metabolic_cost *= 0.1  # 睡觉时代谢降至10%

            # ============================================================
            # v8.0: 无尽边疆 - 探索税
            # 当Agent进入新区块时，PORT_MOTION消耗增加20%
            # ============================================================
            if self.infinite_mode:
                if getattr(agent, 'is_in_new_chunk', False):
                    metabolic_cost *= ChunkManager.EXPLORATION_TAX  # 1.2x
                    agent.is_in_new_chunk = False  # 重置标记

            # v3.0: 昼夜税率
            metabolic_cost *= metabolic_multiplier

            # ============================================================
            # v11.0: 端口干涉代价 (Port Interference Cost)
            # 多端口同时激活时，代谢成本呈指数级上升
            # 公式: Extra_Cost = (Offense × Defense × Carry)^gamma
            # ============================================================
            if self.port_interference_gamma > 0:
                # 获取端口激活值
                off = agent.port_offense
                defe = agent.port_defense
                carry = 1.0 if agent.food_carried > 0 else 0.0
                
                # 计算干涉因子
                interference_product = off * defe * carry
                if interference_product > 0:
                    interference_cost = (interference_product ** self.port_interference_gamma) * 0.5
                    metabolic_cost += interference_cost

            # v5.7 能量守恒: 只能消耗拥有的能量
            actual_cost = min(metabolic_cost, max(0, agent.internal_energy))
            agent.internal_energy -= actual_cost
            agent.energy_spent += actual_cost  # 追踪实际消耗

            # ============================================================
            # v13.0: 热力学能量交换 (Thermodynamic Energy Exchange)
            # 计算渗透膜能量交换 + 移动做功 + 废热排放
            # ============================================================
            if self.energy_field_enabled and self.energy_field:
                # 获取 actuator 输出的移动力
                left_force = getattr(agent, 'left_output', 0.0)
                right_force = getattr(agent, 'right_output', 0.0)
                
                # 应用热力学法则
                energy_stats = self.thermodynamic_law.apply_to_agent(
                    agent=agent,
                    field=self.energy_field,
                    left_force=left_force,
                    right_force=right_force,
                    metabolic_cost=actual_cost
                )
                
                # v13.0: 更新渗透率 (从大脑输出获取)
                # 默认: 端口signal可作为渗透率控制信号
                agent.permeability = max(0.0, min(1.0, agent.port_signal))

            # ============================================================
            # v13.0: 运动阻抗场 (Kinetic Impedance Field)
            # 计算位移抑制 + 梯度反作用力
            # ============================================================
            if self.impedance_field_enabled and self.impedance_field:
                # 获取 actuator 输出的移动力
                left_force = getattr(agent, 'left_output', 0.0)
                right_force = getattr(agent, 'right_output', 0.0)
                
                # 应用阻抗法则
                impedance_info = self.kinetic_impedance_law.apply_to_agent(
                    agent=agent,
                    impedance_field=self.impedance_field,
                    left_force=left_force,
                    right_force=right_force
                )
                
                # 额外的阻抗能耗 (叠加到 ThermodynamicLaw 的 move_cost)
                # 注意: ThermodynamicLaw 已经计算了基础 move_cost
                # 这里我们把阻抗导致的额外能耗也加入
                extra_impedance_cost = impedance_info['move_cost']
                agent.internal_energy -= extra_impedance_cost
                agent.energy_spent += extra_impedance_cost
                
                # 记录阻抗数据供传感器使用
                agent.impedance_value = impedance_info['impedance']
                agent.velocity_actual = impedance_info['velocity']

            # ============================================================
            # v13.0: 压痕场 (Stigmergy Field)
            # Agent 移动时注入信号 + 消耗能量
            # ============================================================
            if self.stigmergy_field_enabled and self.stigmergy_field:
                # 获取上一步位置 (需要在循环开始时记录)
                old_x = getattr(agent, '_stigmergy_last_x', agent.x)
                old_y = getattr(agent, '_stigmergy_last_y', agent.y)
                
                # 应用压痕法则
                stigmergy_info = self.stigmergy_law.apply_to_agent(
                    agent=agent,
                    stigmergy_field=self.stigmergy_field,
                    old_x=old_x,
                    old_y=old_y
                )
                
                # 记录当前位置供下一步使用
                agent._stigmergy_last_x = agent.x
                agent._stigmergy_last_y = agent.y
                
                # 记录信号数据供传感器使用
                agent.stigmergy_value = stigmergy_info['deposited']

            # ============================================================
            # v13.0: 应力场感知 (Stress Field)
            # 采样当前应力 + 时间导数
            # ============================================================
            if self.stress_field_enabled and self.stress_field:
                stress_info = self.stress_law.apply_to_agent(agent, self)
                # 传感器扩展: [当前应力, 应力变化率]
                agent.current_stress = stress_info['stress']
                agent.stress_derivative = stress_info['derivative']

            # ============================================================
            # v11.0: 代谢熵增 (Energy Decay)
            # 体内能量随时间自然挥发，流失率 ∝ E²
            # 公式: Loss = k × Energy_current²
            # ============================================================
            if self.energy_decay_k > 0 and agent.internal_energy > 0:
                energy_decay_loss = self.energy_decay_k * (agent.internal_energy ** 2)
                energy_decay_loss = min(energy_decay_loss, agent.internal_energy * 0.1)  # 最多损失10%
                agent.internal_energy -= energy_decay_loss
                agent.energy_spent += energy_decay_loss
                agent.energy_wasted += energy_decay_loss

            # v5.3/v5.7 追踪代谢浪费
            agent.metabolic_waste += actual_cost

            # ============================================================
            # v9.0 加速衰老 (Age Acceleration)
            # 基础衰老速度提高3倍，只有PORT_REPAIR可以减缓
            # ============================================================
            base_age_increment = 0.15  # v9.0: 3x base (0.1 → 0.3)
            metabolic_age = metabolic_cost * 0.01

            # PORT_REPAIR 持续激活可以延缓衰老
            repair_effect = agent.port_repair * 0.5  # 每单位REPAIR减少0.3年龄

            agent.age += base_age_increment + metabolic_age - repair_effect
            agent.age = max(0, agent.age)  # 年龄不能为负

            # 检查是否老死
            if agent.age >= agent.max_age:
                # v5.7 能量守恒: 死亡时剩余能量散失到环境
                if agent.internal_energy > 0:
                    agent.energy_wasted += agent.internal_energy
                agent.is_alive = False
                agent.internal_energy = 0
                continue

            # 能量耗尽 = 死亡
            if agent.internal_energy <= 0:
                # v5.7 能量守恒: 剩余能量散失到环境
                # 剩余能量 = 之前累积的剩余 (更简单的计算)
                # 实际上: 如果能量归零,说明已耗尽,无剩余可散失
                agent.is_alive = False
                agent.internal_energy = 0
                continue  # 死亡后不再更新

            # 6. 赫布学习更新 (可塑性边)
            # 收集当前 agent 的节点激活
            agent_activations = {node_id: node.activation
                                for node_id, node in agent.genome.nodes.items()}
            agent.genome.hebbian_update(agent_activations, lr=0.01)

            # v0.74: 追踪节点共同激活 (元节点压缩)
            agent_activations = {node_id: node.activation for node_id, node in agent.genome.nodes.items()}
            if not hasattr(agent, 'node_coactivation'):
                agent.node_coactivation = {}

            # 检查哪些节点同时激活
            active_nodes = [nid for nid, act in agent_activations.items() if act > 0.3]
            for i, n1 in enumerate(active_nodes):
                for n2 in active_nodes[i+1:]:
                    key = tuple(sorted([n1, n2]))
                    agent.node_coactivation[key] = agent.node_coactivation.get(key, 0) + 1

            # 7. 更新物理状态
            self._update_agent_physics(agent, actuator_outputs)

            # ============================================================
            # v9.0: 逃逸食物更新
            # ============================================================
            if self.food_escape_enabled and not self.infinite_mode:
                self._update_food_escape(agent)

            # ============================================================
            # v5.3 Battle Royale: 计算战斗信号
            # ============================================================
            self._compute_battle_signals(agent)

        # ============================================================
        # v5.3 Battle Royale: Agent间碰撞与能量掠夺
        # ============================================================
        self._handle_agent_collisions()

        # 8. 记录位置 (Novelty Search)
        for agent in self.agents:
            if agent.is_alive:
                agent.record_position(self.width, self.height)

            # 9. 更新适应度
            # 到最近食物的距离 (v8.0: 无限模式下从区块获取)
            if self.infinite_mode and self.chunk_manager:
                foods = self.chunk_manager.get_foods_in_range(agent.x, agent.y, self.sensor_range)
                if foods:
                    min_dist = min(fp[2] for fp in foods)  # fp[2] is distance
                else:
                    min_dist = self.sensor_range * 2  # 无食物时给较大值
            else:
                if self.food_positions:
                    min_dist = min(
                        np.sqrt((agent.x - fp[0])**2 + (agent.y - fp[1])**2)
                        for fp in self.food_positions
                    )
                else:
                    min_dist = self.sensor_range * 2

            surprise_cost = agent.prediction_error * self.surprise_penalty

            # ============================================================
            # v13.0: 极简热力学适应度
            # 适应度 = 内部能量 (存活时)
            # 死了 = 极低分 (仅给一点挣扎分)
            # ============================================================
            if agent.is_alive:
                # 活着: 适应度 = 体内积蓄的能量 (可用于繁衍的资本)
                agent.fitness = agent.internal_energy
            else:
                # 死了: 仅给一点挣扎的同情分
                agent.fitness = agent.steps_alive * 0.01

        # 清理死亡 agent (可选，保留以便统计)

    def apply_dynamic_pressure(self, generation: int):
        """
        应用动态环境压力 (Malthusian Trap)

        每过 10 代:
        - 增加代谢惩罚 (alpha, beta)
        - 减少感知范围 (sensor_range)
        """
        self.generation = generation

        # 每 10 代提高难度
        difficulty = generation // 10

        # 代谢惩罚增加
        self.metabolic_alpha = self.base_metabolic_alpha * (1 + 0.2 * difficulty)
        self.metabolic_beta = self.base_metabolic_beta * (1 + 0.2 * difficulty)

        # ============================================================
        # v11.0: 季节波动率 (Season Jitter)
        # 每代环境参数 ±X% 随机扰动，模拟"气候变化"
        # 这会让Agent无法"过拟合"固定环境，必须学会预测
        # ============================================================
        if self.season_jitter > 0:
            # 扰动幅度
            jitter = self.season_jitter
            
            # 季节长度扰动 ±10%
            self.season_length = int(self.base_season_length * (1 + np.random.uniform(-jitter, jitter)))
            self.season_length = max(10, self.season_length)  # 至少10帧
            
            # 冬天代谢倍率扰动 ±10%
            self.winter_metabolic_multiplier = self.base_winter_metabolic_multiplier * (1 + np.random.uniform(-jitter, jitter))
            self.winter_metabolic_multiplier = max(1.0, self.winter_metabolic_multiplier)  # 至少1x
            
            # 能量衰减系数扰动 ±10%
            self.energy_decay_k = self.base_energy_decay_k * (1 + np.random.uniform(-jitter, jitter))
            self.energy_decay_k = max(0, self.energy_decay_k)
            
            # 端口干涉gamma扰动 ±10%
            self.port_interference_gamma = self.base_port_interference_gamma * (1 + np.random.uniform(-jitter, jitter))
            self.port_interference_gamma = max(0, self.port_interference_gamma)
            
            # 入库税扰动 ±10%
            self.nest_tax = self.base_nest_tax * (1 + np.random.uniform(-jitter, jitter))
            self.nest_tax = max(0, min(0.5, self.nest_tax))  # 0-50%

            # 大灭绝周期: 每50代有一次超长冬天
            if generation % 50 == 0 and generation > 0:
                # 超长冬天概率 30%
                if np.random.random() < 0.3:
                    self.winter_metabolic_multiplier *= 2.0  # 2倍冬天
                    self.season_length = int(self.season_length * 1.5)  # 1.5倍季节长度

        # 感知范围缩小 (最低 50)
        self.sensor_range = max(50, self.base_sensor_range * (1 - 0.1 * difficulty))

        # 随机移动食物 (增加不确定性)
        if difficulty > 0 and np.random.random() < 0.3:
            margin = 10.0
            self.target_pos = (
                np.random.uniform(margin, self.width - margin),
                np.random.uniform(margin, self.height - margin)
            )

    def get_state(self) -> Dict:
        """获取环境当前状态 (用于调试/可视化)"""
        return {
            'width': self.width,
            'height': self.height,
            'target_pos': self.target_pos,
            'num_agents': len(self.agents),
            'agents': [
                {
                    'id': a.id,
                    'x': a.x,
                    'y': a.y,
                    'theta': a.theta,
                    'fitness': a.fitness
                }
                for a in self.agents
            ]
        }


# ============================================================
# 5. 种群演化 (Population Evolution)
# ============================================================
# 5. 子图提取与冻结系统 (Subgraph Freezing)
# ============================================================

