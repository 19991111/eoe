from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from .node import NodeType

# 尝试导入性能优化函数
try:
    from ..core import compute_sensor_vectorized, update_positions_vectorized, compute_distances_vectorized, NUMBA_AVAILABLE
except ImportError:
    # 如果core未加载，使用纯numpy版本 (环形世界感知)
    NUMBA_AVAILABLE = False
    
    def _toroidal_distance_vec(dx, dy, width, height):
        """计算环形世界距离向量"""
        dx = dx - width * np.floor(dx / width + 0.5)
        dy = dy - height * np.floor(dy / height + 0.5)
        return dx, dy
    
    def compute_distances_vectorized(x1, y1, x2_arr, y2_arr, width=100.0, height=100.0):
        """向量化计算环形世界距离矩阵"""
        dx = x2_arr - x1
        dy = y2_arr - y1
        dx = dx - width * np.floor(dx / width + 0.5)
        dy = dy - height * np.floor(dy / height + 0.5)
        return np.sqrt(dx**2 + dy**2)
    
    def update_positions_vectorized(x, y, theta, left_force, right_force, max_speed, turn_rate, width, height):
        diff = right_force - left_force
        new_theta = (theta + diff * turn_rate) % (2 * np.pi)
        speed = np.clip((left_force + right_force) / 2.0, -max_speed, max_speed)
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
        decay = np.clip(sensor_range / (dist + 1.0), 0.0, 1.0)
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
        metabolic_alpha: float = 0.05,
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
        winter_metabolic_multiplier: float = 2.0
    ):
        self.width = width
        self.height = height
        self.pure_survival_mode = pure_survival_mode  # v0.74
        self.agents: List[Agent] = []
        
        # v5.2: 步数计数器
        self.step_count = 0  # 用于预测偏差计算
        
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
        self.food_escape_speed = 1.2  # 逃逸速度 (Agent max=2.0)
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
        self.winter_food_multiplier = winter_food_multiplier
        self.winter_metabolic_multiplier = winter_metabolic_multiplier
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
        self.sensor_range = 30.0  # 默认传感器范围
        
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
        v9.0: 逃逸食物逻辑
        当Agent接近食物时，食物向反方向逃逸
        """
        if self.food_escape_cooldown > 0:
            self.food_escape_cooldown -= 1
            return
        
        for i, (fx, fy) in enumerate(self.food_positions):
            # 计算到最近Agent的距离
            min_dist = float('inf')
            for agent in self.agents:
                if not agent.is_alive:
                    continue
                dist = np.sqrt((agent.x - fx)**2 + (agent.y - fy)**2)
                if dist < min_dist:
                    min_dist = dist
            
            # 如果Agent进入逃逸范围
            if min_dist < self.food_escape_range:
                # 找到最近的Agent
                nearest_agent = min(self.agents, 
                    key=lambda a: np.sqrt((a.x - fx)**2 + (a.y - fy)**2) if a.is_alive else float('inf'))
                
                # 计算逃逸方向 (远离Agent)
                dx = fx - nearest_agent.x
                dy = fy - nearest_agent.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    dx, dy = dx / dist, dy / dist
                
                # 更新食物速度 (平滑加速)
                current_vx, current_vy = self.food_velocities[i]
                new_vx = current_vx * 0.8 + dx * self.food_escape_speed * 0.2
                new_vy = current_vy * 0.8 + dy * self.food_escape_speed * 0.2
                self.food_velocities[i] = (new_vx, new_vy)
                
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
            else:
                # 逐渐减速
                vx, vy = self.food_velocities[i]
                self.food_velocities[i] = (vx * 0.95, vy * 0.95)
    
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
                    
                    # 增加贮粮计数
                    agent.food_stored += agent.food_carried
                    
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
    
    def _apply_sensor_noise(self, sensor_values: np.ndarray) -> np.ndarray:
        """
        v7.1: 对传感器添加高斯噪声
        """
        if self.sensor_noise_level > 0:
            noise = np.random.normal(0, self.sensor_noise_level, sensor_values.shape)
            sensor_values = sensor_values + noise
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
        """食物被吃掉后重生"""
        if not self.respawn_food:
            return
        
        # v0.78: 冬天不生成食物
        if self.seasonal_cycle and self.current_season == "winter":
            # 冬天没有新食物
            return
        
        # 随机选择一个已消失的位置重生
        margin = 15.0
        pos = (
            np.random.uniform(margin, self.width - margin),
            np.random.uniform(margin, self.height - margin)
        )
        # 找到被吃掉的最近食物位置替换
        # 简化: 随机替换一个
        if self.food_positions:
            idx = np.random.randint(len(self.food_positions))
            self.food_positions[idx] = pos
    
    def _check_food_collision(self, agent: Agent) -> bool:
        """
        检查 Agent 是否吃到食物
        
        v5.4: 应用能量吸收率 η_plant (80%)
        """
        eat_distance = 3.0  # 吃到的距离阈值
        
        for i, food_pos in enumerate(self.food_positions):
            dx = agent.x - food_pos[0]
            dy = agent.y - food_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < eat_distance:
                # 吃到食物!
                if self.nest_enabled:
                    # v0.78: 巢穴模式 - 携带食物回巢
                    agent.food_carried += 1
                else:
                    # 原有模式 - 直接吃掉
                    agent.food_eaten += 1
                
                # v5.4/v5.7 能量守恒: 应用吸收率 η_plant
                # 能量守恒定律: E_total = E_absorbed + E_wasted
                absorbed_energy = self.food_energy * agent.eta_plant
                wasted_energy = self.food_energy * (1 - agent.eta_plant)
                
                agent.internal_energy = min(
                    agent.max_energy, 
                    agent.internal_energy + absorbed_energy
                )
                agent.energy_gained += absorbed_energy  # 追踪获得能量
                agent.energy_wasted += wasted_energy  # 散失到环境
                
                # 食物重生
                self._respawn_food()
                return True
        
        return False
    
    def _attempt_energy_theft(self, agent: Agent, actuator_outputs: np.ndarray) -> None:
        """
        v3.0: 能量夺取逻辑
        
        当 Agent A 的推进力大于 B 且距离足够近时:
        - A 夺取 B 20% 的能量
        - 仅当 A 的总推力 > B 的总推力时发生
        """
        my_force = np.sum(np.abs(actuator_outputs))
        collision_dist = 5.0  # 碰撞距离
        
        for other in self.agents:
            if other is agent or not other.is_alive:
                continue
            
            dist = np.sqrt((agent.x - other.x)**2 + (agent.y - other.y)**2)
            
            if dist < collision_dist:
                other_force = 0.0  # 简化: 假设其他 Agent 推力为0
                
                # 如果自己的推力更大，夺取能量
                if my_force > other_force and other.internal_energy > 10:
                    steal_amount = other.internal_energy * 0.20  # 夺取20%
                    agent.internal_energy = min(
                        agent.max_energy,
                        agent.internal_energy + steal_amount
                    )
                    other.internal_energy -= steal_amount
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
            
            # 1. 食物 (type=0)
            for fx, fy in self.food_positions[:5]:  # 最多5个
                dx, dy = fx - agent.x, fy - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist < sensor_range:
                    angle = np.degrees(np.arctan2(dy, dx) - agent.theta) % 360
                    targets.append({
                        'angle': angle,
                        'distance': dist,
                        'type': 0  # food
                    })
            
            # 2. 巢穴 (type=1)
            if getattr(self, 'nest_enabled', False):
                nx, ny = self.nest_position
                dx, dy = nx - agent.x, ny - agent.y
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
            
            return np.array(sensor_values[:2] if len(sensor_values) >= 2 else sensor_values + [0.0])
        
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
            fatigue_sensor = np.clip(fatigue_sensor, 0.0, 1.0)
            sensor_values = np.append(sensor_values, fatigue_sensor)
        
        return sensor_values
    
    def _compute_light_sensor(self, agent: Agent) -> np.ndarray:
        """
        v7.0: 移动光源传感器
        
        返回 [左光源传感器, 右光源传感器]
        
        光源位置动态变化，迫使Agent:
        - 记住光源的历史位置
        - 使用DELAY回路来预测光源轨迹
        """
        # 目标向量 (光源)
        dx = self.light_pos[0] - agent.x
        dy = self.light_pos[1] - agent.y
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
        
        # 距离衰减
        distance_decay = self.sensor_range / (distance + 1.0)
        distance_decay = np.clip(distance_decay, 0.0, 1.0)
        
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
        # 相对于原点的位移
        rel_x = agent.x - self.origin_x
        rel_y = agent.y - self.origin_y
        
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
        
        # v6.0 GAIA: 标准化到 [0, 1]
        agent.port_motion = np.clip(motion, 0, 1)
        agent.port_offense = np.clip(offense, 0, 1)
        agent.port_defense = np.clip(defense, 0, 1)
        agent.port_repair = np.clip(repair, 0, 1)
        agent.port_signal = np.clip(signal, 0, 1)
        
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
        v6.0 GAIA: 解决单次战斗
        
        - 攻击方: PORT_OFFENSE 高频时触发
        - 防御方: PORT_DEFENSE 激活时减伤
        - 胜者判定: 攻击方攻击 > 防御方防御 → 攻击方胜
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
        
        # 防御修正: SHIELD节点减免伤害 (但不能完全阻挡)
        effective_defense_a = min(defense_a * 0.5, 0.8)  # 最多减免80%
        effective_defense_b = min(defense_b * 0.5, 0.8)
        
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
        
        # 检查Epic Motif: 3连胜
        if winner.battle_wins >= 3:
            winner.is_epic_motif = True
        
        # 淘汰: 能量归零
        if loser.internal_energy <= 0:
            loser.is_alive = False
            loser.internal_energy = 0
    
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
        
        # ============================================================
        # v4.0: 更新诱饵位置
        # ============================================================
        if self.n_bait > 0:
            self._update_bait()
        
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
            
            # 3. 前向传播 - 拼接所有传感器输入
            extended_inputs = np.concatenate([
                sensor_values,           # 食物传感器 [0,1]
                light_sensor_values,     # 光源传感器 [2,3]
                agent_radar_values,      # 社会传感器 [4,5,6]
                gps_sensor_values,       # GPS传感器 [7,8,9,10]
                compass_sensor_values    # 指南针传感器 [11,12,13]
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
                dx = agent.x - self.nest_position[0]
                dy = agent.y - self.nest_position[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < self.nest_radius:
                    # 存储食物到巢穴
                    stored = agent.food_carried
                    agent.food_carried = 0
                    agent.food_stored += stored
                    self.nest_stored_food += stored
            
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
                    base_cost *= 1.4  # 攻击端口
                elif node.node_type == NodeType.PORT_DEFENSE:
                    base_cost *= 1.3  # 防御端口
                elif node.node_type == NodeType.PORT_REPAIR:
                    base_cost *= 1.5  # 修复端口
                elif node.node_type == NodeType.PORT_SIGNAL:
                    base_cost *= 1.2  # 信号端口
                
                node_specific_cost += base_cost
            
            # 基础代谢消耗 + 节点特异性 + 动作消耗 + 预测偏差
            # v5.7 物理规律: 代谢 cost = 结构成本 + 计算成本
            metabolic_cost = n_nodes * self.metabolic_alpha + n_edges * self.metabolic_beta
            metabolic_cost += node_specific_cost  # 表型特化能耗
            metabolic_cost += prediction_penalty  # 加入预测偏差
            
            # 动作越强烈，能量消耗越多 (v0.74: 降低系数以允许更长寿命)
            action_intensity = np.mean(np.abs(actuator_outputs))
            metabolic_cost += action_intensity * 0.05  # 降低50倍
            
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
            
            # v5.7 能量守恒: 只能消耗拥有的能量
            actual_cost = min(metabolic_cost, max(0, agent.internal_energy))
            agent.internal_energy -= actual_cost
            agent.energy_spent += actual_cost  # 追踪实际消耗
            
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
            # v5.1: 适应度硬化 (Fitness Hardening)
            # 饥饿开关: 如果Food_Eaten==0,屏蔽Surprise和Distance加分
            # ============================================================
            if agent.food_eaten == 0:
                # 饥饿模式: 仅基础代谢惩罚,强制捕食突破
                # v9.1: 添加存活奖励，鼓励存活
                survival_bonus = agent.steps_alive * 1.0  # 每步存活+1分
                # v9.3: 添加复杂度奖励 - 鼓励内部算子 (DELAY特别奖励)
                internal_nodes = [n for n in agent.genome.nodes.values() 
                                 if n.node_type.name in ['ADD', 'MULTIPLY', 'DELAY', 'THRESHOLD']]
                # DELAY gets extra bonus for latency wall
                delay_nodes = [n for n in agent.genome.nodes.values() if n.node_type.name == 'DELAY']
                complexity_bonus = len(internal_nodes) * 15.0 + len(delay_nodes) * 25.0
                agent.fitness = -metabolic_cost + survival_bonus + complexity_bonus
            else:
                # v0.74: 纯生存适应度模式
                if getattr(self, 'pure_survival_mode', False):
                    agent.fitness = agent.food_eaten * 1000.0
                    if agent.steps_alive >= agent.lifespan and agent.food_eaten == 0:
                        agent.fitness = -500.0
                else:
                    # 捕食模式: 完整适应度
                    food_bonus = agent.food_eaten * 100.0
                # v9.1: 添加存活奖励
                survival_bonus = agent.steps_alive * 1.0
                # v9.3: 添加复杂度奖励
                internal_nodes = [n for n in agent.genome.nodes.values() 
                                 if n.node_type.name in ['ADD', 'MULTIPLY', 'DELAY', 'THRESHOLD']]
                complexity_bonus = len(internal_nodes) * 10.0
                agent.fitness = -min_dist - metabolic_cost - surprise_cost + food_bonus + survival_bonus + complexity_bonus
        
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

