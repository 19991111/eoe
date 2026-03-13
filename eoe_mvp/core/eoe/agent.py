from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np

from .node import Node, NodeType
from .genome import OperatorGenome

class Agent:
    """
    在沙盒中游动的智能体
    
    属性:
        x, y: 当前坐标
        theta: 朝向角度 (弧度)
        fitness: 适应度得分
        genome: 大脑 (OperatorGenome 实例)
        id: 唯一标识符
        prediction_error: 预测误差 (Surprise)
        last_sensor_inputs: 上一次的传感器输入
        predicted_next_sensor: 预测的下一步传感器值
        surprise_accumulated: 累积 Surprise 值
    """
    
    # 传感器和执行器的默认数量
    NUM_SENSORS = 2    # 左右触角
    NUM_ACTUATORS = 2  # 左右推进器
    
    def __init__(
        self, 
        agent_id: int,
        x: float = 50.0, 
        y: float = 50.0, 
        theta: float = 0.0,
        add_predictors: bool = True,  # 是否添加预测节点
        add_light_sensor: bool = False,  # v7.0: 光源传感器 (默认关闭)
        add_agent_radar: bool = False,  # v7.0: 社会传感器 (默认关闭)
        add_gps_sensors: bool = False  # v8.0: GPS传感器 (默认关闭)
    ):
        self.id = agent_id
        self.x = x
        self.y = y
        self.theta = theta
        self.fitness = 0.0
        
        # 预测误差跟踪 (Surprise 机制)
        self.prediction_error: float = 0.0
        self.last_sensor_inputs: Optional[np.ndarray] = None
        self.predicted_next_sensor: Optional[np.ndarray] = None
        self.surprise_accumulated: float = 0.0
        
        # ============================================================
        # v9.0 感官延迟墙 (The Latency Wall)
        # 传感器延迟: SENSOR看到的永远是过去的数据
        # ============================================================
        self.sensor_delay_frames: int = 10  # 30帧延迟 = 1秒(在30fps)
        self.sensor_history: deque = deque(maxlen=self.sensor_delay_frames)
        
        # Novelty Search: 行为追踪
        self.visited_positions: List[Tuple[float, float]] = []  # 记录到访位置
        self.behavior_grid: Optional[np.ndarray] = None         # 5x5 行为网格
        self.novelty_score: float = 0.0                          # 独特度分数
        
        # ============================================================
        # 内稳态 (Homeostasis) - 能量池系统
        # ============================================================
        self.internal_energy: float = 150.0   # 初始能量 (可用完)
        self.max_energy: float = 100.0        # 最大能量上限
        self.is_alive: bool = True            # 存活状态
        self.steps_alive: int = 0             # 存活步数
        
        # ============================================================
        # v13.0: 统一场物理 - 兼容旧食物系统 (逐步移除)
        # 保留属性以兼容环境中的旧逻辑
        # 能量获取主要通过 EPF 能量场交换
        # ============================================================
        self.food_carried: int = 0            # 携带的食物 (旧系统兼容)
        self.food_stored: int = 0             # 存储的食物 (旧系统兼容)
        
        # ============================================================
        # v13.0: 兼容旧代谢系统 (逐步移除)
        # ============================================================
        self.metabolic_waste: float = 0.0     # 代谢浪费 (旧系统兼容)
        self.attack_power: float = 0.0        # 攻击信号量 (旧系统兼容)
        self.defense_power: float = 0.0       # 防御信号量 (旧系统兼容)
        
        # ============================================================
        # v6.0 GAIA 寿命与年龄系统 (Thermodynamics)
        # ============================================================
        self.age: float = 0.0           # 生理年龄 (增长与代谢率成正比)
        self.max_age: float = 100.0     # 最大寿命
        self.young_threshold: float = 20.0   # 幼年期 (20%)
        self.mature_threshold: float = 60.0  # 壮年期 (60%)
        
        # v6.0 繁衍阈值: 能量>200%且Age在[20%,60%]
        self.reproduction_energy_threshold: float = 200.0  # 200%初始能量
        
        # ============================================================
        # v5.7 能量守恒定律 (Physical Laws)
        # ============================================================
        # 能量收支追踪 (必须平衡)
        self.initial_energy: float = 150.0  # 初始能量
        self.energy_gained: float = 0.0     # 获得的总能量 (含继承)
        self.energy_spent: float = 0.0      # 消耗的总能量
        self.energy_wasted: float = 0.0     # 散失到环境的能量
        self.energy_battle_stolen: float = 0.0  # 战斗中窃取的能量
        
        # 物理规律标志
        self.energy_conservation_error: float = 0.0  # 能量不守恒误差
        
        # v6.0 GAIA 物理输出端口值 (基础算子连接到这些端口)
        self.port_motion: float = 0.0      # 移动 [速度, 转向]
        self.port_offense: float = 0.0     # 攻击强度
        self.port_defense: float = 0.0     # 防御硬度
        self.port_repair: float = 0.0      # 修复输出
        self.port_signal: float = 0.0      # 信号输出
        
        # v6.0 GAIA 进化监控指标
        self.prediction_horizon: float = 0.0    # 预测深度 (PREDICTOR/DELAY环路)
        self.metabolic_efficiency: float = 0.0  # 代谢效率 (能量获取/消耗)
        self.niche_type: str = "general"        # 生态位类型
        self.out_offense: float = 0.0    # 攻击强度
        self.out_defense: float = 0.0    # 防御硬度
        
        # ============================================================
        # v5.4 热力学: 能量吸收率 (Energy Thermodynamics)
        # ============================================================
        self.eta_plant: float = 0.9    # v6.0 GAIA 食物吸收率 (90%)
        self.eta_meat: float = 0.7     # 肉食吸收率 (70%)
        self.energy_wasted: float = 0.0  # 散失到环境的能量
        
        # ============================================================
        # v5.4 物种形成: 生态位系统
        # ============================================================
        self.species_id: int = 0         # 物种标签
        self.niche_type: str = "general" # 生态位: herbivore/carnivore/scavenger/general
        self.genome_distance: float = 0.0  # 与物种原型的距离
        
        # ============================================================
        # 赫布学习: 记录节点激活用于学习
        # ============================================================
        self.node_activations: Dict[int, float] = {}  # 每步的节点激活
        
        # ============================================================
        # v13.0: 行为状态解构
        # 旧的状态标签 (is_sleeping, battle_wins) 已删除
        # 睡眠涌现: F=0 且 κ=0 时代谢自动降低
        # 战斗涌现: 近距离高 κ 尝试能量窃取
        # ============================================================
        
        # v13.0: 保留必要属性
        self.survival_time: float = 0.0      # 生存时间
        
        # ============================================================
        
        # ============================================================
        self.body_temperature: float = 25.0  # 体温
        self.temperature_sensors: List[float] = [0.5, 0.5, 0.5]  # [左温度, 右温度, 舒适度]
        
        # ============================================================
        
        # ============================================================
        self.developmental_stage: str = "juvenile"  # "juvenile" 或 "adult"
        self.age_steps: int = 0                # 出生后的步数
        self.phase_transition_done: bool = False  # 是否已完成相变
        
        # ============================================================
        # v13.0: 统一场物理系统 - 控制器输出 (Controller Outputs)
        # 四个纯物理参数，由大脑神经网络直接控制
        # ============================================================
        
        # κ (Kappa) 渗透率 ∈ [0, 1]
        # - κ = 0: 封闭，不与环境交换能量
        # - κ = 1: 完全开放，与环境能量场平衡
        # 热力学公式: ΔE = κ × (E_env - E_agent)
        #   - 高能区: 能量流入 (进食)
        #   - 低能区: 能量自然流出 (排泄/失血)
        self.permeability: float = 0.0
        
        # F (Force) 推力矢量 ∈ [-1, 1]²
        # 控制智能体在空间中的受力方向和大小
        self.thrust_vector: Tuple[float, float] = (0.0, 0.0)
        
        # λ (Lambda) 信号强度 ∈ [0, 1]
        # 独立于移动的信号释放，用于 ISF 信息场
        # 允许"隐身策略" (移动但不释放信号)
        self.signal_intensity: float = 0.0
        
        # S (Stiffness) 防御刚性 ∈ [0, 1]
        # 控制对外部能量剥夺的抵抗力
        self.defense_rigidity: float = 0.0
        
        # 物理状态缓存
        self.field_energy: float = 0.0         # 所在位置的环境能量 (采样值)
        self.velocity_actual: float = 0.0      # 实际速度 (经阻抗调制)
        
        # 创建大脑
        self.genome = OperatorGenome()
        
        # 初始化基础节点：2个传感器 + 2个执行器 ± 2个预测器
        self._init_base_nodes(
            add_predictors=add_predictors,
            add_light_sensor=add_light_sensor,
            add_agent_radar=add_agent_radar,
            add_gps_sensors=add_gps_sensors
        )
    
    def update_physics_states(self, brain_output: np.ndarray) -> None:
        """
        v13.0 神经拓扑对接 - 将大脑输出映射到物理状态
        
        假设 brain_output 是神经网络最后一层的输出向量 (5维):
            [0] κ (permeability)    ∈ [0, 1]  - Sigmoid 激活
            [1] Fx (thrust_x)       ∈ [-1, 1] - Tanh 激活
            [2] Fy (thrust_y)       ∈ [-1, 1] - Tanh 激活  
            [3] λ (signal)          ∈ [0, 1]  - ReLU + Clamp
            [4] S (stiffness)       ∈ [0, 1]  - Sigmoid 激活
            
        物理语义:
            κ → 毛孔开合程度 (能量交换率)
            F  → 肌肉收缩方向 (空间受力)
            λ  → 信息素释放强度 (独立于移动)
            S  → 防御刚性 (抗能量剥夺)
        """
        if brain_output is None or len(brain_output) < 5:
            # 默认状态：封闭、不动、无信号、软防御
            self.permeability = 0.0
            self.thrust_vector = (0.0, 0.0)
            self.signal_intensity = 0.0
            self.defense_rigidity = 0.0
            return
        
        # ============================================================
        # v13.0: 激活函数钳制 (确保物理合法性)
        # ============================================================
        
        # 1. 渗透率 κ ∈ [0, 1] - Sigmoid 激活
        raw_kappa = brain_output[0]
        self.permeability = 1.0 / (1.0 + np.exp(-np.clip(raw_kappa, -500, 500)))
        
        # 2. 推力矢量 Fx, Fy ∈ [-1, 1] - Tanh 激活
        raw_fx = brain_output[1] if len(brain_output) > 1 else 0.0
        raw_fy = brain_output[2] if len(brain_output) > 2 else 0.0
        self.thrust_vector = (np.tanh(raw_fx), np.tanh(raw_fy))
        
        # 3. 信号强度 λ ∈ [0, 1] - ReLU + Clamp (独立于移动!)
        raw_signal = brain_output[3] if len(brain_output) > 3 else 0.0
        self.signal_intensity = max(0.0, min(1.0, raw_signal))
        
        # 4. 防御刚性 S ∈ [0, 1] - Sigmoid 激活
        raw_stiffness = brain_output[4] if len(brain_output) > 4 else 0.0
        self.defense_rigidity = 1.0 / (1.0 + np.exp(-np.clip(raw_stiffness, -500, 500)))
    
    def get_controller_outputs(self) -> dict:
        """
        v13.0 获取控制器输出 (供 ThermodynamicLaw 使用)
        """
        return {
            'permeability': self.permeability,      # κ
            'thrust': self.thrust_vector,           # F
            'signal': self.signal_intensity,        # λ
            'stiffness': self.defense_rigidity      # S
        }
    
    def _init_base_nodes(self, add_predictors: bool = True, add_light_sensor: bool = True, add_agent_radar: bool = True, add_gps_sensors: bool = True) -> None:
        """
        v7.0 初始化基础节点结构 (空间记忆 + 社会智能)
        
        节点 ID 分配:
            0: 左传感器 (SENSOR)
            1: 右传感器 (SENSOR)
            2: 左执行器 (ACTUATOR) - 兼容旧版
            3: 右执行器 (ACTUATOR) - 兼容旧版
            4: 左预测器 (PREDICTOR)
            5: 右预测器 (PREDICTOR)
            
        v5.6 物理输出端口 (新范式):
            100: PORT_MOTION (运动速度)
            101: PORT_MOTION (转向力矩)
            102: PORT_REPAIR (修复通道)
            103: PORT_OFFENSE (攻击强度)
            104: PORT_DEFENSE (防御硬度)
        """
        # 传感器节点
        left_sensor = Node(node_id=0, node_type=NodeType.SENSOR, name="left_sensor")
        right_sensor = Node(node_id=1, node_type=NodeType.SENSOR, name="right_sensor")
        
        # 执行器节点 (保持兼容)
        left_actuator = Node(node_id=2, node_type=NodeType.ACTUATOR, name="left_actuator")
        right_actuator = Node(node_id=3, node_type=NodeType.ACTUATOR, name="right_actuator")
        
        # 添加到基因组
        self.genome.add_node(left_sensor)
        self.genome.add_node(right_sensor)
        self.genome.add_node(left_actuator)
        self.genome.add_node(right_actuator)
        
        # 预测节点 (PREDICTOR)
        if add_predictors:
            left_predictor = Node(node_id=4, node_type=NodeType.PREDICTOR, name="left_predictor")
            right_predictor = Node(node_id=5, node_type=NodeType.PREDICTOR, name="right_predictor")
            self.genome.add_node(left_predictor)
            self.genome.add_node(right_predictor)
        
        # v5.6 物理输出端口
        out_velocity = Node(node_id=100, node_type=NodeType.PORT_MOTION, name="PORT_MOTION")
        out_steer = Node(node_id=101, node_type=NodeType.PORT_MOTION, name="PORT_MOTION")
        out_repair = Node(node_id=102, node_type=NodeType.PORT_REPAIR, name="PORT_REPAIR")
        out_offense = Node(node_id=103, node_type=NodeType.PORT_OFFENSE, name="PORT_OFFENSE")
        out_defense = Node(node_id=104, node_type=NodeType.PORT_DEFENSE, name="PORT_DEFENSE")
        
        self.genome.add_node(out_velocity)
        self.genome.add_node(out_steer)
        self.genome.add_node(out_repair)
        self.genome.add_node(out_offense)
        self.genome.add_node(out_defense)
        
        # v7.0: 新增传感器 (可选添加)
        # 6: 左光源传感器, 7: 右光源传感器
        # 8,9,10: Agent雷达传感器 (社会信号)
        if add_light_sensor:
            left_light = Node(node_id=6, node_type=NodeType.LIGHT_SENSOR, name="left_light_sensor")
            right_light = Node(node_id=7, node_type=NodeType.LIGHT_SENSOR, name="right_light_sensor")
            self.genome.add_node(left_light)
            self.genome.add_node(right_light)
        
        if add_agent_radar:
            # 3个社会信号传感器槽位
            for i in range(3):
                radar = Node(node_id=8+i, node_type=NodeType.AGENT_RADAR_SENSOR, name=f"agent_radar_{i}")
                self.genome.add_node(radar)
        
        # v8.0: GPS和指南针传感器 (无尽边疆)
        if add_gps_sensors:
            # 11: GPS_X, 12: GPS_Y, 13: GPS_DIST, 14: GPS_BEARING
            # 15,16,17: COMPASS传感器
            gps_x = Node(node_id=11, node_type=NodeType.GPS_SENSOR, name="gps_x")
            gps_y = Node(node_id=12, node_type=NodeType.GPS_SENSOR, name="gps_y") 
            gps_dist = Node(node_id=13, node_type=NodeType.GPS_SENSOR, name="gps_distance")
            gps_bearing = Node(node_id=14, node_type=NodeType.GPS_SENSOR, name="gps_bearing")
            
            compass_0 = Node(node_id=15, node_type=NodeType.COMPASS_SENSOR, name="compass_0")
            compass_1 = Node(node_id=16, node_type=NodeType.COMPASS_SENSOR, name="compass_1")
            compass_2 = Node(node_id=17, node_type=NodeType.COMPASS_SENSOR, name="compass_2")
            
            self.genome.add_node(gps_x)
            self.genome.add_node(gps_y)
            self.genome.add_node(gps_dist)
            self.genome.add_node(gps_bearing)
            self.genome.add_node(compass_0)
            self.genome.add_node(compass_1)
            self.genome.add_node(compass_2)
        
        # 初始化随机边 - "初始突触风暴"
        # 让神经网络能够产生动作，而非永远输出0
        self._init_random_edges(min_edges=3, max_edges=8)
    
    def _init_random_edges(self, min_edges: int = 3, max_edges: int = 8):
        """
        初始化随机边 - 为神经网络注入初始连接
        
        这是"演化算法的标准做法"：
        - 随机选择输入节点（传感器/预测器）
        - 随机选择输出节点（执行器）
        - 随机权重 (-1.0 到 1.0)
        
        绝对不硬编码特定的神经回路，保持"盲盒"原则。
        """
        import random as _random
        
        # 获取所有合法的输入/输出节点
        input_nodes = []
        output_nodes = []
        
        for node_id, node in self.genome.nodes.items():
            if node.node_type in [NodeType.SENSOR, NodeType.PREDICTOR]:
                input_nodes.append(node_id)
            elif node.node_type == NodeType.ACTUATOR:
                output_nodes.append(node_id)
        
        if not input_nodes or not output_nodes:
            return  # 无法创建边
        
        # 随机生成 3-8 条边
        n_edges = _random.randint(min_edges, max_edges)
        
        for _ in range(n_edges):
            try:
                source = _random.choice(input_nodes)
                target = _random.choice(output_nodes)
                weight = _random.uniform(-1.0, 1.0)
                
                # 添加边（可能失败如果连接已存在）
                self.genome.add_edge(source, target, weight)
            except ValueError:
                pass  # 连接已存在，跳过
        
        # 重置拓扑排序
        self.genome._topo_order = None
    
    def get_sensor_ids(self) -> Tuple[int, int]:
        """获取左右传感器的节点 ID"""
        return (0, 1)
    
    def get_actuator_ids(self) -> Tuple[int, int]:
        """获取左右执行器的节点 ID"""
        return (2, 3)
    
    # ============================================================
    # Novelty Search: 行为追踪与独特度计算
    # ============================================================
    
    def record_position(self, width: float = 100.0, height: float = 100.0, grid_size: int = 5):
        """
        记录当前位置到行为历史
        
        参数:
            width, height: 环境尺寸
            grid_size: 行为网格大小 (默认 5x5)
        """
        self.visited_positions.append((self.x, self.y))
        
        # 实时更新行为网格
        self.behavior_grid = self._compute_behavior_grid(width, height, grid_size)
    
    def _compute_behavior_grid(
        self, 
        width: float, 
        height: float, 
        grid_size: int = 5
    ) -> np.ndarray:
        """
        计算行为网格 - 5x5 量化地图到访统计
        """
        grid = np.zeros((grid_size, grid_size))
        
        for (x, y) in self.visited_positions:
            grid_x = min(int(x / width * grid_size), grid_size - 1)
            grid_y = min(int(y / height * grid_size), grid_size - 1)
            grid[grid_y, grid_x] += 1
        
        # 归一化
        if grid.sum() > 0:
            grid = grid / grid.sum()
        
        return grid.flatten()
    
    def compute_novelty(self, other_agents: List['Agent'], k: int = 3) -> float:
        """
        计算独特度分数
        
        基于与种群中其他 agent 的行为距离
        
        参数:
            other_agents: 其他智能体列表
            k: 计算最近邻数量
        
        返回:
            novelty score (越高越独特)
        """
        if not other_agents or self.behavior_grid is None:
            return 0.0
        
        # 收集其他 agent 的行为网格
        other_grids = []
        for agent in other_agents:
            if agent is not self and agent.behavior_grid is not None:
                other_grids.append(agent.behavior_grid)
        
        if not other_grids:
            return 0.0
        
        # 计算到所有其他 agent 的距离
        distances = []
        for og in other_grids:
            # 使用欧氏距离
            dist = np.sqrt(np.sum((self.behavior_grid - og) ** 2))
            distances.append(dist)
        
        distances = np.array(distances)
        
        # 取 k 个最近邻的平均距离
        if len(distances) >= k:
            k_nearest = np.sort(distances)[:k]
            novelty = np.mean(k_nearest)
        else:
            novelty = np.mean(distances)
        
        self.novelty_score = novelty
        return novelty
    
    def get_exploration_score(self) -> float:
        """
        获取探索得分 - 到访的独特格子数
        
        如果行为网格存在，返回非零格子数
        """
        if self.behavior_grid is None:
            return 0.0
        return np.sum(self.behavior_grid > 0)
    
    def verify_energy_conservation(self) -> dict:
        """
        v5.7 能量守恒定律验证
        
        物理规律: E_final = E_initial + E_gained - E_spent - E_wasted
        
        返回:
            dict: 包含误差信息的字典
        """
        expected_energy = (
            self.initial_energy  # 初始能量 (或继承的能量)
            + self.energy_gained
            - self.energy_spent
            - self.energy_wasted
        )
        
        # 允许小幅误差 (浮点精度)
        error = abs(self.internal_energy - expected_energy)
        
        return {
            'current_energy': self.internal_energy,
            'expected_energy': expected_energy,
            'error': error,
            'is_conserved': error < 0.01,
            'energy_initial': self.initial_energy,
            'energy_gained': self.energy_gained,
            'energy_spent': self.energy_spent,
            'energy_wasted': self.energy_wasted
        }
    
    def __repr__(self):
        return f"Agent({self.id}, pos=({self.x:.1f}, {self.y:.1f}), theta={np.degrees(self.theta):.1f}°)"


# ============================================================
# v8.0 无尽边疆 - 分块世界管理器
# ============================================================

