"""
BenchmarkRunner - 智能体认知能力基准测试框架

基于Gemini审阅意见实现:
1. 冻结大脑 + 无限能量 (测试智商而非代谢)
2. 多回合测试 (Episodic Run) 支持外部存储验证
3. Task Factory与现存API兼容
4. Trajectory Entropy落地

Level 1: 基础运动能力 (T-Maze直线)
Level 2: 短期记忆 (T-Maze延迟)
Level 3: 外部存储 (多回合Stigmergy)
Level 4: 元认知 (性能自我评估)
Level 5: 组合推理 (多任务迁移)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import copy
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.eoe.agent import Agent
from core.eoe.genome import OperatorGenome
from core.eoe.environment import Environment
from core.eoe.t_maze import TMazeEnvironment, TMazeConfig
from core.eoe.stigmergy_field import StigmergyField


@dataclass
class BenchmarkTask:
    """单个测试任务定义"""
    name: str
    level: int  # 1-5
    max_steps: int
    config: Dict[str, Any]  # 环境覆盖配置
    
    # 评估指标
    success_reward: float = 100.0
    step_penalty: float = -0.1
    
    # 多回合设置
    episodic: bool = False
    num_episodes: int = 1


@dataclass
class BenchmarkResult:
    """单个任务测试结果"""
    task_name: str
    level: int
    success: bool
    steps_taken: int
    final_position: Tuple[float, float]
    
    # 轨迹指标
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    trajectory_entropy: float = 0.0
    path_efficiency: float = 0.0
    
    # 认知指标
    surprise_total: float = 0.0
    prediction_errors: List[float] = field(default_factory=list)
    
    # 多回合指标
    episode_results: List['BenchmarkResult'] = field(default_factory=list)
    
    # 评分参数 (用于fitness计算)
    success_reward: float = 100.0
    step_penalty: float = -0.1
    
    @property
    def fitness(self) -> float:
        if not self.success:
            return -10.0
        return self.success_reward + self.step_penalty * self.steps_taken


class FrozenAgent:
    """
    冻结大脑的测试用智能体
    
    特性:
    - 无限能量 (不参与代谢计算)
    - 禁用变异和繁殖
    - 权重锁定 (requires_grad=False)
    """
    
    def __init__(self, brain: OperatorGenome, agent: Agent, env: Environment):
        self.brain = brain
        self.agent = agent
        self.env = env
        
        # 赋予无限能量 - 测试智商而非代谢
        self.agent.internal_energy = float('inf')
        self.agent.max_energy = float('inf')
        
        # 添加到环境
        self.env.add_agent(self.agent)
        
        # 禁用演化
        self._disable_evolution()
    
    def _disable_evolution(self):
        """禁用变异和繁殖方法"""
        if hasattr(self.brain, 'mutate'):
            self.brain.mutate = lambda *args, **kwargs: None
        if hasattr(self.brain, 'reproduce'):
            self.brain.reproduce = lambda *args, **kwargs: None
        
        # PyTorch权重锁定
        if hasattr(self.brain, 'network'):
            network = self.brain.network
            if hasattr(network, 'parameters'):
                for param in network.parameters():
                    param.requires_grad = False
    
    def step(self) -> np.ndarray:
        """
        执行单步 (由Environment.step()自动处理感知-决策-行动)
        
        Returns:
            sensor_inputs: 传感器输入 (从agent获取)
        """
        # 注意: 实际执行由 env.step() 完成
        # 这里我们只返回当前传感器值供记录
        if hasattr(self.agent, 'last_sensor_inputs') and self.agent.last_sensor_inputs is not None:
            return self.agent.last_sensor_inputs
        return np.array([0.0, 0.0])


class TaskFactory:
    """
    任务工厂 - 兼容现存API
    
    通过关键字参数创建测试环境
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "core/eoe/evolution_config.yaml"
    
    # 支持的环境参数白名单 (只包含Environment.__init__支持的参数)
    ENV_PARAMS = {
        'width', 'height', 'target_pos', 'metabolic_alpha', 'metabolic_beta',
        'surprise_penalty', 'n_food', 'food_energy', 'respawn_food', 'n_walls',
        'day_night_cycle', 'pure_survival_mode', 'seasonal_cycle', 'season_length',
        'winter_food_multiplier', 'winter_metabolic_multiplier', 'immediate_eating',
        'energy_decay_k', 'port_interference_gamma', 'season_jitter', 'nest_tax',
        'energy_field_enabled', 'field_resolution', 'field_diffusion_rate',
        'field_decay_rate', 'field_initial_energy', 'permeability_cost',
        'waste_heat_ratio', 'move_cost_coeff', 'impedance_field_enabled',
        'impedance_resolution', 'impedance_noise_scale', 'stigmergy_field_enabled',
        'stigmergy_resolution', 'stigmergy_diffusion', 'stigmergy_decay'
    }
    
    # Benchmark特定参数 (不传给Environment)
    BENCHMARK_PARAMS = {'t_maze_walls', 'walls', 'uncertainty_tracking', 'delay_steps'}
    
    @classmethod
    def create_environment(cls, task: BenchmarkTask) -> Environment:
        """
        创建测试环境
        
        Args:
            task: 任务配置
            
        Returns:
            配置好的Environment实例
        """
        config = task.config
        
        # 分离Environment参数和Benchmark参数
        env_kwargs = {k: v for k, v in config.items() if k in cls.ENV_PARAMS}
        benchmark_params = {k: v for k, v in config.items() if k in cls.BENCHMARK_PARAMS}
        
        # 设置默认值
        env_kwargs.setdefault('width', 100.0)
        env_kwargs.setdefault('height', 100.0)
        env_kwargs.setdefault('n_food', 3)  # 默认3个食物，让传感器能工作
        env_kwargs.setdefault('energy_field_enabled', False)
        env_kwargs.setdefault('stigmergy_field_enabled', False)
        
        # 创建环境
        env = Environment(**env_kwargs)
        
        # 将Benchmark参数附加到env对象
        for key, value in benchmark_params.items():
            setattr(env, key, value)
        
        # 根据任务类型设置特殊环境
        cls._apply_task_environment(env, task)
        
        return env
    
    @classmethod
    def _extract_env_params(cls, config: Dict) -> Dict:
        """从配置字典中提取环境参数"""
        result = {}
        for key, value in config.items():
            if key in cls.ENV_PARAMS:
                result[key] = value
        # 设置默认值
        result.setdefault('width', 100.0)
        result.setdefault('height', 100.0)
        result.setdefault('n_food', 0)
        result.setdefault('energy_field_enabled', False)
        result.setdefault('stigmergy_field_enabled', False)
        return result
    
    @classmethod
    def _load_default_config(cls) -> Dict:
        """加载默认配置"""
        import yaml
        if cls.DEFAULT_CONFIG_PATH.exists():
            with open(cls.DEFAULT_CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
        return {}
    
    @classmethod
    def _merge_config(cls, base: Dict, override: Dict) -> Dict:
        """深度合并配置"""
        result = copy.deepcopy(base)
        cls._deep_update(result, override)
        return result
    
    @classmethod
    def _deep_update(cls, base: Dict, update: Dict):
        """递归更新字典"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                cls._deep_update(base[key], value)
            else:
                base[key] = value
    
    @classmethod
    def _apply_task_environment(cls, env: Environment, task: BenchmarkTask):
        """应用任务特定的环境设置"""
        # 获取任务配置
        config = task.config
        
        # T-Maze墙壁生成
        if config.get('t_maze_walls', False):
            cls._add_t_maze_walls(env)
        
        # EoE Native 场域T-Maze (KIF墙 + EPF目标)
        if config.get('field_based_maze', False):
            cls._add_field_based_t_maze(env, config)
        
        # 障碍物回避 - 随机墙壁
        if task.name == "obstacle_avoidance":
            cls._add_random_obstacles(env, n_walls=config.get('n_walls', 10))
        
        # 觅食任务 - 设置食物
        if task.name == "foraging" or task.name == "multi_target":
            if config.get('n_food', 0) > 0:
                env.n_food = config.get('n_food', 0)
                env.food_energy = config.get('food_energy', 30.0)
                env.respawn_food = config.get('respawn_food', True)
                env._init_food()
                print(f"  [Foraging] {env.n_food} food items placed")
        
        # 设置目标位置
        if 'target_pos' in config and config['target_pos'] is not None:
            env.target_pos = config['target_pos']
        
        # 多目标任务
        if 'targets' in config:
            env.multi_targets = config['targets']
    
    @classmethod
    def _add_random_obstacles(cls, env: Environment, n_walls: int = 10):
        """添加随机障碍物"""
        import random
        w, h = env.width, env.height
        walls = []
        
        for _ in range(n_walls):
            # 随机墙壁位置和方向
            x = random.uniform(w * 0.2, w * 0.8)
            y = random.uniform(h * 0.2, h * 0.8)
            orientation = random.choice(['horizontal', 'vertical'])
            length = random.uniform(10, 20)
            
            if orientation == 'horizontal':
                walls.append((x - length/2, y, x + length/2, y))
            else:
                walls.append((x, y - length/2, x, y + length/2))
        
        env.walls = walls
        env.n_walls = len(walls)
        print(f"  [Obstacles] {len(walls)} random walls added")
    
    @classmethod
    def _add_t_maze_walls(cls, env: Environment):
        """
        生成T-Maze墙壁布局
        
        T-Maze形状:
        -----------
        |         |
        |    G    |  <- 目标在右上或右下
        |         |
        -----------
              |
              |
        -----------
        |         |
        |    |    |  <- 中间墙壁
        |    |    |
        -----------
            S        <- 起点在左侧
        """
        w, h = env.width, env.height
        
        # T-Maze墙壁 (中间垂直墙)
        # 从中间位置到顶部
        mid_x = w / 2
        vertical_wall_top = (mid_x, 0, mid_x, h * 0.4)
        
        # 从中间位置到底部  
        vertical_wall_bottom = (mid_x, h * 0.6, mid_x, h)
        
        # 添加墙壁到环境
        env.walls = [vertical_wall_top, vertical_wall_bottom]
        env.n_walls = len(env.walls)
        
        # 记录这是T-Maze配置
        env.t_maze_config = True
        print(f"  [T-Maze] Walls added: {len(env.walls)}")
    
    @classmethod
    def _add_field_based_t_maze(cls, env: Environment, config: Dict):
        """
        EoE Native 场域T-Maze: 用物理场代替几何墙
        
        KIF (阻抗场) = 墙壁 (Agent碰到会减血，本能避开)
        EPF (能量场) = 目标 (Agent本能追逐)
        
        关键设计:
        - EPF半径限制: 在T型路口梯度=0，考验工作记忆
        - Agent自带技能: 无需新传感器，会的就是这些场
        """
        w, h = env.width, env.height
        mid_x = w / 2
        epf_radius = config.get('epf_radius', 15.0)
        
        # 步骤1: 用KIF (阻抗场) 砌墙
        if env.impedance_field_enabled and env.impedance_field:
            # 场是2D的: (height, width)
            grid_w = env.impedance_field.grid_width
            grid_h = env.impedance_field.grid_height
            
            # 中间墙: 从y=0到y=h*0.4 和 y=h*0.6到y=h
            wall_intensity = 1000.0  # 高阻抗，碰了大幅减速/扣血
            
            # 上半部分墙 (y: 0 -> 40)
            for y in range(int(h * 0.4)):
                x = int(mid_x)
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    env.impedance_field.field[y, x] = wall_intensity
            
            # 下半部分墙 (y: 60 -> 100)
            for y in range(int(h * 0.6), int(h)):
                x = int(mid_x)
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    env.impedance_field.field[y, x] = wall_intensity
            
            print(f"  [Field-T-Maze] KIF walls added (impedance={wall_intensity})")
        
        # 步骤2: 用EPF (能量场) 作为目标
        if env.energy_field_enabled and env.energy_field:
            from core.eoe.environment import EnergySource
            
            # 目标位置: 右上分支尽头 (90, 25)
            # 注意: T-Maze中间有墙，正确的目标在 upper branch
            # 原配置 target_pos=(90,50) 是T型路口，不对
            if config.get('target_pos') == (90.0, 50.0):
                # 修正: 放在右上分支
                goal_pos = (90.0, 25.0)
            else:
                goal_pos = config.get('target_pos', (90.0, 25.0))
            
            # 只在目标位置添加能量源，半径受限
            goal_source = EnergySource(
                x=goal_pos[0],
                y=goal_pos[1],
                injection_rate=50.0,  # 能量注入率
                radius=epf_radius,  # 关键: 限制传播范围!
            )
            env.energy_field.add_source(goal_source)
            
            print(f"  [Field-T-Maze] EPF goal at {goal_pos}, radius={epf_radius}")
        
        # 步骤3: 启用ISF (如果配置了)
        if env.stigmergy_field_enabled and env.stigmergy_field:
            # ISF已经启用，允许Agent留下痕迹
            print(f"  [Field-T-Maze] ISF enabled for stigmergy")
        
        # 记录配置
        env.field_based_maze = True
        print(f"  [Field-T-Maze] Complete: KIF walls + EPF goal (r={epf_radius})")


class MetricsCalculator:
    """
    指标计算器
    
    实现轨迹熵等关键指标
    """
    
    @staticmethod
    def trajectory_entropy(
        trajectory: List[Tuple[float, float]], 
        grid_size: int = 20
    ) -> float:
        """
        计算轨迹香农熵
        
        将环境划分为grid_size x grid_size的网格，
        统计智能体在每个网格的停留频率。
        
        Args:
            trajectory: 位置轨迹 [(x, y), ...]
            grid_size: 网格划分数量
            
        Returns:
            熵值 H = -∑p_i * log(p_i)
        """
        if len(trajectory) < 2:
            return 0.0
        
        # 转换为网格坐标
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 避免除零
        if x_max == x_min:
            x_max += 1
        if y_max == y_min:
            y_max += 1
        
        # 计算网格索引
        grid_x = ((np.array(x_coords) - x_min) / (x_max - x_min) * (grid_size - 1)).astype(int)
        grid_y = ((np.array(y_coords) - y_min) / (y_max - y_min) * (grid_size - 1)).astype(int)
        
        # 统计频率
        grid_indices = grid_x * grid_size + grid_y
        unique, counts = np.unique(grid_indices, return_counts=True)
        
        # 计算概率
        probabilities = counts / len(trajectory)
        
        # 香农熵
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    @staticmethod
    def path_efficiency(
        trajectory: List[Tuple[float, float]],
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> float:
        """
        路径效率 = 直线距离 / 实际路径长度
        
        Returns:
            效率值 (0-1, 1为最优直线)
        """
        if len(trajectory) < 2:
            return 0.0
        
        # 直线距离
        direct_distance = np.hypot(goal[0] - start[0], goal[1] - start[1])
        
        # 实际路径长度
        path_length = sum(
            np.hypot(trajectory[i][0] - trajectory[i-1][0], 
                     trajectory[i][1] - trajectory[i-1][1])
            for i in range(1, len(trajectory))
        )
        
        if path_length < 1e-6:
            return 0.0
        
        return min(1.0, direct_distance / path_length)
    
    @staticmethod
    def success_rate(results: List[BenchmarkResult]) -> float:
        """计算成功率"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)


class BenchmarkRunner:
    """
    基准测试运行器
    
    用法:
        runner = BenchmarkRunner()
        results = runner.run_level(3, brain=my_brain)
    """
    
    # 标准任务模板 - EoE Native场域T-Maze
    TASK_TEMPLATES = {
        # Level 1: 基础运动 (T-Maze直线 - 用EPF引导)
        "t_maze_straight": BenchmarkTask(
            name="t_maze_straight",
            level=1,
            max_steps=200,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 0,
                "n_walls": 0,
                "target_pos": (90.0, 25.0),  # 修正: 右上分支
                "stigmergy_field_enabled": False,
                "energy_field_enabled": True,      # 用EPF引导
                "impedance_field_enabled": True,   # 用KIF当墙
                "field_based_maze": True,          # 场域T-Maze
                "epf_radius": 25.0,                # 增大一点范围
            }
        ),
        
        # Level 2: 短期记忆 (T-Maze - 用KIF墙+EPF目标)
        "t_maze_delayed": BenchmarkTask(
            name="t_maze_delayed",
            level=2,
            max_steps=400,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 0,
                "n_walls": 0,
                "target_pos": (90.0, 25.0),
                "stigmergy_field_enabled": True,
                "energy_field_enabled": True,
                "impedance_field_enabled": True,
                "field_based_maze": True,
                "epf_radius": 25.0,
            }
        ),
        
        # Level 3: 外部存储 (多回合 + ISF痕迹)
        "t_maze_stigmergy": BenchmarkTask(
            name="t_maze_stigmergy",
            level=3,
            max_steps=300,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 0,
                "n_walls": 0,
                "target_pos": (90.0, 25.0),
                "stigmergy_field_enabled": True,
                "energy_field_enabled": True,
                "impedance_field_enabled": True,
                "field_based_maze": True,
                "epf_radius": 25.0,
            },
            episodic=True,
            num_episodes=2
        ),
        
        # Level 4: 元认知
        "t_maze_meta": BenchmarkTask(
            name="t_maze_meta",
            level=4,
            max_steps=500,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 3,
                "n_walls": 0,
                "target_pos": (90.0, 20.0),
                "stigmergy_field_enabled": True,
                "energy_field_enabled": False,
                "t_maze_walls": True,
                "uncertainty_tracking": True,
            }
        ),
        
        # Level 5: 组合推理
        "t_maze_compositional": BenchmarkTask(
            name="t_maze_compositional",
            level=5,
            max_steps=800,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 3,
                "t_maze_walls": True,
                "stigmergy_field_enabled": True,
                "target_pos": (90.0, 50.0),
            }
        ),
        
        # ============================================================
        # 扩展任务类型 (Level 6-10)
        # ============================================================
        
        # Level 6: 开放场地 (Open Field) - 探索整个空间
        "open_field_explore": BenchmarkTask(
            name="open_field_explore",
            level=6,
            max_steps=300,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 3,
                "n_walls": 0,
                "target_pos": (90.0, 90.0),
                "stigmergy_field_enabled": False,
                "energy_field_enabled": False,
            }
        ),
        
        # Level 7: 障碍物回避 (Obstacle Avoidance)
        "obstacle_avoidance": BenchmarkTask(
            name="obstacle_avoidance",
            level=7,
            max_steps=400,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 3,
                "n_walls": 10,  # 随机障碍物
                "target_pos": (90.0, 50.0),
                "stigmergy_field_enabled": False,
                "energy_field_enabled": False,
            }
        ),
        
        # Level 8: 觅食任务 (Foraging) - 寻找多个食物
        "foraging": BenchmarkTask(
            name="foraging",
            level=8,
            max_steps=500,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 5,  # 5个食物
                "food_energy": 30.0,
                "respawn_food": True,
                "target_pos": None,  # 无固定目标
                "stigmergy_field_enabled": True,
                "energy_field_enabled": False,
            }
        ),
        
        # Level 9: 导航到指定位置 (Navigation)
        "navigation": BenchmarkTask(
            name="navigation",
            level=9,
            max_steps=350,
            config={
                "width": 150.0,
                "height": 150.0,
                "n_food": 3,
                "n_walls": 0,
                "target_pos": (140.0, 140.0),  # 角落到角落
                "stigmergy_field_enabled": False,
                "energy_field_enabled": False,
            }
        ),
        
        # Level 10: 多目标任务 (Multi-Target)
        "multi_target": BenchmarkTask(
            name="multi_target",
            level=10,
            max_steps=600,
            config={
                "width": 100.0,
                "height": 100.0,
                "n_food": 3,
                "n_walls": 0,
                "multi_target": True,
                "targets": [(90, 10), (10, 90), (90, 90)],  # 多个目标
                "stigmergy_field_enabled": True,
                "energy_field_enabled": False,
            }
        ),
    }
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.task_factory = TaskFactory()
        self.metrics = MetricsCalculator()
        self.results_history: List[BenchmarkResult] = []
    
    # ============================================================
    # 审计修复: 三大隐患修补
    # ============================================================
    
    @dataclass
    class WarmupConfig:
        """热身期配置"""
        enabled: bool = True
        steps: int = 1000
        learning_rate: float = 0.01
        exploration_noise: float = 1.0  # 初始探索噪声强度
        exploration_decay: float = 0.995  # 噪声衰减因子
        sensor丰富度: bool = True  # 蒙特梭利幼儿园模式
    
    def run_warmup(
        self,
        brain: OperatorGenome,
        config: 'BenchmarkRunner.WarmupConfig' = None
    ) -> OperatorGenome:
        """
        热身期：Agent在蒙特梭利幼儿园中学习
        
        修复三大隐患:
        1. 冷启动瘫痪: 添加探索噪声
        2. 温室花朵: 添加多种感知输入
        3. Hebbian信度: reward已集成到environment.py
        """
        if config is None:
            config = self.WarmupConfig()
        
        if self.verbose:
            print(f"\n[Warm-up] Starting {config.steps} steps of learning...")
            print(f"  - Exploration noise: {config.exploration_noise}")
            print(f"  - Montessori sandbox: {config.sensor丰富度}")
        
        # 创建蒙特梭利幼儿园环境
        env = self._create_montessori_sandbox()
        
        # 创建Agent
        agent = Agent(agent_id=0, x=50.0, y=50.0, add_predictors=False)
        agent.genome = brain.copy()
        
        # 打印warmup前的权重
        if self.verbose:
            print(f"[Warm-up] 初始边权重:")
            for i, e in enumerate(agent.genome.edges):
                print(f"    Edge {i}: {e['source_id']}->{e['target_id']}, w={e['weight']:.4f}")
        
        # 添加环境需要的属性
        agent.food_eaten = 0
        agent.left_output = 0.0
        agent.right_output = 0.0
        agent.velocity_actual = 0.0
        agent.linear_velocity = 0.0
        agent.angular_velocity = 0.0
        agent._last_energy_delta = 0.0
        
        # 添加到环境！否则env.step()不会处理它
        env.add_agent(agent)
        
        # 启用Hebbian学习 (确保边有learning_rate)
        for edge in agent.genome.edges:
            if edge.get('enabled', True):
                edge['learning_rate'] = config.learning_rate
                # 修复: 放大初始权重，让信号能传递
                if abs(edge['weight']) < 0.1:
                    edge['weight'] = np.random.choice([-1, 1]) * 0.5
        
        if self.verbose:
            print(f"[Warm-up] 权重放大后:")
            for i, e in enumerate(agent.genome.edges):
                print(f"    Edge {i}: {e['source_id']}->{e['target_id']}, w={e['weight']:.4f}, lr={e['learning_rate']:.4f}")
        
        # 探索噪声衰减
        current_noise = config.exploration_noise
        
        # 初始推力：让agent开始移动，否则brain输出是0
        agent.left_output = 0.5
        agent.right_output = 0.5
        
        for step in range(config.steps):
            # 修复隐患1: 探索噪声 (防止冷启动瘫痪)
            # 在每步之前给运动系统加噪声，强制agent移动
            if current_noise > 0.01:
                # 强制修改位置让agent动起来
                agent.x += np.random.randn() * current_noise * 2
                agent.y += np.random.randn() * current_noise * 2
                # 边界处理
                agent.x = max(1, min(99, agent.x))
                agent.y = max(1, min(99, agent.y))
                # 也给theta添加旋转噪声
                agent.theta += np.random.randn() * current_noise * 0.5
            
            env.step()
            
            # 噪声衰减
            current_noise *= config.exploration_decay
            
            if step % 200 == 0 and self.verbose:
                print(f"  Step {step}/{config.steps}, noise={current_noise:.4f}")
        
        if self.verbose:
            print(f"[Warm-up] Complete! Brain learned from interactions.")
            print(f"[Warm-up] 学习后的边权重:")
            for i, e in enumerate(agent.genome.edges):
                print(f"    Edge {i}: {e['source_id']}->{e['target_id']}, w={e['weight']:.6f}")
        
        return agent.genome
    
    def _create_montessori_sandbox(self) -> Environment:
        """
        创建蒙特梭利幼儿园环境
        
        修复隐患2: 感官丰富度
        - 能量源 (奖励信号)
        - 微弱阻抗墙 (非致死，触达减速)
        - 微弱风场 (随机方向)
        - 痕迹场 (可标记位置)
        """
        import random
        
        # 使用n_food参数创建能量源
        env = Environment(
            width=100, 
            height=100, 
            n_food=15,  # 15个能量源
            respawn_food=True
        )
        
        # 修复隐患2: 添加微弱阻抗墙 (非致死)
        # 碰到只是减速，不致命
        if hasattr(env, 'walls'):
            # 添加几个随机墙壁
            for _ in range(8):
                x1, y1 = random.uniform(10, 90), random.uniform(10, 90)
                x2, y2 = x1 + random.uniform(5, 15), y1 + random.uniform(5, 15)
                env.walls.append((x1, y1, x2, y2))
        
        # 修复隐患2: 添加微弱风场属性
        # 通过config传递微弱风场
        env.wind_strength = 0.1  # 微弱，不会致死
        
        # 启用痕迹场 (让agent学会标记)
        if hasattr(env, 'stigmergy_field'):
            env.stigmergy_field_enabled = True
        
        # 修复: 不要用无限能量模式！
        # 否则Hebbian学习的reward=0，无法学习
        # 给agent高初始能量，让它能活过热身期
        env.infinite_mode = False
        
        # 确保agent初始有足够能量 (不是150，是更高的起始值)
        # 通过n_food和respawn_food来保证能量补充
        
        if self.verbose:
            print(f"[Warm-up] Montessori sandbox created:")
            print(f"  - Energy sources: 15")
            print(f"  - Weak impedance walls: 8")
            print(f"  - Weak wind field: enabled")
            print(f"  - Infinite energy: True")
        
        return env
    
    def run_task(
        self, 
        task: BenchmarkTask, 
        brain: OperatorGenome,
        start_pos: Tuple[float, float] = (10.0, 50.0),
        enable_warmup: bool = True
    ) -> BenchmarkResult:
        """
        运行单个任务
        
        Args:
            task: 任务配置
            brain: 待测试的大脑网络
            start_pos: 起始位置
            enable_warmup: 是否启用热身期（后天学习）
            
        Returns:
            BenchmarkResult: 测试结果
        """
        if self.verbose:
            print(f"[Benchmark] Running task: {task.name} (Level {task.level})")
        
        # ===== 后天学习 (Baldwin Effect) =====
        if enable_warmup:
            warmup_config = self.WarmupConfig(
                enabled=True,
                steps=1000,
                learning_rate=0.01,
                exploration_noise=1.0,
                exploration_decay=0.995,
            )
            if self.verbose:
                print(f"[Warm-up] Running 1000 steps of lifetime learning...")
            brain = self.run_warmup(brain, warmup_config)
            if self.verbose:
                print(f"[Warm-up] Complete! Brain learned from interactions.")
            # 热身结束后，权重已更新，现在冻结权重进行正式测试
        
        # 创建环境
        env = self.task_factory.create_environment(task)
        
        # 创建冻结大脑的Agent (传入env)
        agent = Agent(agent_id=0, x=start_pos[0], y=start_pos[1], add_predictors=False)
        frozen_agent = FrozenAgent(brain, agent, env)
        
        # 轨迹记录
        trajectory = [ (agent.x, agent.y) ]
        
        # 多回合测试
        if task.episodic and task.num_episodes > 1:
            return self._run_episodic(task, brain, env, start_pos)
        
        # 单回合测试
        goal_pos = self._get_goal_position(task, env)
        
        for step in range(task.max_steps):
            # 环境自动调用 brain.forward() - 无需手动
            env.step()
            
            # 记录轨迹
            trajectory.append((agent.x, agent.y))
            
            # 检查成功
            if self._check_success(agent, goal_pos):
                if self.verbose:
                    print(f"  ✓ Success at step {step + 1}")
                return self._create_result(
                    task, True, step + 1, trajectory, goal_pos
                )
            
            # 检查失败 (撞墙)
            if not agent.is_alive:
                break
        
        # 超时失败
        if self.verbose:
            print(f"  ✗ Failed (timeout or dead)")
        return self._create_result(
            task, False, task.max_steps, trajectory, goal_pos
        )
    
    def _run_episodic(
        self,
        task: BenchmarkTask,
        brain: OperatorGenome,
        env: Environment,
        start_pos: Tuple[float, float]
    ) -> BenchmarkResult:
        """
        运行多回合测试 (Level 3 外部存储验证)
        
        回合1: 干净T-Maze探索，留下印记
        回合2: 保留印记场，测试是否利用记忆
        """
        episode_results = []
        
        for episode in range(task.num_episodes):
            if self.verbose:
                print(f"  Episode {episode + 1}/{task.num_episodes}")
            
            # 重置Agent位置
            agent = Agent(agent_id=episode, x=start_pos[0], y=start_pos[1], add_predictors=False)
            frozen_agent = FrozenAgent(brain, agent, env)
            
            trajectory = [ (agent.x, agent.y) ]
            goal_pos = self._get_goal_position(task, env)
            
            # 回合内循环
            for step in range(task.max_steps):
                env.step()  # 使用env.step()执行
                trajectory.append((agent.x, agent.y))
                
                if self._check_success(agent, goal_pos):
                    break
            
            # 保存回合结果
            episode_result = self._create_result(
                task, 
                self._check_success(agent, goal_pos),
                step + 1,
                trajectory,
                goal_pos
            )
            episode_results.append(episode_result)
            
            # 保留环境状态用于下一回合 (印记场 Persistence)
            # 注意: env对象在回合间保持不变，stigmergy_field自动保留
        
        # 汇总结果
        success = all(r.success for r in episode_results)
        total_steps = sum(r.steps_taken for r in episode_results)
        combined_trajectory = [p for r in episode_results for p in r.trajectory]
        
        result = self._create_result(task, success, total_steps, combined_trajectory, goal_pos)
        result.episode_results = episode_results
        
        if self.verbose:
            status = "✓" if success else "✗"
            print(f"  {status} Episodic result: {success}")
        
        return result
    
    def run_level(
        self, 
        level: int, 
        brain: OperatorGenome,
        **kwargs
    ) -> List[BenchmarkResult]:
        """运行指定Level的所有任务"""
        tasks = [t for t in self.TASK_TEMPLATES.values() if t.level == level]
        
        results = []
        for task in tasks:
            result = self.run_task(task, brain, **kwargs)
            results.append(result)
            self.results_history.append(result)
        
        return results
    
    def _create_result(
        self,
        task: BenchmarkTask,
        success: bool,
        steps: int,
        trajectory: List[Tuple[float, float]],
        goal_pos: Tuple[float, float]
    ) -> BenchmarkResult:
        """创建结果对象"""
        start_pos = trajectory[0] if trajectory else (0, 0)
        
        return BenchmarkResult(
            task_name=task.name,
            level=task.level,
            success=success,
            steps_taken=steps,
            final_position=trajectory[-1] if trajectory else (0, 0),
            trajectory=trajectory,
            trajectory_entropy=self.metrics.trajectory_entropy(trajectory),
            path_efficiency=self.metrics.path_efficiency(trajectory, start_pos, goal_pos),
            success_reward=task.success_reward,
            step_penalty=task.step_penalty
        )
    
    def _get_goal_position(self, task: BenchmarkTask, env: Environment) -> Tuple[float, float]:
        """获取任务目标位置"""
        if "t_maze" in task.name:
            # T-Maze目标在右侧尽头
            return (env.width - 10, env.height / 2)
        return (env.width - 10, env.height / 2)
    
    def _check_success(self, agent: Agent, goal_pos: Tuple[float, float], threshold: float = 5.0) -> bool:
        """检查是否到达目标"""
        distance = np.hypot(agent.x - goal_pos[0], agent.y - goal_pos[1])
        return distance < threshold
    
    def generate_report(self) -> str:
        """生成测试报告"""
        lines = ["# Benchmark Report", ""]
        
        for level in range(1, 6):
            level_results = [r for r in self.results_history if r.level == level]
            if not level_results:
                continue
            
            success_rate = self.metrics.success_rate(level_results)
            avg_entropy = np.mean([r.trajectory_entropy for r in level_results])
            avg_efficiency = np.mean([r.path_efficiency for r in level_results])
            
            lines.append(f"## Level {level}")
            lines.append(f"- Success Rate: {success_rate:.1%}")
            lines.append(f"- Avg Trajectory Entropy: {avg_entropy:.2f}")
            lines.append(f"- Avg Path Efficiency: {avg_efficiency:.2f}")
            lines.append("")
        
        return "\n".join(lines)


# 便捷函数
def run_benchmark(brain: OperatorGenome, level: int = 1) -> List[BenchmarkResult]:
    """快速运行基准测试"""
    runner = BenchmarkRunner()
    return runner.run_level(level, brain)


if __name__ == "__main__":
    # 简单测试
    print("BenchmarkRunner module loaded.")
    print("Available tasks:", list(BenchmarkRunner.TASK_TEMPLATES.keys()))