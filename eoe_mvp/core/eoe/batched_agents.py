"""
v14.0: 掩码张量池 (Masked Object Pool) 批量 Agent 引擎
======================================================
GPU 加速的 Agent 批处理系统 - 100% VRAM 常驻 + 异步连续生死

核心特性:
- 预分配最大容量 (MAX_AGENTS)，无动态显存分配
- 生命掩码 (alive_mask) 管理生死，O(1) 复杂度
- 异步连续运行，摆脱代际循环
- 能量驱动的自我繁衍与鲸落机制

架构:
- 预分配张量池: 所有状态按 MAX_AGENTS 预分配
- 计算屏蔽: 只对 alive_mask=True 的槽位进行计算
- 无锁生死: 掩码翻转即生死，无需张量拼接
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.eoe.genome import OperatorGenome
from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass

# 诊断模块
try:
    from core.eoe.diagnostics import EvolutionDiagnostics
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

class PoolConfig:
    """掩码池配置"""
    # 池大小
    MAX_AGENTS = 10000

    # 繁衍参数
    REPRODUCTION_THRESHOLD = 180.0
    CHILD_ENERGY_RATIO = 0.5
    MIN_REPRO_ENERGY = 30.0
    SPAWN_RADIUS = 0.5
    MUTATION_RATE = 0.1

    # 鲸落参数
    WHALE_RETURN_RATIO = 0.8
    BIOMASS_PER_NODE = 10.0

    # 代谢参数
    BASE_METABOLISM = 0.05  # 极低基础代谢
    ACTIVATION_COST = 0.001
    
    # v16.2 运动惩罚 (惩罚盲目移动，奖励精准伏击)
    MOVEMENT_PENALTY = 0.05  # 50倍于原ACTIVATION_COST，迫使"看准再动"

    # v16.3 认知溢价 (Cognitive Premium) - 隐身进食10倍奖励
    COGNITIVE_PREMIUM_MULTIPLIER = 10.0  # 隐身状态进食获得10倍能量
    ENABLE_INVISIBLE_SENSING = True      # 允许感知隐身能量(弱感知)
    
    # v16.4 基础代谢 (BMR) - 打破"伏地魔"策略
    # 即使静止也有不可避免的代谢成本，且复杂大脑消耗更多
    BASAL_COST = 0.10  # 静息基础代谢 (2倍于原BASE_METABOLISM)
    NEURAL_COST = 0.02  # 每节点额外代谢成本 (复杂脑更贵!)
    
    # v16.5 主动感知 (Active Sensing) - 运动-感知耦合
    # 不动就看不见！模拟生物界"主动感知"机制
    ACTIVE_SENSING_ENABLED = True        # 启用主动感知
    ACTIVE_SENSING_THRESHOLD = 0.3       # 达到此速度则100%感知
    ACTIVE_SENSING_MIN_EFFICIENCY = 0.05 # 静止时最低感知效率 (5%)
    INVISIBLE_SENSING_BOOST = 2.0        # 隐身能量需要更多运动来感知
    
    # v16.6 认知重构：只有隐身捕食才配获得10x奖励！
    # 伏地魔策略的核心漏洞：可见能量也给10x
    # 现在修复：可见=1x，隐身=10x (只有主动出击才能赢)
    COGNITIVE_PREMIUM_ONLY_INVISIBLE = True  # 只有隐身阶段有10x奖励
    
    # v16.7 极端不对称策略（彻底饿死伏地魔）
    VISIBLE_REWARD_MULTIPLIER = 0.1    # 可见能量只有0.1x (垃圾食品!)
    INVISIBLE_REWARD_MULTIPLIER = 20.0 # 隐身能量20x (超级大补丸!)
    VISIBLE_RATIO = 0.05               # 可见时间降至5%

    # 繁衍参数调整
    REPRODUCTION_THRESHOLD = 100.0  # 进一步降低

    # 红皇后捕食参数 (Red Queen Hypothesis) - 黑暗森林
    PREDATION_ENABLED = True        # 启用同类捕食
    PREDATION_RANGE = 4.0           # 捕食范围
    PREDATION_RATE = 0.8            # 吸血效率
    PREDATION_COST = 0.05           # 捕食代谢成本
    PREDATION_MUTATION = 0.15       # 产生捕食者突变概率
    ATTACK_RADIUS = 3.0             # 攻击半径
    ATTACK_THRESHOLD = 0.5          # 攻击阈值
    STRIKE_COST = 2.0               # 爆发攻击成本

    # 代谢衰老参数 (熵增定律)
    AGE_ENABLED = True              # 启用年龄惩罚
    AGE_ALPHA = 0.00001             # 衰老系数: cost * (1 + alpha * age^2)

    # 演化棘轮 (Supernode Ratchet)
    SUPERNODE_ENABLED = True        # 启用超级节点
    SUPERNODE_METABOLIC_BONUS = 0.5 # 超级节点代谢折扣 (1.5节点算0.5!)
    SUPERNODE_DETECTION_FREQUENCY = 500  # 检测频率
    SUPERNODE_MIN_OCCURRENCE = 5    # 最少出现次数才触发折叠

    # ================================================================
    # v14.0 鲍德温效应: 能量调制赫布学习
    # ================================================================
    HEBBIAN_ENABLED = True           # 启用Hebbian学习
    HEBBIAN_ELIGIBILITY_TRACE = 5    # 资格迹长度 (追踪过去5步)
    HEBBIAN_BASE_LR = 0.01           # 基础学习率
    HEBBIAN_REWARD_MODULATION = True # 能量调制开关
    HEBBIAN_DEADZONE = 1.0           # 死区: 能量变化<1.0不计
    HEBBIAN_MAX_WEIGHT_DELTA = 0.1   # 每步最大权重变化
    HEBBIAN_TRACE_DECAY = 0.9        # 资格迹衰减

    # ================================================================
    # v14.1 寒武纪大爆发 (初始种群多样性)
    # ================================================================
    CAMBRIAN_INIT = True             # 启用寒武纪初始化
    CAMBRIAN_MIN_NODES = 3           # 最小节点数
    CAMBRIAN_MAX_NODES = 7           # 最大节点数
    CAMBRIAN_DELAY_PROB = 0.3        # 混入DELAY节点概率
    CAMBRIAN_MULTIPLY_PROB = 0.3     # 混入MULTIPLY节点概率

    # ================================================================
    # v14.1 静默拓扑突变 (新节点权重初始化)
    # ================================================================
    SILENT_MUTATION = True           # 启用静默突变
    SILENT_WEIGHT = 0.001            # 新连接初始权重 (极小)

    # ================================================================
    # v14.1 代谢宽限期 (新拓扑折扣)
    # ================================================================
    METABOLIC_GRACE = True           # 启用代谢宽限期
    METABOLIC_GRACE_STEPS = 100      # 宽限期步数
    METABOLIC_GRACE_DISCOUNT = 0.5   # 折扣: 前100步只付50%代谢

    # ================================================================
    # v15 认知溢价: 非线性代谢 (Economies of Scale)
    # ================================================================
    NONLINEAR_METABOLISM = True      # 启用非线性代谢 (log(N+1))
    LOG_BASE = 2.0                   # 对数底 (2.0 = log2)
    FREE_NODES = 5                   # 免费基础脑容量 (前5节点免代谢)
    SPARSE_ACTIVATION = True         # 稀疏激活 (休眠节点不耗能)
    SPARSE_THRESHOLD = 0.01          # 发放率低于此值视为休眠

    # ================================================================
    # v14.2 动态环境难度 (季节/干旱)
    # ================================================================
    SEASONS_ENABLED = True           # 启用季节变化
    SEASON_LENGTH = 2000             # 季节周期 (步)
    WINTER_MULTIPLIER = 0.1          # 冬季能量倍率 (10%)
    SUMMER_MULTIPLIER = 1.5          # 夏季能量倍率 (150%)
    DROUGHT_ENABLED = True           # 启用干旱期
    DROUGHT_INTENSITY = 0.05         # 干旱期能量倍率 (5%)

    # ================================================================
    # v15 T型迷宫 (POMDP - 强制记忆测试)
    # ================================================================
    T_MAZE_ENABLED = False           # 启用T型迷宫任务
    T_MAZE_SIGNAL_DURATION = 5       # 信号持续步数
    T_MAZE_BLIND_ZONE = 20           # 盲区步数
    T_MAZE_DECISION_DELAY = 25       # 信号到决策的延迟
    T_MAZE_CORRECT_REWARD = 100.0    # 正确奖励
    T_MAZE_WRONG_REWARD = 0.0        # 错误惩罚
    T_MAZE_STEP_PENALTY = 0.1        # 每步惩罚

    # 资源周期性消失
    RESOURCE_CYCLE_ENABLED = False   # 启用资源周期
    RESOURCE_CYCLE_LENGTH = 500      # 周期长度
    RESOURCE_FADE_STEPS = 50         # 消失过渡步数

    # ================================================================
    # v15 Red Queen Dynamics (智能猎物)
    # ================================================================
    RED_QUEEN_ENABLED = False        # 启用智能猎物

    # ================================================================
    # v15 能量循环 (代谢能量回归环境)
    # ================================================================
    ENERGY_RECIRCULATION_ENABLED = True   # 启用能量循环
    ENERGY_RECIRCULATION_RATIO = 0.6      # 60%代谢能量回归环境

    # ================================================================
    # v16.0 构成性环境 (Matter Grid)
    # ================================================================
    MATTER_GRID_ENABLED = False        # 启用物质场
    MATTER_RESOLUTION = 1.0            # 物质场分辨率

    # 建造/分解参数
    CONSTRUCT_ENERGY_COST = 15.0       # 建造消耗能量
    CONSTRUCT_MIN_ENERGY = 10.0        # 建造所需最小能量
    DECONSTRUCT_ENERGY_GAIN = 15.0     # 分解回收能量 (全额返还)
    CONSTRUCT_COOLDOWN = 5             # 建造/分解冷却步数
    CONSTRUCT_DISTANCE_FACTOR = 1.0    # 建造距离 = radius * factor

    # 脑输出通道配置
    N_BRAIN_OUTPUTS = 5                # 默认5通道 (兼容旧版)
    N_BRAIN_OUTPUTS_V16 = 7            # v16.0 7通道 (含建造/分解)
    PREY_DETECTION_RANGE = 25.0      # 猎物感知范围
    PREY_ESCAPE_TRIGGER = 15.0       # 触发逃跑距离
    PREY_ESCAPE_SPEED = 2.0          # 逃跑速度
    PREY_ZIGZAG_PERIOD = 8           # Z字形周期
    PREY_ZIGZAG_AMPLITUDE = 0.5      # Z字形幅度
    PREY_FATIGUE_DURATION = 30       # 逃跑后疲劳步数

    # ================================================================
    # v14.1 诊断系统
    # ================================================================
    DIAGNOSTICS_ENABLED = True       # 启用诊断监控

    # ================================================================
    # v15.3 可微演化 (Differentiable Evolution)
    # ================================================================
    DIFFERENTIABLE_BRAIN = False      # 启用可微大脑
    DIFFERENTIABLE_USE_PYG = True     # 使用torch_geometric
    PREDICTION_LOSS_WEIGHT = 0.1      # 预测损失权重
    ENERGY_LR_MODULATOR = True        # 能量调节学习率
    DIFFERENTIABLE_LR = 0.001         # 生命周期学习率
    DIFFERENTIABLE_UPDATE_INTERVAL = 10  # 每N步更新一次权重
    DIFFERENTIABLE_MIN_STEPS = 5      # 最少累积经验步数
    DIFFERENTIABLE_MAX_BUFFER = 50    # 经验buffer大小

    # 深度鲍德温配置
    BALDWIN_ASSIMILATION_KAPPA = 0.5  # 同化率 (0-1)
    BALDWIN_EXPLORATION_SIGMA = 0.01  # 变异噪声

    # ================================================================
    # v15 预加载脑结构机制 (Brain Bootstrap)
    # ================================================================
    PRETRAINED_INIT = False          # 启用预加载脑结构
    PRETRAINED_STRUCTURES_FILE = ""  # 结构文件路径
    PRETRAINED_TOP_N = 20            # 使用Top N个最复杂结构
    PRETRAINED_DUPLICATE_FACTOR = 1  # 每种结构复制次数


# ============================================================================
# Agent 状态张量
# ============================================================================

@dataclass
class AgentState:
    """Agent 状态容器 (GPU 张量, 预分配 MAX_AGENTS 大小)"""
    positions: torch.Tensor      # [MAX_AGENTS, 2] (x, y)
    velocities: torch.Tensor     # [MAX_AGENTS, 2] (vx, vy) - 旧版兼容
    energies: torch.Tensor       # [MAX_AGENTS] 内部能量 (活动能量)
    thetas: torch.Tensor         # [MAX_AGENTS] 朝向角
    permeabilities: torch.Tensor # [MAX_AGENTS] 渗透率 (0-1)
    defenses: torch.Tensor       # [MAX_AGENTS] 防御力 (0-1)
    signals: torch.Tensor        # [MAX_AGENTS] 信号强度 (0-1)

    # 具身运动学状态 (v13.1+)
    linear_velocity: torch.Tensor   # [MAX_AGENTS] 线速度
    angular_velocity: torch.Tensor  # [MAX_AGENTS] 角速度

    # 结构能量 (v14.0 鲸落用)
    structural_energy: torch.Tensor # [MAX_AGENTS] 躯体生物量
    node_counts: torch.Tensor       # [MAX_AGENTS] 脑节点数

    # 代谢衰老 (熵增)
    ages: torch.Tensor              # [MAX_AGENTS] 存活步数

    # 超级节点 (演化棘轮)
    supernodes: torch.Tensor        # [MAX_AGENTS] 超级节点数量

    # ================================================================
    # v14.0 鲍德温效应: 能量调制赫布学习
    # ================================================================
    prev_energies: torch.Tensor     # [MAX_AGENTS] 上一步能量 (计算ΔE)
    hebbian_plastic_mask: torch.Tensor  # [MAX_AGENTS, max_edges] 可塑性边掩码

    # ================================================================
    # v14.1 代谢宽限期 (Metabolic Grace Period)
    # ================================================================
    mutation_timestamp: torch.Tensor  # [MAX_AGENTS] 上次拓扑突变的时间步

    # ================================================================
    # v15 T型迷宫 (POMDP - 强制记忆测试)
    # ================================================================
    t_maze_signal: torch.Tensor       # [MAX_AGENTS] 当前信号 (0=无, 1=左, 2=右)
    t_maze_signal_timer: torch.Tensor # [MAX_AGENTS] 信号剩余步数
    t_maze_episode_step: torch.Tensor # [MAX_AGENTS] 当前回合步数
    t_maze_correct_dir: torch.Tensor  # [MAX_AGENTS] 当前回合正确方向 (0=左, 1=右)
    t_maze_decision_made: torch.Tensor # [MAX_AGENTS] 是否已决策
    t_maze_episodes: torch.Tensor     # [MAX_AGENTS] 完成的回合数
    t_maze_correct: torch.Tensor      # [MAX_AGENTS] 正确决策次数

    # 资源周期状态
    resource_visible: torch.Tensor    # [MAX_AGENTS] 资源是否可见


class ActiveBatch:
    """活跃 Agent 的批量切片 (不存储数据，只存索引和视图)"""

    def __init__(self, indices: torch.Tensor, state: AgentState):
        self.indices = indices  # [M] 活跃 Agent 的索引
        self.state = state      # 完整状态张量

    @property
    def n(self) -> int:
        return len(self.indices)

    @property
    def positions(self) -> torch.Tensor:
        return self.state.positions[self.indices]

    @property
    def energies(self) -> torch.Tensor:
        return self.state.energies[self.indices]

    @property
    def linear_velocity(self) -> torch.Tensor:
        return self.state.linear_velocity[self.indices]

    @property
    def angular_velocity(self) -> torch.Tensor:
        return self.state.angular_velocity[self.indices]

    @property
    def thetas(self) -> torch.Tensor:
        return self.state.thetas[self.indices]


class BatchedAgents:
    """
    掩码张量池批量 Agent 管理器
    ==========================
    预分配最大容量，无动态显存分配，支持异步连续生死
    """

    def __init__(
        self,
        initial_population: int = 300,
        max_agents: int = 10000,
        env_width: float = 100.0,
        env_height: float = 100.0,
        device: str = 'cuda:0',
        init_energy: float = 150.0,
        config: PoolConfig = None,
        env: 'EnvironmentGPU' = None
    ):
        self.max_agents = max_agents
        self.config = config or PoolConfig()
        self.env_width = env_width
        self.env_height = env_height
        self.device = device
        self.env = env  # 用于采样阻抗场
        
        # 调试: 建造尝试计数器
        self._construct_debug_step = 0

        print(f"[BatchedAgents] 初始化掩码池 on {device}")
        print(f"  池大小: {max_agents}, 初始人口: {initial_population}")

        # 预分配状态张量
        self._init_state_tensor(init_energy, initial_population)

        # 生命掩码 (核心！)
        self.alive_mask = torch.zeros(max_agents, dtype=torch.bool, device=device)
        self.alive_mask[:initial_population] = True

        # 活跃索引缓存
        self._active_indices: Optional[torch.Tensor] = None
        self._indices_dirty = True

        # 大脑矩阵 (延迟初始化)
        self.brain_matrix = None
        self.brain_masks = None
        self.node_counts_tensor = None

        # 基因组字典 {idx: OperatorGenome}
        self.genomes: Dict[int, 'OperatorGenome'] = {}

        # BMR 预编译
        self.agent_bmr = torch.zeros(max_agents, device=device)

        # ================================================================
        # v14.0 鲍德温效应: 资格迹缓冲区 (Eligibility Trace)
        # 循环缓冲区: self.eligibility_trace[:, trace_ptr, :]
        # ================================================================
        if self.config.HEBBIAN_ENABLED:
            max_edges = 50  # 假设最多边数
            trace_len = self.config.HEBBIAN_ELIGIBILITY_TRACE
            self.eligibility_trace = torch.zeros(
                max_agents, trace_len, max_edges,
                device=device, dtype=torch.float32
            )
            self.trace_ptr = 0  # 循环指针
            self.hebbian_step_count = 0
            print(f"  ✅ Hebbian eligibility trace: ({max_agents}, {trace_len}, {max_edges})")

        # ================================================================
        # v14.0 演化棘轮: 子图挖掘器 + SuperNode注册表
        # ================================================================
        if self.config.SUPERNODE_ENABLED:
            try:
                from core.eoe.subgraph_miner import SubgraphMiner
                from core.eoe.supernode_registry import SuperNodeRegistry

                self.subgraph_miner = SubgraphMiner(
                    min_support=0.3,
                    min_size=3,
                    max_size=5,
                    device=device
                )
                self.supernode_registry = SuperNodeRegistry(
                    cost_discount=0.7,
                    max_supernodes=10,
                    device=device
                )
                self.subgraph_mining_enabled = True
                self.total_steps = 0
                print(f"  ✅ SuperNode挖掘器已启用 (每{self.config.SUPERNODE_DETECTION_FREQUENCY}步)")
            except ImportError as e:
                print(f"  ⚠️ SuperNode挖掘器导入失败: {e}")
                self.subgraph_mining_enabled = False
        else:
            self.total_steps = 0  # 默认初始化

        # 世代计数器 (用于RedQueen)
        self.generation = 0

        # 性能统计
        self.step_times = []

        # ================================================================
        # v14.1 演化机制: 从manifest加载并注册
        # ================================================================
        try:
            from core.eoe.manifest import PhysicsManifest
            self.manifest = PhysicsManifest.from_yaml("full")
            self.evo_mechanisms = self.manifest.registry.get_evo_mechanisms()
            self.event_mechanisms = self.manifest.registry.get_event_mechanisms()

            evo_names = [m.name for m in self.evo_mechanisms]
            event_names = [m.name for m in self.event_mechanisms]
            print(f"  ✅ 演化机制已加载: 每Step={evo_names}, 事件={event_names}")
        except Exception as e:
            print(f"  ⚠️ 演化机制加载失败: {e}")
            self.manifest = None
            self.evo_mechanisms = []
            self.event_mechanisms = []

        # ================================================================
        # v14.1 诊断系统
        # ================================================================
        if DIAGNOSTICS_AVAILABLE and getattr(self.config, 'DIAGNOSTICS_ENABLED', True):
            self.diagnostics = EvolutionDiagnostics(
                max_agents=max_agents,
                device=device,
                log_interval=500,
                history_size=2000
            )
            print(f"  ✅ 诊断系统已启用")
        else:
            self.diagnostics = None

        print(f"  ✅ 掩码池初始化完成")

    def _init_state_tensor(self, init_energy: float, init_population: int):
        """预分配所有状态张量 (MAX_AGENTS 大小)"""
        max_agents = self.max_agents

        # 随机位置
        positions = torch.rand(max_agents, 2, device=self.device) * \
                    torch.tensor([self.env_width, self.env_height], device=self.device)

        self.state = AgentState(
            positions = positions,
            velocities = torch.zeros(max_agents, 2, device=self.device),
            energies = torch.zeros(max_agents, device=self.device),
            thetas = torch.rand(max_agents, device=self.device) * 2 * np.pi,
            permeabilities = torch.ones(max_agents, device=self.device) * 0.5,
            defenses = torch.ones(max_agents, device=self.device) * 0.5,
            signals = torch.zeros(max_agents, device=self.device),

            # 具身运动学
            linear_velocity = torch.zeros(max_agents, device=self.device),
            angular_velocity = torch.zeros(max_agents, device=self.device),

            # 结构能量 (v14.0)
            structural_energy = torch.zeros(max_agents, device=self.device),
            node_counts = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            ages = torch.zeros(max_agents, device=self.device),
            supernodes = torch.zeros(max_agents, device=self.device, dtype=torch.long),

            # v14.0 鲍德温效应: 能量调制赫布学习
            prev_energies = torch.zeros(max_agents, device=self.device),
            hebbian_plastic_mask = torch.zeros(max_agents, 50, device=self.device, dtype=torch.bool),  # 假设最多50条边

            # v14.1 代谢宽限期
            mutation_timestamp = torch.full((max_agents,), -1000, device=self.device, dtype=torch.long),  # -1000表示无突变

            # v15 T型迷宫 (POMDP)
            t_maze_signal = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            t_maze_signal_timer = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            t_maze_episode_step = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            t_maze_correct_dir = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            t_maze_decision_made = torch.zeros(max_agents, device=self.device, dtype=torch.bool),
            t_maze_episodes = torch.zeros(max_agents, device=self.device, dtype=torch.long),
            t_maze_correct = torch.zeros(max_agents, device=self.device, dtype=torch.long),

            # 资源周期状态
            resource_visible = torch.ones(max_agents, device=self.device, dtype=torch.bool),
        )

        # 初始人口能量 - 使用initial_population而非MAX_AGENTS
        init_n = init_population
        self.state.energies[:init_n] = init_energy
        self.state.prev_energies[:init_n] = init_energy  # 初始化prev_energy
        self.state.structural_energy[:init_n] = init_energy * 0.5

        # ============================================================
        # v15 T型迷宫初始化 (初始人口也需要)
        # ============================================================
        if self.config.T_MAZE_ENABLED and init_n > 0:
            correct_dirs = torch.randint(0, 2, (init_n,), device=self.device)
            self.state.t_maze_correct_dir[:init_n] = correct_dirs
            self.state.t_maze_signal[:init_n] = correct_dirs + 1
            self.state.t_maze_signal_timer[:init_n] = self.config.T_MAZE_SIGNAL_DURATION
            self.state.t_maze_episode_step[:init_n] = 0
            self.state.t_maze_decision_made[:init_n] = False

        print(f"  ✅ 预分配张量: {self.state.positions.shape}")

    # ============================================================================
    # 核心 API
    # ============================================================================

    def get_active_batch(self) -> ActiveBatch:
        """
        获取当前活跃 Agent 的批量切片

        Returns:
            ActiveBatch: 活跃 Agent 的索引和状态视图
        """
        if self._indices_dirty or self._active_indices is None:
            self._active_indices = self.alive_mask.nonzero(as_tuple=True)[0]
            self._indices_dirty = False

        return ActiveBatch(self._active_indices, self.state)

    def step(
        self,
        env: 'EnvironmentGPU' = None,
        dt: float = 0.1,
        brain_fn: Optional[Callable] = None
    ) -> Dict[str, any]:
        """
        连续步进: 物理 + 代谢 + 生死

        Args:
            env: GPU 环境 (可选)
            dt: 时间步长
            brain_fn: 大脑前向函数 (可选)

        Returns:
            dict: 统计信息
        """
        batch = self.get_active_batch()
        if batch.n == 0:
            return {'n_alive': 0, 'births': 0, 'deaths': 0}

        # 1. 大脑推理
        # v16.0: 根据配置确定输出通道数
        n_outputs = self.config.N_BRAIN_OUTPUTS if not self.config.MATTER_GRID_ENABLED else self.config.N_BRAIN_OUTPUTS_V16

        if brain_fn is not None:
            brain_outputs = brain_fn(batch)
        elif self.brain_matrix is not None and env is not None:
            # 默认使用内置大脑前向传播
            sensors = self.get_sensors(env)
            brain_outputs = self.forward_brains(sensors)
        else:
            brain_outputs = torch.zeros(batch.n, n_outputs, device=self.device)

        # 2. 物理更新
        self._apply_physics(batch, brain_outputs, dt)

        # 2.5 v16.0: 建造/分解动作
        if self.config.MATTER_GRID_ENABLED and env is not None:
            self._apply_construction(batch, brain_outputs, env)

        # 3. 代谢扣除
        self._apply_metabolism(batch, dt)

        # 3.3 T型迷宫状态更新 (POMDP)
        if self.config.T_MAZE_ENABLED:
            self._update_t_maze(batch)

        # 3.5 鲍德温效应: 能量调制赫布学习
        if self.config.HEBBIAN_ENABLED:
            self._apply_reward_hebbian(batch)

        # 4. 环境交互
        if env is not None:
            self._apply_environment_interaction(batch, env)

        # 4.5 演化机制 (每Step调用)
        if self.evo_mechanisms:
            self._apply_evo_mechanisms(batch, env)

        # 5. 黑暗森林同类捕食 + RedQueen事件触发
        if self.config.PREDATION_ENABLED:
            predation_occurred = self._apply_predation(batch, brain_outputs)

            # RedQueen事件触发: 捕食发生时
            if predation_occurred and self.event_mechanisms:
                self._trigger_event_mechanisms(batch, env)

        # 6. 鲸落 (死亡)
        deaths = self._process_deaths(batch, env)

        # 6. 分裂 (繁衍)
        births = self._process_reproduction(batch)

        # 7. 边界
        self._apply_boundaries(batch)

        # 8. 演化棘轮: 子图挖掘 (后台异步运行)
        if self.config.SUPERNODE_ENABLED and self.subgraph_mining_enabled:
            self.total_steps += 1
            if self.total_steps % self.config.SUPERNODE_DETECTION_FREQUENCY == 0:
                self._run_subgraph_mining()

        # 9. 世代计数 (每100步 = 1代, 用于RedQueen)
        if self.total_steps % 100 == 0:
            self.generation += 1

        return {
            'n_alive': self.get_active_batch().n,
            'births': births,
            'deaths': deaths
        }

    def _apply_physics(self, batch: ActiveBatch, outputs: torch.Tensor, dt: float):
        """应用物理 (仅对活跃 Agent)"""
        idx = batch.indices

        # 解码脑输出
        permeabilities = torch.sigmoid(outputs[:, 0])
        thrust_x = torch.tanh(outputs[:, 1]) * 5.0
        thrust_y = torch.tanh(outputs[:, 2]) * 5.0
        signals = torch.relu(outputs[:, 3])
        defenses = torch.sigmoid(outputs[:, 4])

        # ============================================================
        # 空间阻抗: 根据环境阻抗场调整阻尼
        # 高阻抗区域 (墙壁) = 更大的阻尼 = 更难移动
        # ============================================================
        base_friction = 0.9

        # 尝试从环境获取阻抗
        impedance_friction = 0.0
        if hasattr(self, 'env') and self.env is not None:
            try:
                if hasattr(self.env, 'impedance_field') and self.env.impedance_field is not None:
                    # 采样当前位置的阻抗
                    positions = batch.positions
                    grid_x = (positions[:, 0] / self.env.width * self.env.impedance_field.grid_width).long().clamp(0, self.env.impedance_field.grid_width - 1)
                    grid_y = (positions[:, 1] / self.env.height * self.env.impedance_field.grid_height).long().clamp(0, self.env.impedance_field.grid_height - 1)

                    # 获取阻抗值 [N]
                    impedance = self.env.impedance_field.field[0, 0, grid_y, grid_x]

                    # 阻抗越高，阻尼越大 (更难移动)
                    # impedance=0 -> friction=0.9, impedance=10 -> friction=0.5
                    impedance_friction = 0.9 - (impedance.clamp(max=10) / 10) * 0.4
            except Exception as e:
                # 静默失败但记录 (阻抗场可能不存在)
                pass

        friction = base_friction - impedance_friction

        # 线速度更新
        self.state.linear_velocity[idx] *= friction
        self.state.linear_velocity[idx] += torch.sqrt(thrust_x**2 + thrust_y**2) * dt

        # 角速度更新
        self.state.angular_velocity[idx] *= friction
        self.state.angular_velocity[idx] += (thrust_x * 0.1) * dt

        # 记录旧位置（用于碰撞恢复）
        old_positions = self.state.positions[idx].clone()

        # 位置更新 (非全向)
        self.state.positions[idx, 0] += self.state.linear_velocity[idx] * \
            torch.cos(self.state.thetas[idx]) * dt
        self.state.positions[idx, 1] += self.state.linear_velocity[idx] * \
            torch.sin(self.state.thetas[idx]) * dt

        # 朝向更新
        self.state.thetas[idx] += self.state.angular_velocity[idx] * dt

        # ============================================================
        # v16.0: MatterGrid 碰撞检测
        # 如果新位置是固体，恢复到旧位置
        # ============================================================
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'matter_grid') and self.env.matter_grid is not None:
                self._apply_matter_collision(batch, old_positions)

        # 写回其他状态
        self.state.permeabilities[idx] = permeabilities
        self.state.defenses[idx] = defenses
        self.state.signals[idx] = signals

    def _apply_matter_collision(self, batch: ActiveBatch, old_positions: torch.Tensor):
        """应用 MatterGrid 碰撞检测"""
        idx = batch.indices
        device = self.state.positions.device

        # 获取新位置
        new_pos = self.state.positions[idx]

        # 转换为网格坐标
        if not hasattr(self.env, 'matter_grid') or self.env.matter_grid is None:
            return

        # 使用环境分辨率
        resolution = getattr(self.env, 'matter_resolution', 1.0)
        grid_w = self.env.matter_grid.shape[3]  # [1, 1, H, W]
        grid_h = self.env.matter_grid.shape[2]

        # 计算新位置的网格坐标
        new_gx = (new_pos[:, 0] / resolution).long() % grid_w
        new_gy = (new_pos[:, 1] / resolution).long() % grid_h

        # 采样 matter_grid
        # matter_grid shape: [1, 1, H, W]
        try:
            collision = self.env.matter_grid[0, 0, new_gy, new_gx].bool()

            # 如果发生碰撞，恢复到旧位置
            if collision.any():
                self.state.positions[idx][collision] = old_positions[collision]
                # 碰撞时速度清零
                self.state.linear_velocity[idx][collision] = 0.0
        except Exception:
            pass  # 静默失败

    def _apply_construction(self, batch: ActiveBatch, brain_outputs: torch.Tensor, env: 'EnvironmentGPU'):
        """
        v16.0: 应用建造/分解动作

        脑输出通道:
        - 5: CONSTRUCT (建造) - 激活则建造
        - 6: DECONSTRUCT (分解) - 激活则分解

        Review #2: 建造距离 = agent.radius + resolution (防止自我活埋)
        Review #3: 能量守恒 - 墙壁存储能量，分解全额返还
        Review #4: GPU并发竞争处理 - 使用去重逻辑
        """
        if not self.config.MATTER_GRID_ENABLED:
            return

        idx = batch.indices
        n = batch.n

        # 检查输出维度是否足够
        if brain_outputs.shape[1] < 7:
            return

        # 解码建造/分解输出
        construct_activation = torch.sigmoid(brain_outputs[:, 5])  # [N]
        deconstruct_activation = torch.sigmoid(brain_outputs[:, 6])  # [N]

        # 建造阈值
        CONSTRUCT_THRESHOLD = 0.5
        DECONSTRUCT_THRESHOLD = 0.5

        # 获取当前能量
        energies = self.state.energies[idx]

        # ========== 建造动作 ==========
        can_construct = (
            (construct_activation > CONSTRUCT_THRESHOLD) &
            (energies >= self.config.CONSTRUCT_MIN_ENERGY)
        )

        # 调试: 跟踪建造尝试
        if can_construct.any() and hasattr(self, '_construct_debug_step'):
            self._construct_debug_step += 1
            if self._construct_debug_step % 100 == 0:
                print(f"  [建造] 尝试: {can_construct.sum().item()} agents")

        if can_construct.any():
            # 计算目标位置 (前方)
            # Review #2: 距离必须大于 agent.radius
            positions = self.state.positions[idx]
            thetas = self.state.thetas[idx]

            # 假设 agent.radius = 1.0 (或从配置获取)
            agent_radius = 1.0
            forward_dist = agent_radius + self.config.CONSTRUCT_DISTANCE_FACTOR

            target_x = positions[can_construct, 0] + torch.cos(thetas[can_construct]) * forward_dist
            target_y = positions[can_construct, 1] + torch.sin(thetas[can_construct]) * forward_dist
            
            # 调试: 检查目标位置
            if self._construct_debug_step % 100 == 0:
                print(f"    目标: x={target_x[0].item():.1f}, y={target_y[0].item():.1f}")

            # 执行建造 (CPU 环境)
            if hasattr(env, 'matter_grid') and env.matter_grid is not None:
                # Review #4: GPU并发竞争处理 - 去重逻辑
                # 将目标坐标转换为网格索引，进行去重
                resolution = getattr(env, 'matter_resolution', 1.0)
                
                # 调试: 记录建造前墙壁数
                walls_before = env.matter_grid.sum().item()
                grid_w = env.matter_grid.shape[3]

                target_gx = (target_x.cpu().numpy() / resolution).astype(np.int32) % grid_w
                target_gy = (target_y.cpu().numpy() / resolution).astype(np.int32) % env.matter_grid.shape[2]

                # 创建唯一索引
                grid_indices = target_gy * grid_w + target_gx
                unique_indices, inverse_idx = np.unique(grid_indices, return_inverse=True)

                # 每个唯一格子只执行一次建造
                construct_indices = torch.where(can_construct)[0]

                for uidx in unique_indices:
                    # 找到所有想建造在这个格子的 agent
                    mask = (inverse_idx == np.where(unique_indices == uidx)[0][0])

                    # 只处理第一个（按能量排序可选）
                    agent_idx = construct_indices[np.where(mask)[0][0]]

                    tx = target_x[mask][0].item()
                    ty = target_y[mask][0].item()

                    if env.is_solid(tx, ty):
                        continue  # 已有物质，跳过

                    # 建造并存储能量
                    built = env.add_matter(tx, ty, stored_energy=self.config.CONSTRUCT_ENERGY_COST)
                    if built:
                        # 调试
                        if self._construct_debug_step % 100 == 0:
                            print(f"    [墙壁] 成功建造! 共 {env.matter_grid.sum().item()} 个")
                        # 扣除能量 (只扣一次)
                        self.state.energies[idx[agent_idx]] -= self.config.CONSTRUCT_ENERGY_COST

        # ========== 分解动作 ==========
        can_deconstruct = deconstruct_activation > DECONSTRUCT_THRESHOLD

        if can_deconstruct.any():
            positions = self.state.positions[idx]
            thetas = self.state.thetas[idx]

            # 假设 agent.radius = 1.0
            agent_radius = 1.0
            forward_dist = agent_radius + self.config.CONSTRUCT_DISTANCE_FACTOR

            target_x = positions[can_deconstruct, 0] + torch.cos(thetas[can_deconstruct]) * forward_dist
            target_y = positions[can_deconstruct, 1] + torch.sin(thetas[can_deconstruct]) * forward_dist

            # 执行分解 (带去重)
            if hasattr(env, 'matter_grid') and env.matter_grid is not None:
                # Review #4: 去重逻辑
                resolution = getattr(env, 'matter_resolution', 1.0)
                grid_w = env.matter_grid.shape[3]

                target_gx = (target_x.cpu().numpy() / resolution).astype(np.int32) % grid_w
                target_gy = (target_y.cpu().numpy() / resolution).astype(np.int32) % env.matter_grid.shape[2]

                grid_indices = target_gy * grid_w + target_gx
                unique_indices, inverse_idx = np.unique(grid_indices, return_inverse=True)

                deconstruct_indices = torch.where(can_deconstruct)[0]

                for uidx in unique_indices:
                    mask = (inverse_idx == np.where(unique_indices == uidx)[0][0])

                    agent_idx = deconstruct_indices[np.where(mask)[0][0]]

                    tx = target_x[mask][0].item()
                    ty = target_y[mask][0].item()

                    if not env.is_solid(tx, ty):
                        continue  # 无物质可分解

                    # 获取存储的能量并回收
                    stored_energy = env.get_matter_energy(tx, ty)
                    if stored_energy is None:
                        stored_energy = self.config.DECONSTRUCT_ENERGY_GAIN

                    if env.remove_matter(tx, ty):
                        # Review #3: 全额返还能量 (只返一次)
                        self.state.energies[idx[agent_idx]] += stored_energy

    def _apply_metabolism(self, batch: ActiveBatch, dt: float):
        """代谢能耗 (v16.4 含BMR基础代谢 + 神经成本)"""
        idx = batch.indices

        # 基础代谢 (最小能量消耗)
        base_cost = self.config.BASE_METABOLISM * dt
        
        # v16.4 基础代谢 (BMR) - 不可避免的静息成本
        # 这打破了"伏地魔"策略: 即使不动也要消耗能量
        basal_cost = self.config.BASAL_COST * dt
        
        # v16.4 神经成本 - 复杂大脑更"饿"
        # 模拟: 高级神经网络需要更多能量维持
        node_counts = self.state.node_counts[idx].float()
        neural_cost = node_counts * self.config.NEURAL_COST * dt

        # 运动代谢 (v16.2: 使用MOVEMENT_PENALTY惩罚盲目移动)
        linear_speed = batch.linear_velocity.norm(dim=-1)  # 线速度模长
        kinetic_cost = (linear_speed + batch.angular_velocity.abs()) * \
                       self.config.MOVEMENT_PENALTY * dt

        # ============================================================
        # 迷宫阻抗 (空间记忆压力)
        # 在高阻抗区域移动将指数级消耗能量
        # ============================================================
        impedance_multiplier = 1.0
        if hasattr(self, 'env') and self.env is not None:
            try:
                if hasattr(self.env, 'impedance_field') and self.env.impedance_field is not None:
                    positions = batch.positions
                    grid_x = (positions[:, 0] / self.env.width * self.env.impedance_field.grid_width).long().clamp(0, self.env.impedance_field.grid_width - 1)
                    grid_y = (positions[:, 1] / self.env.height * self.env.impedance_field.grid_height).long().clamp(0, self.env.impedance_field.grid_height - 1)

                    impedance = self.env.impedance_field.field[0, 0, grid_y, grid_x]

                    # 指数阻抗: impedance=10 -> 2x, impedance=50 -> 50x!
                    impedance_multiplier = 1.0 + (impedance / 10).clamp(max=100)
            except Exception as e:
                pass

        # ============================================================
        # 年龄惩罚 (Metabolic Senescence) - 熵增定律
        # ============================================================
        if self.config.AGE_ENABLED:
            ages = self.state.ages[idx]
            alpha = self.config.AGE_ALPHA
            age_multiplier = 1.0 + alpha * (ages ** 2)
            age_multiplier = age_multiplier.clamp(max=10.0)
        else:
            age_multiplier = 1.0

        # ============================================================
        # v14.1 代谢宽限期 (Metabolic Grace Period)
        # 新拓扑变异的子代在前N步享受代谢折扣
        # ============================================================
        if self.config.METABOLIC_GRACE:
            grace_steps = self.config.METABOLIC_GRACE_STEPS
            grace_discount = self.config.METABOLIC_GRACE_DISCOUNT

            # 计算自上次突变以来的步数
            mutation_ages = self.total_steps - self.state.mutation_timestamp[idx].float()
            in_grace = mutation_ages < grace_steps

            if in_grace.any():
                # 宽限期内: 应用折扣
                grace_multiplier = torch.ones_like(age_multiplier)
                grace_multiplier[in_grace] = grace_discount
                age_multiplier = age_multiplier * grace_multiplier

        # 组合乘数
        total_multiplier = impedance_multiplier * age_multiplier

        # ============================================================
        # v15 认知溢价: 非线性代谢
        # 核心公式: Cost = log(N + 1) × BaseCost (超出FREE_NODES部分)
        # 特点: 规模经济 - 节点越多，每个节点平均成本越低
        # ============================================================
        node_counts = self.state.node_counts[idx].float()

        if self.config.NONLINEAR_METABOLISM:
            # 对数代谢: Cost = log_base(N + 1) × BaseCost
            # 核心思想: 规模经济 - 节点越多，每个节点的边际成本越低
            log_base = self.config.LOG_BASE
            free_nodes = self.config.FREE_NODES

            # 方案A: 免费基础脑容量
            # 前free_nodes个节点完全免费 (但保留最小基础代谢)
            taxable_nodes = (node_counts - free_nodes).clamp(min=0)

            # 方案B: 对数增长 (不是线性的 N*base)
            # log_base(N+1) vs N: 当N=100时, log2(101)=6.7 vs 100
            log_cost = torch.log(taxable_nodes + 1) / torch.log(torch.tensor(log_base))

            # 最终成本 = 对数值 × 基础成本 + 最小基础代谢
            # 关键修复: 即使所有节点都在免费范围,也要支付最小基础代谢
            min_base_cost = 0.1 * base_cost  # 最小代谢 = 10%基础成本
            node_metabolism = log_cost * base_cost + min_base_cost

            # 确保至少有基础代谢 (避免0成本)
            node_metabolism = node_metabolism.clamp(min=0.0)

            # v15 SuperNode成本 = 1节点 (形态学锁定激励)
            if self.config.SUPERNODE_ENABLED:
                n_supernodes = self.state.supernodes[idx].float()
                # SuperNode成本 = 1节点 (不是0.5!)
                # 这会刺激复杂结构的层级嵌套
                supernode_cost = n_supernodes * 1.0 * base_cost
                node_metabolism = node_metabolism + supernode_cost
        else:
            # v14 线性代谢 (向后兼容)
            if self.config.SUPERNODE_ENABLED:
                n_supernodes = self.state.supernodes[idx].float()
                effective_nodes = node_counts - n_supernodes * self.config.SUPERNODE_METABOLIC_BONUS
                node_metabolism = effective_nodes.clamp(min=1.0) * base_cost
            else:
                node_metabolism = node_counts * base_cost

        # v16.4 总代谢 = 基础代谢 + BMR静息成本 + 神经成本 + 运动成本
        # 关键: basal_cost是不可避免的 (即使完全静止也要支付)
        total_cost = (node_metabolism + kinetic_cost + basal_cost + neural_cost) * total_multiplier

        # 扣除活动能量
        self.state.energies[idx] -= total_cost

        # ============================================================
        # v15 能量循环: 代谢能量部分回归环境
        # 模拟"排泄物被分解者回收"或"热辐射再利用"
        # ============================================================
        # v15 能量循环: 代谢能量部分回归环境
        # 模拟"排泄物被分解者回收"或"热辐射再利用"
        # ============================================================
        if self.config.ENERGY_RECIRCULATION_ENABLED and hasattr(self, 'env') and self.env is not None:
            try:
                # 代谢能量回归环境
                recirculated = total_cost * self.config.ENERGY_RECIRCULATION_RATIO
                
                # 注入到能量场 (简化为中心区域)
                if hasattr(self.env, 'energy_field') and self.env.energy_field is not None:
                    if batch.n > 0:
                        total_recirc = recirculated.sum().item()
                        if total_recirc > 0:
                            center = self.env.energy_field.grid_width // 2
                            self.env.energy_field.field[0, 0, center, center] += total_recirc
                        
            except Exception as e:
                pass  # 静默失败

        # ============================================================
        # v16.0: 风场伤害 (Wind Field)
        # 暴露在风中会受到持续伤害
        # ============================================================
        if self.config.MATTER_GRID_ENABLED and hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'wind_field') and self.env.wind_field is not None:
                if self.env.wind_field.enabled:
                    # 使用射线投射检测是否有遮挡
                    positions = batch.positions

                    # 简化实现: 检查智能体位置是否在墙后
                    # (完整实现需要 ray_cast_batch)
                    if hasattr(self.env, 'matter_grid') and self.env.matter_grid is not None:
                        resolution = getattr(self.env, 'matter_resolution', 1.0)
                        grid_w = self.env.matter_grid.shape[3]
                        grid_h = self.env.matter_grid.shape[2]

                        gx = (positions[:, 0] / resolution).long() % grid_w
                        gy = (positions[:, 1] / resolution).long() % grid_h

                        # 采样 matter_grid
                        in_shelter = self.env.matter_grid[0, 0, gy, gx].bool()

                        # 暴露在风中的受伤
                        wind_damage = self.env.wind_field.damage_rate
                        damage_mask = ~in_shelter

                        if damage_mask.any():
                            self.state.energies[idx[damage_mask]] -= wind_damage

        # 更新年龄
        if self.config.AGE_ENABLED:
            self.state.ages[idx] += dt

    # ============================================================================
    # v15 T型迷宫更新 (POMDP)
    # ============================================================================
    def _update_t_maze(self, batch: ActiveBatch):
        """更新T型迷宫状态"""
        if not self.config.T_MAZE_ENABLED:
            return

        idx = batch.indices

        # 更新回合步数
        self.state.t_maze_episode_step[idx] += 1

        # 更新信号计时器
        signal_active = self.state.t_maze_signal_timer[idx] > 0
        self.state.t_maze_signal_timer[idx] = (self.state.t_maze_signal_timer[idx] - 1).clamp(min=0)

        # 信号结束时清除信号
        signal_just_ended = signal_active & (self.state.t_maze_signal_timer[idx] == 0)
        if signal_just_ended.any():
            self.state.t_maze_signal[idx[signal_just_ended]] = 0

        # 决策点检测: 信号结束后BLIND_ZONE步到达决策点
        decision_step = self.config.T_MAZE_SIGNAL_DURATION + self.config.T_MAZE_DECISION_DELAY
        at_decision = (self.state.t_maze_episode_step[idx] == decision_step) & ~self.state.t_maze_decision_made[idx]

        # 回合结束检测: 超过决策点后重置
        reset_step = decision_step + 10  # 决策后10步结束回合
        reset_mask = (self.state.t_maze_episode_step[idx] > reset_step) & ~self.state.t_maze_decision_made[idx]

        if reset_mask.any():
            # 重置回合 (开始新回合)
            new_dirs = torch.randint(0, 2, (reset_mask.sum(),), device=self.device)
            self.state.t_maze_correct_dir[idx[reset_mask]] = new_dirs
            self.state.t_maze_signal[idx[reset_mask]] = new_dirs + 1
            self.state.t_maze_signal_timer[idx[reset_mask]] = self.config.T_MAZE_SIGNAL_DURATION
            self.state.t_maze_episode_step[idx[reset_mask]] = 0
            self.state.t_maze_decision_made[idx[reset_mask]] = False
            # 更新回合计数
            self.state.t_maze_episodes[idx[reset_mask]] += 1

        # 资源周期性可见性
        if self.config.RESOURCE_CYCLE_ENABLED:
            cycle_pos = self.total_steps % self.config.RESOURCE_CYCLE_LENGTH
            visible = cycle_pos < (self.config.RESOURCE_CYCLE_LENGTH - self.config.RESOURCE_FADE_STEPS)
            self.state.resource_visible[idx] = visible

    def _apply_environment_interaction(self, batch: ActiveBatch, env: 'EnvironmentGPU'):
        """环境交互 - 向量化摄食 v16.5 主动感知版"""
        idx = batch.indices

        # 从能量场摄食
        energy_source = None
        if hasattr(env, 'energy_field') and env.energy_field is not None:
            energy_source = env.energy_field
        elif hasattr(env, 'flickering_energy_field') and env.flickering_energy_field is not None:
            energy_source = env.flickering_energy_field
        
        if energy_source is not None:
            try:
                positions = batch.positions  # [N, 2] (x, y)
                N = positions.shape[0]
                
                # ===== v16.5 主动感知: 计算感知效率 =====
                # 核心思想: 不动就看不见！模拟生物界"主动感知"机制
                if self.config.ACTIVE_SENSING_ENABLED:
                    # 计算速度幅度
                    linear_speed = batch.linear_velocity.norm(dim=-1)  # [N]
                    angular_speed = batch.angular_velocity.abs()       # [N]
                    total_speed = linear_speed + angular_speed
                    
                    # 感知效率: 达到阈值速度则100%感知，否则按比例衰减
                    threshold = self.config.ACTIVE_SENSING_THRESHOLD
                    min_eff = self.config.ACTIVE_SENSING_MIN_EFFICIENCY
                    
                    # perception_efficiency = clamp(speed / threshold, min_eff, 1.0)
                    perception_eff = (total_speed / threshold).clamp(min=min_eff, max=1.0)
                    
                    # 对可见能量: 轻度衰减 (保留基础感知)
                    visible_perception = perception_eff * 0.8 + 0.2  # 最低20%感知
                    
                    # 对隐身能量: 严厉衰减 (需要主动感知才能捕获!)
                    invisible_boost = self.config.INVISIBLE_SENSING_BOOST
                    invisible_perception = perception_eff.clamp(min=min_eff) * invisible_boost
                    invisible_perception = invisible_perception.clamp(max=1.0)
                else:
                    # 旧模式: 无主动感知
                    visible_perception = torch.ones(N, device=self.device)
                    invisible_perception = torch.ones(N, device=self.device)
                
                # ===== 能量感知 (v16.3 支持隐身感知) =====
                sensor_range = 15.0
                
                # 检查是否支持全源采样(含隐身)
                if hasattr(energy_source, 'sample_all_sources_batch') and self.config.ENABLE_INVISIBLE_SENSING:
                    # 新方法: 同时获取可见和不可见能量
                    total_energy, invisible_energy = energy_source.sample_all_sources_batch(positions, sensor_range)
                    visible_energy = total_energy - invisible_energy
                elif hasattr(energy_source, 'get_energy_at'):
                    # 回退到旧方法
                    energy_values = torch.zeros(N, device=self.device)
                    for i in range(N):
                        x = positions[i, 0].item()
                        y = positions[i, 1].item()
                        energy_values[i] = energy_source.get_energy_at(x, y, sensor_range)
                    visible_energy = energy_values
                    invisible_energy = torch.zeros(N, device=self.device)
                else:
                    energy_values = energy_source.sample_batch(positions)
                    visible_energy = energy_values
                    invisible_energy = torch.zeros(N, device=self.device)
                
                # ===== 应用主动感知效率 =====
                # 可见能量: 轻度感知衰减
                effective_visible = visible_energy * visible_perception
                # 隐身能量: 严厉感知衰减 (需要"动起来"才能感知!)
                effective_invisible = invisible_energy * invisible_perception
                
                # ===== 能量消耗 =====
                feed_rate = 0.3
                feed_amount_visible = effective_visible * feed_rate
                feed_amount_invisible = effective_invisible * feed_rate
                
                # 处理可见能量消耗 (从field扣除)
                if feed_amount_visible.sum() > 0:
                    grid_w = energy_source.grid_width
                    grid_h = energy_source.grid_height
                    grid_x = (positions[:, 0] / energy_source.resolution).long().clamp(0, grid_w - 1)
                    grid_y = (positions[:, 1] / energy_source.resolution).long().clamp(0, grid_h - 1)
                    
                    flat_field = energy_source.field.view(-1)
                    flat_indices = grid_y * grid_w + grid_x
                    
                    total_requested = torch.zeros(grid_h * grid_w, device=self.device)
                    total_requested.scatter_add_(0, flat_indices, feed_amount_visible)
                    
                    actual_consumed = torch.min(total_requested, flat_field)
                    flat_field = flat_field - actual_consumed
                    energy_source.field.view(-1)[:] = flat_field
                    
                    # 按比例分配
                    safe_requested = total_requested.clone()
                    safe_requested[safe_requested == 0] = 1.0
                    supply_ratio = actual_consumed / safe_requested
                    agent_supply_ratio = supply_ratio[flat_indices]
                    actual_feed_visible = feed_amount_visible * agent_supply_ratio
                else:
                    actual_feed_visible = torch.zeros(N, device=self.device)
                
                # ===== v16.7 极端不对称：0.1x可见 vs 20x隐身！=====
                # 核心修复：彻底饿死伏地魔
                # 可见=0.1x (垃圾食品)，隐身=20x (超级大补丸)
                
                if hasattr(self.config, 'VISIBLE_REWARD_MULTIPLIER'):
                    # v16.7新版: 极端不对称
                    # 可见阶段：0.1x (垃圾食品，连代谢都抵消不了)
                    # 隐身阶段：20x (只有主动出击才能赢)
                    premium_visible = self.config.VISIBLE_REWARD_MULTIPLIER
                    premium_invisible = self.config.INVISIBLE_REWARD_MULTIPLIER
                elif self.config.COGNITIVE_PREMIUM_ONLY_INVISIBLE:
                    # v16.6: 1x vs 10x
                    premium_visible = 1.0
                    premium_invisible = self.config.COGNITIVE_PREMIUM_MULTIPLIER
                else:
                    # 旧版: 全部10x (导致伏地魔)
                    premium_visible = self.config.COGNITIVE_PREMIUM_MULTIPLIER
                    premium_invisible = self.config.COGNITIVE_PREMIUM_MULTIPLIER
                
                # 修复: premium应该应用于实际获取的能量，而不是原始请求量!
                actual_feed_visible = actual_feed_visible * premium_visible
                actual_feed_invisible = feed_amount_invisible * premium_invisible
                
                # 总能量获取 = 可见部分(1x) + 隐身暴击部分(10x)
                actual_feed = actual_feed_visible + actual_feed_invisible
                
                self.state.energies[idx] += actual_feed
                
                # 记录统计 (如果有统计追踪器)
                if hasattr(self, 'stats') and self.stats is not None:
                    pass  # 可扩展统计
                
            except Exception as e:
                pass

    def _apply_predation(self, batch: ActiveBatch, brain_outputs: torch.Tensor):
        """
        黑暗森林同类捕食
        ================
        黑暗森林 - 真正的捕食者军备竞赛
        使用 torch.cdist 计算全局距离矩阵

        阶段1: 同类相食 - 发现吃同类比吃食物爽
        阶段2: 大灭绝 - 捕食者过多导致饥荒
        阶段3: 军备竞赛 - 猎物进化逃跑，捕食者进化追踪
        """
        if batch.n < 2:
            return

        idx = batch.indices
        positions = batch.positions
        energies = batch.energies

        # 获取攻击输出 (Channel 3)
        attack_power = torch.relu(brain_outputs[:, 3])  # [N]

        # 攻击阈值: 只有超过阈值才发动攻击
        ATTACK_THRESHOLD = 0.5
        potential_predators = attack_power > ATTACK_THRESHOLD

        if not potential_predators.any():
            return

        # 爆发能量成本 (发动攻击本身就很昂贵)
        STRIKE_COST = 2.0
        strike_mask = potential_predators & (energies > STRIKE_COST)
        if strike_mask.any():
            self.state.energies[idx[strike_mask]] -= STRIKE_COST

        # 使用 torch.cdist 计算全局距离矩阵 [N, N]
        dist_matrix = torch.cdist(positions, positions)  # O(N²) but GPU accelerated

        # 攻击半径
        ATTACK_RADIUS = 3.0
        close_encounters = (dist_matrix < ATTACK_RADIUS) & (dist_matrix > 0.1)

        # 对每个捕食者，找到最近的猎物
        # 近距离掩码 [N, N]
        valid_attacks = close_encounters & potential_predators.unsqueeze(0)

        # 计算每个捕食者能攻击多少猎物
        n_targets = valid_attacks.float().sum(dim=1).clamp(min=1)  # [N]

        # 有效攻击强度 = 攻击功率 / 猎物数量
        effective_power = attack_power / n_targets  # [N]

        # ============================================================
        # 吸血转移 (Vampiric Transfer)
        # 捕食者获得能量，受害者损失能量
        # ============================================================
        predation_occurred = False
        for i in range(batch.n):
            if not potential_predators[i]:
                continue

            # 找到这个捕食者能攻击的所有猎物
            prey_indices = valid_attacks[i].nonzero(as_tuple=True)[0]
            if len(prey_indices) == 0:
                continue

            predation_occurred = True

            # 攻击强度
            power = effective_power[i].item()

            # 从每个猎物那里"吸血"
            for j in prey_indices:
                if energies[j] > 5.0:  # 猎物必须有足够能量
                    # 偷取能量 (50% 转移率)
                    drain = min(power * 5.0, energies[j].item() * 0.3)
                    self.state.energies[idx[j]] -= drain
                    self.state.energies[idx[i]] += drain * 0.8  # 80% 效率

        return predation_occurred

    def _trigger_event_mechanisms(self, batch: ActiveBatch, env):
        """
        v14.1 事件触发的演化机制

        在特定事件发生时调用:
        - 捕食事件 (red_queen)
        - 繁衍事件
        - 死亡事件
        """
        if batch.n == 0 or not self.event_mechanisms:
            return

        world = {
            'dt': 1.0,
            'step': getattr(self, 'total_steps', 0),
            'env': env,
            'generation': getattr(self, 'generation', self.total_steps // 100),
        }

        for mechanism in self.event_mechanisms:
            try:
                mechanism.apply(batch, world)
            except Exception as e:
                print(f"  ⚠️ {mechanism.name} 事件触发失败: {e}")

    def _apply_evo_mechanisms(self, batch: ActiveBatch, env):
        """
        v14.1 演化机制: 每Step调用的可开关机制

        触发机制:
        - morphology: 物理碰撞/吸附 (在_apply_physics中处理)
        - ontogeny: 年龄增长和阶段转换 (每Step)
        - stigmergy: 信息素场更新 (每Step)
        - thermal: 温度场影响 (每Step)
        """
        if batch.n == 0 or not self.evo_mechanisms:
            return

        # 构建world字典供法则使用
        world = {
            'dt': 1.0,
            'step': getattr(self, 'total_steps', 0),
            'env': env,
        }

        # 获取活跃agent列表（需要兼容基因组系统）
        # 注意: 这里需要将batch转换为Agent对象列表
        # 对于GPU批处理，我们直接传递batch和相关状态
        for mechanism in self.evo_mechanisms:
            try:
                mechanism.apply(batch, world)
            except Exception as e:
                print(f"  ⚠️ {mechanism.name} 应用失败: {e}")

    def _apply_reward_hebbian(self, batch: ActiveBatch):
        """
        v14.0 能量调制赫布学习 (Reward-modulated Hebbian Learning)

        核心算法:
        1. 计算能量变化 ΔE = E_t - E_{t-1}
        2. 死区过滤: 只有|ΔE| > DEADZONE 才触发
        3. 多巴胺信号: dopamine = sign(ΔE) * min(|ΔE|/50, 1.0)
        4. 资格迹更新: trace = decay * trace + pre * post
        5. 权重更新: w += lr * dopamine * trace

        涌现: Agent学会"记住"导致能量增加的行为
        """
        if batch.n == 0:
            return

        idx = batch.indices

        # 1. 计算能量变化 ΔE
        current_energy = self.state.energies[idx]
        prev_energy = self.state.prev_energies[idx]
        energy_delta = current_energy - prev_energy  # [N]

        # 2. 死区过滤 (Gemini建议)
        deadzone = self.config.HEBBIAN_DEADZONE
        significant_change = energy_delta.abs() > deadzone

        if not significant_change.any():
            # 无显著变化，只更新prev_energy
            self.state.prev_energies[idx] = current_energy
            return

        # 3. 计算多巴胺信号
        if self.config.HEBBIAN_REWARD_MODULATION:
            # dopamine = sign(ΔE) * min(|ΔE|/50, 1.0)
            dopamine = torch.sign(energy_delta) * (energy_delta.abs() / 50.0).clamp(max=1.0)
            # 只对显著变化的Agent计算
            dopamine = dopamine * significant_change.float()
        else:
            dopamine = torch.ones_like(energy_delta) * 0.1

        # ================================================================
        # 4. 简化版Hebbian更新 (不需要完整的脑矩阵)
        # 假设每个Agent有一条"隐式边"记录协同激活
        # 这里用能量变化作为全局奖励信号
        # ================================================================

        # 简化实现: 随机选择一些"边"进行更新
        # 实际上应该从brain_matrix中获取边权重，但这里先用简化版
        max_edges = self.state.hebbian_plastic_mask.shape[1]

        # 为每个活跃Agent随机更新一些"边"
        n_edges_to_update = min(batch.n, 10)  # 每步最多更新10条
        if batch.n >= n_edges_to_update:
            # 随机选择一些Agent
            perm = torch.randperm(batch.n)[:n_edges_to_update]
            selected_idx = idx[perm]

            # 能量增加 -> 强化; 能量减少 -> 弱化
            for i, agent_idx in enumerate(selected_idx):
                agent_delta = dopamine[perm[i]].item()

                # 更新该Agent的隐式学习状态
                # 这里用简单的标量记录"学习进度"
                if not hasattr(self, '_hebbian_progress'):
                    self._hebbian_progress = torch.zeros(self.max_agents, device=self.device)

                # 学习进度 += dopamine * lr
                lr = self.config.HEBBIAN_BASE_LR
                self._hebbian_progress[agent_idx] += agent_delta * lr
                # 限制范围
                self._hebbian_progress[agent_idx] = self._hebbian_progress[agent_idx].clamp(-1.0, 1.0)

        # 5. 更新prev_energy
        self.state.prev_energies[idx] = current_energy
        self.hebbian_step_count += 1

        # 每N步打印统计
        if self.hebbian_step_count % 500 == 0:
            n_learning = significant_change.sum().item()
            avg_delta = energy_delta.mean().item()
            print(f"  🧠 Hebbian: {n_learning}/{batch.n} agents learning, avg ΔE={avg_delta:.2f}")

    def _process_deaths(self, batch: ActiveBatch, env) -> int:
        """
        死亡结算 - 鲸落机制
        鲸落能量 = Biomass (节点×10) + max(0, 活动能量)
        """
        idx = batch.indices

        # 判定死亡
        death_mask = batch.energies <= 0
        if not death_mask.any():
            return 0

        dead_indices = idx[death_mask]
        n_deaths = len(dead_indices)

        # 计算鲸落能量
        node_counts = self.state.node_counts[dead_indices].float()
        biomass_energy = node_counts * self.config.BIOMASS_PER_NODE
        active_energy = torch.clamp(self.state.energies[dead_indices], min=0)
        whale_energy = biomass_energy + active_energy * self.config.WHALE_RETURN_RATIO

        # 写入环境
        if env is not None and hasattr(env, 'energy_field') and env.energy_field is not None:
            try:
                death_positions = self.state.positions[dead_indices]
                env.energy_field.scatter_add_(death_positions, whale_energy)
            except Exception:
                pass

        # 标记死亡
        self.alive_mask[dead_indices] = False
        self._indices_dirty = True

        # 从基因组字典移除
        for di in dead_indices.tolist():
            if di in self.genomes:
                del self.genomes[di]

        return n_deaths

    def _run_subgraph_mining(self):
        """
        v14.0 演化棘轮: 子图挖掘 + SuperNode冻结

        每N步运行一次:
        1. 收集Top 10% Elite Agent
        2. 挖掘频繁子图
        3. 注册为SuperNode
        4. 压缩Agent大脑
        """
        if not hasattr(self, 'subgraph_mining_enabled') or not self.subgraph_mining_enabled:
            return

        if len(self.genomes) < 10:
            print(f"\n🧬 [Step {self.total_steps}] 基因组不足: {len(self.genomes)}")
            return

        alive_batch = self.get_active_batch()
        if alive_batch.n < 10:
            return

        print(f"\n🧬 [Step {self.total_steps}] 开始子图挖掘...")

        # 挖掘频繁子图
        try:
            patterns = self.subgraph_miner.mine(
                self.genomes,
                alive_batch.indices,
                top_k=max(10, alive_batch.n // 10)  # Top 10%
            )

            # 调试: 打印发现的模式数量和详细信息
            print(f"   🔍 Top K={max(10, alive_batch.n // 10)}, 发现 {len(patterns)} 个模式")
            if len(patterns) > 0:
                for i, p in enumerate(patterns[:3]):
                    print(f"      模式{i}: 节点={p.node_types}, 支持度={p.support}")

            # 注册新SuperNode
            for pattern in patterns[:2]:  # 最多注册2个
                spec = self.supernode_registry.register(
                    pattern,
                    discovered_at_step=self.total_steps
                )
                if spec:
                    print(f"   ✅ 新SuperNode: {spec.name}, 成本节省: {spec.original_cost - spec.frozen_cost:.4f}")

            # 统计
            stats = self.supernode_registry.get_stats()
            print(f"   📊 SuperNode统计: {stats['n_supernodes']}个, 共节省{stats['total_savings']:.4f}")

        except Exception as e:
            import traceback
            print(f"   ⚠️ 子图挖掘失败: {e}")
            traceback.print_exc()

    def _process_reproduction(self, batch: ActiveBatch) -> int:
        """
        分裂结算 - 能量驱动有丝分裂
        """
        idx = batch.indices

        # 判定分裂
        repro_mask = (batch.energies > self.config.REPRODUCTION_THRESHOLD) & \
                     (batch.energies > self.config.MIN_REPRO_ENERGY)

        if not repro_mask.any():
            return 0

        parent_indices = idx[repro_mask]
        n_parents = len(parent_indices)

        # 寻找空槽位
        empty_slots = (~self.alive_mask).nonzero(as_tuple=True)[0]
        if len(empty_slots) == 0:
            return 0

        # 容量限制
        n_spawn = min(n_parents, len(empty_slots))

        parent_indices = parent_indices[:n_spawn]
        child_indices = empty_slots[:n_spawn]

        # 能量平分
        parent_energy = self.state.energies[parent_indices]
        child_energy = parent_energy * self.config.CHILD_ENERGY_RATIO

        self.state.energies[parent_indices] = parent_energy * (1 - self.config.CHILD_ENERGY_RATIO)
        self.state.energies[child_indices] = child_energy

        # 结构能量继承
        self.state.structural_energy[child_indices] = self.state.structural_energy[parent_indices]

        # 位置偏移 (避免数值奇点)
        offset = torch.randn(n_spawn, 2, device=self.device) * self.config.SPAWN_RADIUS
        self.state.positions[child_indices] = self.state.positions[parent_indices] + offset

        # 环形边界
        self.state.positions[child_indices, 0] = self.state.positions[child_indices, 0] % self.env_width
        self.state.positions[child_indices, 1] = self.state.positions[child_indices, 1] % self.env_height

        # 朝向继承 + 扰动
        self.state.thetas[child_indices] = self.state.thetas[parent_indices] + \
            torch.randn(n_spawn, device=self.device) * 0.1

        # ============================================================
        # 层次一：参数变异 (Parametric Mutation)
        # 子代 = 父代 + 高斯噪声
        # ============================================================

        # 速度继承 + 变异
        self.state.linear_velocity[child_indices] = self.state.linear_velocity[parent_indices] * 0.5
        self.state.angular_velocity[child_indices] = self.state.angular_velocity[parent_indices] * 0.5

        # 添加速度变异 (10% 概率，每个维度 ±噪声)
        if self.config.MUTATION_RATE > 0:
            vel_noise_mask = torch.rand(n_spawn, device=self.device) < self.config.MUTATION_RATE
            vel_noise = torch.randn(n_spawn, device=self.device) * 0.5
            self.state.linear_velocity[child_indices] += vel_noise_mask.float() * vel_noise

            ang_noise_mask = torch.rand(n_spawn, device=self.device) < self.config.MUTATION_RATE
            ang_noise = torch.randn(n_spawn, device=self.device) * 0.2
            self.state.angular_velocity[child_indices] += ang_noise_mask.float() * ang_noise

        # 节点数量继承 + 拓扑变异
        self.state.node_counts[child_indices] = self.state.node_counts[parent_indices]

        # 以低概率增加节点 (拓扑变异)
        add_node_prob = 0.05  # 5% 概率增加1个节点
        add_mask = torch.rand(n_spawn, device=self.device) < add_node_prob
        self.state.node_counts[child_indices] += add_mask.long()

        # 以低概率减少节点 (拓扑变异)
        remove_node_prob = 0.02  # 2% 概率减少1个节点
        remove_mask = (torch.rand(n_spawn, device=self.device) < remove_node_prob) & (self.state.node_counts[child_indices] > 2)
        self.state.node_counts[child_indices] -= remove_mask.long()

        # 限制节点数量范围
        self.state.node_counts[child_indices] = torch.clamp(self.state.node_counts[child_indices], min=1, max=20)

        # ============================================================
        # v14.1 代谢宽限期: 标记拓扑突变时间
        # 任何拓扑变异都会触发宽限期
        # ============================================================
        if self.config.METABOLIC_GRACE:
            topology_mutated = add_mask | remove_mask
            if topology_mutated.any():
                self.state.mutation_timestamp[child_indices[topology_mutated]] = self.total_steps

        # ============================================================
        # 子代 age = 0 (获得新生)
        # ============================================================
        self.state.ages[child_indices] = 0.0

        # ============================================================
        # v15 T型迷宫初始化: 新agent开始新回合
        # ============================================================
        if self.config.T_MAZE_ENABLED:
            # 随机选择正确方向 (0=左, 1=右)
            correct_dirs = torch.randint(0, 2, (n_spawn,), device=self.device)
            self.state.t_maze_correct_dir[child_indices] = correct_dirs

            # 设置信号 (信号持续T_MAZE_SIGNAL_DURATION步)
            self.state.t_maze_signal[child_indices] = correct_dirs + 1  # 1=左, 2=右
            self.state.t_maze_signal_timer[child_indices] = self.config.T_MAZE_SIGNAL_DURATION

            # 重置回合状态
            self.state.t_maze_episode_step[child_indices] = 0
            self.state.t_maze_decision_made[child_indices] = False

            # 注意: episodes和correct在agent死亡时重置，这里继承父代

        # ============================================================
        # 超级节点继承 (演化棘轮)
        # 子代继承父代的超级节点数量
        # 有一定概率增加新的超级节点 (进化!)
        # ============================================================
        self.state.supernodes[child_indices] = self.state.supernodes[parent_indices]

        # 进化新超级节点: 5%概率
        if self.config.SUPERNODE_ENABLED and n_spawn > 0:
            evolve_new = torch.rand(n_spawn, device=self.device) < 0.05
            self.state.supernodes[child_indices[evolve_new]] += 1

        # 限制超级节点数量
        max_supernodes = (self.state.node_counts[child_indices] // 2).long()
        self.state.supernodes[child_indices] = torch.minimum(
            self.state.supernodes[child_indices],
            max_supernodes
        ).clamp(min=0)

        # ============================================================
        # 层次一续：大脑权重矩阵变异 (Weight Mutation)
        # ============================================================

        # 如果有大脑矩阵，对子代进行权重变异
        if self.brain_matrix is not None and self.brain_masks is not None:
            max_nodes = self.brain_matrix.shape[1]

            # 获取子代的大脑矩阵切片
            W_child = self.brain_matrix[child_indices]  # [n_spawn, max_nodes, max_nodes]
            M_child = self.brain_masks[child_indices]   # [n_spawn, max_nodes, max_nodes]

            # 1. 突触权重微调 (Weight Shift) - 连续变异
            # 10% 的非零权重发生微小偏移
            weight_mutate_prob = 0.1
            weight_mutate_scale = 0.2

            # 只对已启用的连接进行变异
            active_weights = W_child * M_child.float()
            weight_mutate_mask = (torch.rand_like(W_child) < weight_mutate_prob) & M_child

            if weight_mutate_mask.any():
                noise = torch.randn_like(W_child) * weight_mutate_scale
                W_child = W_child + weight_mutate_mask.float() * noise

            # 2. 突触断裂 (Edge Break) - 拓扑变异
            # 2% 的非零连接断裂
            break_prob = 0.02
            break_mask = (torch.rand_like(W_child) < break_prob) & M_child
            W_child = W_child.clone()
            W_child[break_mask] = 0.0
            M_child[break_mask] = False

            # 3. 突触生成 (Edge Genesis) - 拓扑变异
            # 3% 的空槽位生成新连接
            genesis_prob = 0.03
            empty_mask = ~M_child  # 当前是 False 的位置
            genesis_mask = (torch.rand_like(W_child) < genesis_prob) & empty_mask

            if genesis_mask.any():
                # 赋予随机初始权重
                new_weights = torch.randn_like(W_child) * 0.5
                W_child[genesis_mask] = new_weights[genesis_mask]
                M_child[genesis_mask] = True

            # 写回
            self.brain_matrix[child_indices] = W_child
            self.brain_masks[child_indices] = M_child

        # 标记存活
        self.alive_mask[child_indices] = True
        self._indices_dirty = True

        # 基因组复制与变异
        for pi, ci in zip(parent_indices.tolist(), child_indices.tolist()):
            if pi in self.genomes:
                parent_genome = self.genomes[pi]
                child_genome = parent_genome.mutate(rate=self.config.MUTATION_RATE)
                self.genomes[ci] = child_genome

        return n_spawn

    def _apply_boundaries(self, batch: ActiveBatch):
        """环形世界边界"""
        idx = batch.indices
        self.state.positions[idx, 0] = self.state.positions[idx, 0] % self.env_width
        self.state.positions[idx, 1] = self.state.positions[idx, 1] % self.env_height

    # ============================================================================
    # 大脑管理 (保留兼容)
    # ============================================================================

    def _load_pretrained_genomes(self, n_agents: int) -> List['OperatorGenome']:
        """从预训练文件加载脑结构"""
        import json
        from copy import deepcopy

        filepath = self.config.PRETRAINED_STRUCTURES_FILE
        top_n = self.config.PRETRAINED_TOP_N
        dup_factor = self.config.PRETRAINED_DUPLICATE_FACTOR

        print(f"[预加载] 从 {filepath} 加载Top {top_n} 结构...")

        with open(filepath, 'r') as f:
            data = json.load(f)

        structures = data.get('structures', {})
        if isinstance(structures, dict):
            structures = list(structures.values())

        # 按复杂度排序
        structures = sorted(structures, key=lambda s: s.get('complexity_score', 0), reverse=True)
        structures = structures[:top_n]

        if not structures:
            print("[预加载] ⚠️ 未找到有效结构，回退到寒武纪初始化")
            return None

        # 转换为OperatorGenome
        genomes = []
        for s in structures:
            try:
                genome = self._genome_from_structure(s)
                genomes.append(genome)
                print(f"  ✅ {s.get('structure_id', '?')}: {s.get('complexity_score', 0):.2f}分")
            except Exception as e:
                print(f"  ❌ 结构转换失败: {e}")

        if not genomes:
            return None

        # 复制填充到n_agents
        result = []
        agent_id = 0
        while agent_id < n_agents:
            for g in genomes:
                if agent_id >= n_agents:
                    break
                result.append(deepcopy(g))
                agent_id += 1

        print(f"[预加载] ✅ 创建 {len(result)} 个Agent，使用 {len(genomes)} 种结构")
        return result

    def _genome_from_structure(self, struct: Dict) -> 'OperatorGenome':
        """从结构字典创建基因组"""
        from core.eoe.node import Node, NodeType
        from core.eoe.genome import OperatorGenome

        genome = OperatorGenome()

        # 添加节点
        node_map = {}
        next_node_id = 0

        for i, node_type in enumerate(struct['nodes']):
            node = Node(node_id=next_node_id, node_type=NodeType(node_type))
            genome.add_node(node)
            node_map[i] = next_node_id
            next_node_id += 1

        # 添加边
        for edge in struct.get('edges', []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 3:
                src, tgt, w = edge[0], edge[1], edge[2]
            elif isinstance(edge, dict):
                src = edge.get('source_id', edge.get('source', 0))
                tgt = edge.get('target_id', edge.get('target', 0))
                w = edge.get('weight', 0.5)
            else:
                continue

            if src in node_map and tgt in node_map:
                genome.add_edge(node_map[src], node_map[tgt], w)

        return genome

    def _create_cambrian_genomes(self, n_agents: int) -> List['OperatorGenome']:
        """寒武纪初始化: 随机生成初始基因组"""
        import numpy as np
        from core.eoe.node import Node, NodeType
        from core.eoe.genome import OperatorGenome

        print(f"[寒武纪] 随机初始化 {n_agents} 个Agent...")

        genomes = []
        for i in range(n_agents):
            genome = OperatorGenome()

            n_nodes = np.random.randint(
                self.config.CAMBRIAN_MIN_NODES,
                self.config.CAMBRIAN_MAX_NODES + 1
            )

            # 构建节点类型链
            node_types = [NodeType.SENSOR]
            for _ in range(n_nodes - 2):
                rt = np.random.random()
                if rt < self.config.CAMBRIAN_DELAY_PROB:
                    node_types.append(NodeType.DELAY)
                elif rt < self.config.CAMBRIAN_DELAY_PROB + self.config.CAMBRIAN_MULTIPLY_PROB:
                    node_types.append(NodeType.MULTIPLY)
                else:
                    node_types.append(NodeType.THRESHOLD)
            node_types.append(NodeType.ACTUATOR)

            # 添加节点
            for j, nt in enumerate(node_types):
                genome.add_node(Node(node_id=j, node_type=nt))

            # 添加边 (前馈为主)
            for src in range(len(node_types) - 1):
                if np.random.random() < 0.7:
                    tgt = np.random.randint(src + 1, len(node_types))
                    weight = self.config.SILENT_WEIGHT if self.config.SILENT_MUTATION else np.random.uniform(-0.5, 0.5)
                    genome.add_edge(src, tgt, weight=weight)

            # 确保SENSOR有输出
            if not any(e['source_id'] == 0 for e in genome.edges):
                tgt = np.random.randint(1, len(node_types))
                genome.add_edge(0, tgt, weight=self.config.SILENT_WEIGHT)

            genomes.append(genome)

        return genomes

    def set_brains(self, genomes: List['OperatorGenome'] = None):
        """设置大脑矩阵 (异构大脑掩码对齐)

        Args:
            genomes: 可选的基因组列表。如果PRETRAINED_INIT启用且genomes为空，则自动从文件加载
        """
        # 只为活着的 Agent 构建
        batch = self.get_active_batch()
        n_alive = batch.n

        if n_alive == 0:
            return

        # ================================================================
        # 预加载脑结构机制: 如果启用且未提供genomes，则从文件加载
        # ================================================================
        if genomes is None and self.config.PRETRAINED_INIT and self.config.PRETRAINED_STRUCTURES_FILE:
            genomes = self._load_pretrained_genomes(n_alive)

        # 如果仍然没有genomes且启用了寒武纪初始化，则随机生成
        if not genomes and self.config.CAMBRIAN_INIT:
            genomes = self._create_cambrian_genomes(n_alive)

        if not genomes:
            raise ValueError("No genomes provided and neither pretrained nor cambrian init enabled")

        max_nodes = max(len(g.nodes) for g in genomes[:n_alive]) if genomes else 4
        max_edges = max(len(g.edges) for g in genomes[:n_alive]) if genomes else 4

        # 确保max_nodes至少为10 (传感器维度)
        max_nodes = max(max_nodes, 10)

        # 预分配大脑矩阵
        self.brain_matrix = torch.zeros(
            self.max_agents, max_nodes, max_nodes,
            device=self.device, dtype=torch.float32
        )

        self.brain_masks = torch.zeros(
            self.max_agents, max_nodes, max_nodes,
            device=self.device, dtype=torch.bool
        )

        self.node_counts_tensor = torch.zeros(
            self.max_agents, device=self.device, dtype=torch.long
        )

        # 填充活着的 Agent
        for i, (idx, genome) in enumerate(zip(batch.indices.tolist(), genomes[:n_alive])):
            # 同时填充genomes字典（供复杂度追踪器使用）
            self.genomes[idx] = genome

            nodes = list(genome.nodes.values())
            node_ids = {n.node_id: idx for idx, n in enumerate(nodes)}

            self.node_counts_tensor[idx] = len(nodes)
            self.state.node_counts[idx] = len(nodes)

            # 处理edges (可能是dict或对象)
            for edge in genome.edges:
                if isinstance(edge, dict):
                    src = edge.get('source_id', edge.get('source'))
                    tgt = edge.get('target_id', edge.get('target'))
                    w = edge.get('weight', 0.5)
                else:
                    src = edge.source if hasattr(edge, 'source') else edge.source_id
                    tgt = edge.target if hasattr(edge, 'target') else edge.target_id
                    w = edge.weight if hasattr(edge, 'weight') else 0.5

                if src in node_ids and tgt in node_ids:
                    src_idx = node_ids[src]
                    tgt_idx = node_ids[tgt]
                    self.brain_matrix[idx, src_idx, tgt_idx] = w
                    self.brain_masks[idx, src_idx, tgt_idx] = True

        # BMR 预编译
        self._compute_bmr_precompiled(genomes[:n_alive])

        print(f"  ✅ 大脑矩阵: {self.brain_matrix.shape}")

    def _compute_bmr_precompiled(self, genomes: List['OperatorGenome']):
        """预编译 BMR"""
        node_costs = torch.tensor([0.001, 0.001, 0.002, 0.001, 0.005, 0.01, 0.02], device=self.device)

        bmr_values = []
        for genome in genomes:
            nodes = list(genome.nodes.values()) if hasattr(genome.nodes, 'values') else genome.nodes
            node_cost = sum(node_costs[min(n.node_type.value, 6)].item() for n in nodes)
            # 处理edges (可能是list)
            edges = genome.edges if isinstance(genome.edges, list) else list(genome.edges.values())
            edge_cost = len([e for e in edges if (e.get('weight') if isinstance(e, dict) else e.get('weight', 0)) != 0]) * 0.0005
            bmr_values.append(node_cost + edge_cost)

        batch = self.get_active_batch()
        self.agent_bmr[batch.indices] = torch.tensor(bmr_values, device=self.device)

    def forward_brains(self, sensors: torch.Tensor) -> torch.Tensor:
        """批量前向传播"""
        # v16.0: 根据配置确定输出通道数
        n_outputs = self.config.N_BRAIN_OUTPUTS if not self.config.MATTER_GRID_ENABLED else self.config.N_BRAIN_OUTPUTS_V16

        if self.brain_matrix is None:
            return torch.zeros(sensors.shape[0], n_outputs, device=self.device)

        batch = self.get_active_batch()
        idx = batch.indices
        n = batch.n

        if n == 0:
            return torch.zeros(0, n_outputs, device=self.device)

        sensor_dim = sensors.shape[1]
        max_nodes = self.brain_matrix.shape[1]

        # 填充传感器到max_nodes维度
        if sensor_dim < max_nodes:
            padding = torch.zeros(n, max_nodes - sensor_dim, device=self.device)
            sensors = torch.cat([sensors, padding], dim=1)

        # 获取当前 Agent 的脑矩阵
        W = self.brain_matrix[idx, :max_nodes, :32]
        M = self.brain_masks[idx, :max_nodes, :32]

        W_masked = W * M
        hidden = torch.bmm(sensors.unsqueeze(1), W_masked).squeeze(1)
        hidden = torch.relu(hidden)

        # v16.0: 根据配置使用不同的输出维度
        W2 = self.brain_matrix[idx, :32, :n_outputs]
        M2 = self.brain_masks[idx, :32, :n_outputs]
        W2_masked = W2 * M2
        output = torch.bmm(hidden.unsqueeze(1), W2_masked).squeeze(1)

        return output

    # ============================================================================
    # 兼容方法 (旧版 API)
    # ============================================================================

    def get_sensors(self, env: 'EnvironmentGPU') -> torch.Tensor:
        """批量获取传感器 (兼容旧版)"""
        batch = self.get_active_batch()
        if batch.n == 0:
            return torch.zeros(0, 10, device=self.device)  # 9 field + 1 energy

        try:
            field_values = env.get_field_values(batch.positions)
        except Exception:
            field_values = torch.zeros(batch.n, 9, device=self.device)  # 实际是9维

        energy_norm = torch.clamp(batch.energies / 200.0, 0, 1)

        return torch.cat([field_values, energy_norm.unsqueeze(1)], dim=1)

    def step_old(self, brain_outputs: torch.Tensor, dt: float = 1.0) -> Dict:
        """旧版 step (兼容)"""
        return self.step(dt=dt, brain_fn=lambda _: brain_outputs)

    # ============================================================================
    # 统计
    # ============================================================================

    def get_population_stats(self) -> Dict:
        """获取种群统计"""
        batch = self.get_active_batch()
        if batch.n == 0:
            return {'n_alive': 0, 'mean_energy': 0, 'max_energy': 0}

        return {
            'n_alive': batch.n,
            'mean_energy': batch.energies.mean().item(),
            'max_energy': batch.energies.max().item(),
            'min_energy': batch.energies.min().item()
        }


# ============================================================================
# 兼容导出
# ============================================================================

AgentState = AgentState  # 保留兼容

__all__ = ['BatchedAgents', 'AgentState', 'PoolConfig', 'ActiveBatch']