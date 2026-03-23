"""
EOE 基础配置 (Base Configuration)
=================================
保留 PoolConfig 和 AgentState 的完整定义
所有参数使用默认值，不做任何修改

使用方式:
    from configs import PoolConfig
    config = PoolConfig()
    
    # 或继承修改
    class MyConfig(PoolConfig):
        BASE_METABOLISM = 0.01
"""

from dataclasses import dataclass
from typing import List
import torch


class PoolConfig:
    """掩码池配置"""
    # 池大小
    MAX_AGENTS = 10000

    # 繁衍参数
    REPRODUCTION_THRESHOLD = 180.0
    CHILD_ENERGY_RATIO = 0.5
    MIN_REPRO_ENERGY = 30.0
    SPAWN_RADIUS = 0.5
    
    # v17.2: 高压配置 - 强制节点演化
    MUTATION_RATE = 0.5  # 提高突变率促进结构演化

    # 鲸落参数
    WHALE_RETURN_RATIO = 0.8
    BIOMASS_PER_NODE = 10.0

    # 代谢参数
    # v17.2: 平衡代谢压力 (不要太高以免种群崩溃)
    BASE_METABOLISM = 0.008  # 温和压力
    ACTIVATION_COST = 0.001
    
    # v16.30 运动惩罚参数 - 追捕昂贵
    MOVEMENT_PENALTY = 0.5  # v17.5: 提高10倍 - 追捕需要消耗更多能量

    # v16.3 认知溢价 (Cognitive Premium) - 隐身进食10倍奖励
    COGNITIVE_PREMIUM_MULTIPLIER = 10.0  # 隐身状态进食获得10倍能量
    ENABLE_INVISIBLE_SENSING = True      # 允许感知隐身能量(弱感知)
    
    # v16.4 基础代谢 (BMR) - 打破"伏地魔"策略
    # v16.29: 取消静息代谢以确保能量收支平衡
    BASAL_COST = 0.0  # 静息基础代谢 (v16.29: 取消)
    # v17.2: 降低神经成本鼓励复杂化
    NEURAL_COST = 0.0008
    
    # v16.5 主动感知 (Active Sensing) - 运动-感知耦合
    # v16.29: 调整参数让感知更容易
    ACTIVE_SENSING_ENABLED = True        # 启用主动感知
    ACTIVE_SENSING_THRESHOLD = 0.1       # v16.29: 降低阈值
    ACTIVE_SENSING_MIN_EFFICIENCY = 0.3  # v16.29: 静止时也有30%感知
    INVISIBLE_SENSING_BOOST = 2.0        # 隐身能量需要更多运动来感知
    
    # v16.6 认知重构：只有隐身捕食才配获得10x奖励！
    # 伏地魔策略的核心漏洞：可见能量也给10x
    # 现在修复：可见=1x，隐身=10x (只有主动出击才能赢)
    COGNITIVE_PREMIUM_ONLY_INVISIBLE = True  # 只有隐身阶段有10x奖励
    
    # v16.30 能量收支平衡修复
    VISIBLE_REWARD_MULTIPLIER = 1.0    # v16.31: 基础1x (复杂度有额外加成)
    INVISIBLE_REWARD_MULTIPLIER = 2.0  # v16.31: 基础2x (复杂度有额外加成)
    VISIBLE_RATIO = 0.80               # 80%可见 (更温和的欺骗性景观)

    # v17.2: 降低繁殖阈值促进种群增长
    REPRODUCTION_THRESHOLD = 60.0  # 更容易繁殖

    # ============================================================
    # v17.5: 冰河世纪协议 (Ice Age Protocol) - 课程学习
    # ============================================================
    # 阶段1 (0-ICE_AGE_START_STEP): 寒武纪大爆发 - 低压长脑子
    # 阶段2 (ICE_AGE_START_STEP+): 冰河世纪 - 高压筛选
    ICE_AGE_ENABLED = False            # 是否启用冰河世纪协议
    ICE_AGE_START_STEP = 2000          # 冰河世纪开始步数
    
    # 能量源动态 (阶段2)
    ENERGY_DYNAMIC_ENABLED = False     # 能量源是否移动
    ENERGY_MOVE_INTERVAL = 3           # 能量源移动间隔(步)
    ENERGY_JUMP_PROB = 0.4             # 跳跃概率
    ENERGY_JUMP_DIST = 15.0            # 跳跃距离
    
    # KIF风暴 (阶段2) - 移动的高阻抗区域
    KIF_STORM_ENABLED = False          # 是否启用KIF风暴
    KIF_STORM_COUNT = 5                # 风暴数量
    KIF_STORM_INTENSITY = 800.0        # 风暴强度
    KIF_STORM_MOVE_SPEED = 1.0         # 风暴移动速度
    KIF_STORM_RADIUS = 15.0            # 风暴半径

    # 红皇后捕食参数 (Red Queen Hypothesis) - 黑暗森林
    PREDATION_ENABLED = True        # 启用同类捕食
    PREDATION_RANGE = 4.0           # 捕食范围
    PREDATION_RATE = 0.8            # 吸血效率
    PREDATION_COST = 0.05           # 捕食代谢成本
    PREDATION_MUTATION = 0.15       # 产生捕食者突变概率
    ATTACK_RADIUS = 3.0             # 攻击半径
    ATTACK_THRESHOLD = 0.5          # 攻击阈值
    STRIKE_COST = 2.0               # 爆发攻击成本

    # v16.23 Stigmergy Field (信息素场)
    # 通过manifest配置或PoolConfig启用
    STIGMERGY_ENABLED = True         # 启用信息素场
    STIGMERGY_DEPOSIT_AMOUNT = 0.02  # 移动时沉积量
    STIGMERGY_ALARM_AMOUNT = 0.5     # 捕食时警报量
    # 注意: 可通过manifest.stigmergy_field_enabled覆盖

    # 代谢衰老参数 (熵增定律)
    AGE_ENABLED = True              # 启用年龄惩罚
    AGE_ALPHA = 0.00001             # 衰老系数: cost * (1 + alpha * age^2)

    # 演化棘轮 (Supernode Ratchet)
    SUPERNODE_ENABLED = True        # 启用超级节点
    SUPERNODE_METABOLIC_BONUS = 0.5 # 超级节点代谢折扣 (1.5节点算0.5!)
    SUPERNODE_DETECTION_FREQUENCY = 100  # v16.16 修复: 提高检测频率 (方案3)
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
    CAMBRIAN_DELAY_PROB = 0.25       # 混入DELAY节点概率
    CAMBRIAN_MULTIPLY_PROB = 0.25    # 混入MULTIPLY节点概率
    CAMBRIAN_MODULATOR_PROB = 0.10   # v17.0: 混入MODULATOR节点概率

    # ================================================================
    # v14.1 静默拓扑突变 (新节点权重初始化)
    # ================================================================
    SILENT_MUTATION = True           # 启用静默突变
    SILENT_WEIGHT = 0.001            # 新连接初始权重 (极小)

    # ================================================================
    # Net-2-Net 无损结构变异 (实验性)
    # ================================================================
    # 目标：在增加新节点时保持行为完全一致
    # 对照组: ADD_NODE_OUTPUT_WEIGHT=1.0 (当前方案)
    # 实验组: ADD_NODE_OUTPUT_WEIGHT=0.0 (零权重方案)
    # 回退方式: 只需将 ADD_NODE_OUTPUT_WEIGHT 改为 1.0
    ADD_NODE_ZERO_WEIGHT = False     # [已废弃: 保留用于兼容]
    ADD_NODE_OUTPUT_WEIGHT = 1.0     # 新节点输出权重 (1.0=当前方案, 0.0=无损)

    # ================================================================
    # v14.1 代谢宽限期 (新拓扑折扣)
    # ================================================================
    METABOLIC_GRACE = True           # 启用代谢宽限期
    METABOLIC_GRACE_STEPS = 100      # 宽限期步数
    METABOLIC_GRACE_DISCOUNT = 0.5   # 折扣: 前100步只付50%代谢

    # ================================================================
    # v17.1 阶段一修复: Net-2-Net 零权重死亡之谷
    # ================================================================
    # 1. 代谢适应期: 已由 METABOLIC_GRACE 提供 (100步50%折扣)
    
    # 2. 带噪恒等初始化 (Noisy Identity Init)
    # 在Net-2-Net零初始化基础上注入微小噪声，打破对称性
    NOISY_IDENTITY_INIT = True       # 启用带噪恒等初始化
    NOISY_IDENTITY_SIGMA = 0.1       # 噪声标准差 N(0, 0.1)
    
    # 3. MODULATOR预偏置 (Pre-bias)
    # 初始偏置+2.0使sigmoid(2.0)≈0.88，让新节点默认"开启"
    MODULATOR_BIAS = 2.0             # MODULATOR初始偏置

    # ================================================================
    # v17.2 Phase 2: 软承载力与拥挤度惩罚 (生态稳态)
    # ================================================================
    # 1. 软承载力 (Soft Carrying Capacity)
    # v17.2: 降低预算增加压力
    SOFT_CARRYING_CAP = True         # 启用软承载力
    GLOBAL_ENERGY_BUDGET = 3000.0    # 紧张预算促进竞争
    
    # 2. 拥挤度惩罚 (Crowding Penalty)
    # v17.2: 激进配置促进空间分散
    CROWDING_PENALTY_ENABLED = True  # 启用拥挤惩罚
    CROWDING_RADIUS = 8.0            # 较小半径，更敏感的密度检测
    CROWDING_DECAY_EXPONENT = 1.0    # 线性衰减，更强效果
    CROWDING_MIN_FACTOR = 0.1        # 最低10%，强制分散

    # ================================================================
    # v16.33 认知溢价: 平滑代谢曲线 (消除断崖)
    # ================================================================
    NONLINEAR_METABOLISM = True      # 启用非线性代谢
    LOG_BASE = 1.5                   # 对数底 (保留用于对数方案)
    # v16.33: 移除FREE_NODES，使用平滑sigmoid成本
    # 旧方案: FREE_NODES=8 导致8→9节点断崖
    # 新方案: 所有节点微小成本 + sigmoid平滑增长
    SPARSE_ACTIVATION = True         # 稀疏激活 (休眠节点不耗能)
    SPARSE_THRESHOLD = 0.01          # 发放率低于此值视为休眠
    
    # v16.33 平滑代谢参数 (进一步降低以突破8节点)
    METABOLISM_SIGMOID = True        # 启用sigmoid平滑成本
    METABOLISM_SLOPE = 0.12          # sigmoid斜率 (更低=更平滑)
    METABOLISM_MIDPOINT = 12         # sigmoid中点移到12节点
    METABOLISM_MIN_COST = 0.005      # 极小初始成本 (1-5节点几乎免费)
    METABOLISM_MAX_COST = 0.15       # 最大成本 (15节点)
    
    # ================================================================
    # v17.0 代谢成本阶梯 (Metabolic Cost Ladder)
    # ================================================================
    # 静态成本: 节点存在就需要支付
    # 动态成本: 节点激活时额外支付 (使用成本)
    COST_MULTIPLY_ACTIVATION = 0.02  # MULTIPLY 动态激活系数
    COST_MODULATOR_ACTIVATION = 0.08 # MODULATOR 动态激活系数

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
    
    # v16.33 增强红皇后参数
    RED_QUEEN_PREDATION_RATE = 1.5   # 捕食效率 (0.8 → 1.5)
    RED_QUEEN_N_RIVALS = 10          # 敌对Agent数量 (3 → 10)
    RED_QUEEN_REFRESH_INTERVAL = 20  # 刷新间隔 (40 → 20)
    RED_QUEEN_ATTACK_RADIUS = 5.0    # 攻击半径 (3.0 → 5.0)

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


    # ================================================================
    # v17.3 动态能量源 (追猎动力学) - MVP
    # ================================================================
    # 开关: True=启用动态能量源, False=静态能量源(回退)
    USE_DYNAMIC_ENERGY_SOURCE: bool = False
    
    # 移动参数 (仅当USE_DYNAMIC_ENERGY_SOURCE=True时生效)
    ENERGY_MOVE_INTERVAL: int = 10      # 每N步尝试移动
    ENERGY_JUMP_PROB: float = 0.2       # 20%概率跳跃
    ENERGY_JUMP_DIST: float = 5.0       # 跳跃距离(格)
    ENERGY_WIGGLE_RADIUS: float = 1.0   # 静止时蠕动半径
    
    # 课程学习: 按step启用动态 (0=立即, >0=延迟启用)
    ENERGY_DYNAMIC_START_STEP: int = 0  # 第N步后启用动态移动


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
    
    # ... 其他属性方法保持不变，引用原始文件