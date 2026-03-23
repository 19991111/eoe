"""
实验版配置 (Experiment Preset)
==============================
用于新功能测试的参数组合

使用方式:
    from configs.presets.experiment import ExperimentConfig
    config = ExperimentConfig()
"""

from configs.base import PoolConfig


class ExperimentConfig(PoolConfig):
    """实验版配置 - 启用新功能"""
    
    # ========== v17.2 Phase 1 + Phase 2 ==========
    # Net-2-Net 修复
    NOISY_IDENTITY_INIT = True
    NOISY_IDENTITY_SIGMA = 0.1
    MODULATOR_BIAS = 2.0
    
    # 软承载力 + 拥挤度
    SOFT_CARRYING_CAP = True
    GLOBAL_ENERGY_BUDGET = 3000.0
    CROWDING_PENALTY_ENABLED = True
    CROWDING_RADIUS = 8.0
    CROWDING_DECAY_EXPONENT = 1.0
    CROWDING_MIN_FACTOR = 0.1
    
    # ========== 代谢 (平衡) ==========
    NONLINEAR_METABOLISM = True       # 启用sigmoid
    METABOLISM_SIGMOID = True
    METABOLISM_SLOPE = 0.12
    METABOLISM_MIDPOINT = 12
    BASE_METABOLISM = 0.008
    
    # ========== 繁殖 ==========
    REPRODUCTION_THRESHOLD = 60.0
    
    # ========== 演化 ==========
    MUTATION_RATE = 0.5
    
    # ========== 启用所有机制 ==========
    HEBBIAN_ENABLED = True
    SUPERNODE_ENABLED = True
    CAMBRIAN_INIT = True
    METABOLIC_GRACE = True
    
    # ========== 环境 ==========
    SEASONS_ENABLED = True
    
    # ========== 捕食 ==========
    PREDATION_ENABLED = True


class V17Phase1Config(ExperimentConfig):
    """v17.1 Phase 1 配置 - Net-2-Net 修复"""
    
    # 仅启用 Phase 1
    SOFT_CARRYING_CAP = False
    CROWDING_PENALTY_ENABLED = False
    
    # 保持稳定
    NONLINEAR_METABOLISM = False
    AGE_ENABLED = False


class V17Phase2Config(ExperimentConfig):
    """v17.2 Phase 2 配置 - 生态稳态"""
    
    # Phase 1 已启用
    NOISY_IDENTITY_INIT = True
    MODULATOR_BIAS = 2.0
    
    # Phase 2 启用
    SOFT_CARRYING_CAP = True
    GLOBAL_ENERGY_BUDGET = 3000.0
    CROWDING_PENALTY_ENABLED = True
    CROWDING_RADIUS = 8.0
    CROWDING_DECAY_EXPONENT = 1.0
    CROWDING_MIN_FACTOR = 0.1


class DeceptiveLandscapeConfig(ExperimentConfig):
    """欺骗性景观配置 - 课程学习"""
    
    # 欺骗性景观
    VISIBLE_RATIO = 0.80             # 阶段I: 80%可见
    VISIBLE_REWARD_MULTIPLIER = 1.0
    INVISIBLE_REWARD_MULTIPLIER = 2.0
    
    # 课程学习 (需要在代码中动态调整 VISIBLE_RATIO)
    # 阶段I (0-1000): 80% 可见
    # 阶段II (1000-2000): 50% 可见
    # 阶段III (2000+): 30% 可见
    
    # 高认知溢价
    COGNITIVE_PREMIUM_MULTIPLIER = 10.0
    COGNITIVE_PREMIUM_ONLY_INVISIBLE = True
    
    # 主动感知
    ACTIVE_SENSING_ENABLED = True
    INVISIBLE_SENSING_BOOST = 2.0


class DifferentiableConfig(ExperimentConfig):
    """可微演化配置 - 启用梯度学习"""
    
    # 可微大脑
    DIFFERENTIABLE_BRAIN = True
    DIFFERENTIABLE_USE_PYG = True
    DIFFERENTIABLE_LR = 0.001
    DIFFERENTIABLE_UPDATE_INTERVAL = 10
    DIFFERENTIABLE_MIN_STEPS = 5
    DIFFERENTIABLE_MAX_BUFFER = 50
    
    # 鲍德温同化
    BALDWIN_ASSIMILATION_KAPPA = 0.5
    BALDWIN_EXPLORATION_SIGMA = 0.01
    
    # 预测损失
    PREDICTION_LOSS_WEIGHT = 0.1
    ENERGY_LR_MODULATOR = True


class BaldwinLearningConfig(ExperimentConfig):
    """
    鲍德温效应配置 - 一生学习 + 拓扑变异
    
    核心思想:
    - 演化只遗传拓扑结构，不遗传权重
    - 个体在一生中通过Hebbian学习调整权重
    - 演化压力会选择"可学习性(Learnability)"强的拓扑
    
    目标:
    - 解决前馈断层问题 (FEEDFORWARD_DISCONNECT)
    - 涌现出sensor→actuator直连的结构
    """
    
    # ========== 一生学习 (Hebbian) ==========
    HEBBIAN_ENABLED = True
    HEBBIAN_LEARNING_RATE = 0.05  # 较高学习率加速学习
    
    # ========== 权重不遗传 (关键!) ==========
    # 禁用权重继承，新个体使用随机权重
    INHERIT_WEIGHTS = False
    
    # ========== 拓扑变异 (结构遗传) ==========
    MUTATION_RATE = 0.8           # 高变异率加速结构演化
    ADD_NODE_PROB = 0.3           # 30%概率添加节点
    REMOVE_NODE_PROB = 0.05       # 5%概率移除节点
    
    # ========== 强制移动 (增加学习压力) ==========
    # 让不动无法存活，迫使agent学习运动
    BASE_METABOLISM = 0.015       # 稍高代谢
    MOVEMENT_PENALTY = 0.0        # 不惩罚移动
    
    # 能量源设置
    FOOD_VALUE = 30.0             # 高价值奖励
    FOOD_RESPAWN_RATE = 10        # 快速刷新
    
    # ========== 环境复杂度 ==========
    N_FOOD = 15                   # 更多能量源
    SENSOR_RANGE = 20             # 感知范围
    
    # ========== 课程学习 (可选) ==========
    # 可以动态调整难度
    ENABLE_CURRICULUM = False     # 先关闭简化测试