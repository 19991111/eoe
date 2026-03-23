"""
稳定版配置 (Stable Preset)
==========================
经过验证的参数组合，适合长期运行

使用方式:
    from configs.presets.stable import StableConfig
    config = StableConfig()
"""

from configs.base import PoolConfig


class StableConfig(PoolConfig):
    """稳定版配置 - 已验证的参数组合"""
    
    # ========== 代谢 (禁用非线性，防崩溃) ==========
    NONLINEAR_METABOLISM = False      # 禁用sigmoid成本曲线
    AGE_ENABLED = False               # 禁用年龄惩罚
    BASE_METABOLISM = 0.005           # 低代谢压力
    BASAL_COST = 0.0                  # 无静息代谢
    NEURAL_COST = 0.0005              # 低神经成本
    
    # ========== 繁殖 (降低阈值促进种群) ==========
    REPRODUCTION_THRESHOLD = 40.0     # 较低阈值
    CHILD_ENERGY_RATIO = 0.5          # 子代获得50%能量
    
    # ========== 演化 (中等突变) ==========
    MUTATION_RATE = 0.3               # 中等突变率
    
    # ========== 生态 (启用拥挤惩罚) ==========
    CROWDING_PENALTY_ENABLED = True   # 启用拥挤惩罚
    CROWDING_RADIUS = 10.0            # 适中半径
    CROWDING_DECAY_EXPONENT = 0.8     # 中等衰减
    CROWDING_MIN_FACTOR = 0.2         # 最低20%
    SOFT_CARRYING_CAP = True          # 启用软承载力
    GLOBAL_ENERGY_BUDGET = 5000.0     # 适中预算
    
    # ========== 认知 (保持默认) ==========
    COGNITIVE_PREMIUM_MULTIPLIER = 10.0
    ENABLE_INVISIBLE_SENSING = True
    ACTIVE_SENSING_ENABLED = True
    
    # ========== Hebbian (启用学习) ==========
    HEBBIAN_ENABLED = True
    HEBBIAN_BASE_LR = 0.01
    
    # ========== SuperNode (启用演化棘轮) ==========
    SUPERNODE_ENABLED = True
    SUPERNODE_DETECTION_FREQUENCY = 100
    
    # ========== 环境 (启用季节) ==========
    SEASONS_ENABLED = True
    SEASON_LENGTH = 2000
    
    # ========== 捕食 (启用) ==========
    PREDATION_ENABLED = True
    
    # ========== 诊断 (启用) ==========
    DIAGNOSTICS_ENABLED = True


class MinimalConfig(StableConfig):
    """最小配置 - 用于调试和基准测试"""
    
    # 关闭所有复杂机制
    HEBBIAN_ENABLED = False
    SUPERNODE_ENABLED = False
    SEASONS_ENABLED = False
    PREDATION_ENABLED = False
    STIGMERGY_ENABLED = False
    ENERGY_RECIRCULATION_ENABLED = False
    
    # 极简代谢
    BASE_METABOLISM = 0.003
    MOVEMENT_PENALTY = 0.0
    
    # 单一能量源
    CROWDING_PENALTY_ENABLED = False
    SOFT_CARRYING_CAP = False


class HighPressureConfig(StableConfig):
    """高压配置 - 强制快速演化"""
    
    # 高突变
    MUTATION_RATE = 0.8
    
    # 高代谢压力
    BASE_METABOLISM = 0.012
    NONLINEAR_METABOLISM = True
    
    # 紧张能量预算
    GLOBAL_ENERGY_BUDGET = 2000.0
    CROWDING_RADIUS = 6.0
    CROWDING_MIN_FACTOR = 0.1
    
    # 降低繁殖阈值
    REPRODUCTION_THRESHOLD = 30.0