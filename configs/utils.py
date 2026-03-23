"""
配置工具函数 (Configuration Utilities)
======================================
配置验证、加载、合并等功能

使用方式:
    from configs.utils import validate_config, merge_config
"""

from typing import List, Dict, Any, Optional
import copy


def validate_config(config) -> List[str]:
    """
    验证配置合理性，返回警告列表
    
    Args:
        config: PoolConfig 实例
        
    Returns:
        警告信息列表 (空列表表示通过)
    """
    warnings = []
    
    # ========== 代谢参数 ==========
    if config.NONLINEAR_METABOLISM and config.BASE_METABOLISM > 0.015:
        warnings.append("NONLINEAR_METABOLISM + 高BASE_METABOLISM (>0.015) 可能导致种群崩溃")
    
    if config.AGE_ENABLED and config.AGE_ALPHA > 0.0001:
        warnings.append("AGE_ALPHA 过高 (>0.0001) 会导致老年个体迅速死亡")
    
    # ========== 繁殖参数 ==========
    if config.REPRODUCTION_THRESHOLD < 20:
        warnings.append("REPRODUCTION_THRESHOLD 过低 (<20) 可能导致过度繁殖")
    
    if config.REPRODUCTION_THRESHOLD > 200:
        warnings.append("REPRODUCTION_THRESHOLD 过高 (>200) 可能导致种群无法恢复")
    
    # ========== 生态参数 ==========
    if config.CROWDING_PENALTY_ENABLED:
        if config.CROWDING_RADIUS > 30:
            warnings.append("CROWDING_RADIUS 过大 (>30) 可能导致效果不明显")
        
        if config.CROWDING_RADIUS < 3:
            warnings.append("CROWDING_RADIUS 过小 (<3) 可能导致过度分散")
        
        if config.CROWDING_MIN_FACTOR < 0.05:
            warnings.append("CROWDING_MIN_FACTOR 过低 (<0.05) 可能导致完全无法进食")
    
    if config.SOFT_CARRYING_CAP:
        if config.GLOBAL_ENERGY_BUDGET < 1000:
            warnings.append("GLOBAL_ENERGY_BUDGET 过低 (<1000) 可能导致种群频繁崩溃")
        
        if config.GLOBAL_ENERGY_BUDGET > 20000:
            warnings.append("GLOBAL_ENERGY_BUDGET 过高 (>20000) 可能导致失去人口控制")
    
    # ========== 演化参数 ==========
    if config.MUTATION_RATE > 0.9:
        warnings.append("MUTATION_RATE 过高 (>0.9) 可能导致结构不稳定")
    
    if config.HEBBIAN_ENABLED and config.HEBBIAN_BASE_LR > 0.1:
        warnings.append("HEBBIAN_BASE_LR 过高 (>0.1) 可能导致学习不稳定")
    
    # ========== SuperNode ==========
    if config.SUPERNODE_ENABLED:
        if config.SUPERNODE_DETECTION_FREQUENCY < 50:
            warnings.append("SUPERNODE_DETECTION_FREQUENCY 过低 (<50) 可能影响性能")
        
        if config.SUPERNODE_DETECTION_FREQUENCY > 500:
            warnings.append("SUPERNODE_DETECTION_FREQUENCY 过高 (>500) 可能错过模式")
    
    # ========== 能量收支 ==========
    if config.BASE_METABOLISM > 0.02:
        warnings.append("BASE_METABOLISM 过高 (>0.02) 大概率导致能量入不敷出")
    
    return warnings


def merge_config(base: Any, override: Dict[str, Any]) -> Any:
    """
    合并配置字典到基础配置
    
    Args:
        base: 基础配置对象 (PoolConfig 或其子类)
        override: 要覆盖的参数字典
        
    Returns:
        合并后的配置对象
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if hasattr(result, key):
            setattr(result, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return result


def config_to_dict(config) -> Dict[str, Any]:
    """
    将配置对象转换为字典
    
    Args:
        config: PoolConfig 实例
        
    Returns:
        配置参数字典
    """
    return {k: v for k, v in config.__class__.__dict__.items() 
            if not k.startswith('_') and not callable(v)}


def print_config(config, category: Optional[str] = None):
    """
    打印配置 (可按类别过滤)
    
    Args:
        config: PoolConfig 实例
        category: 可选，按类别过滤 (见 CONFIG_CATEGORIES)
    """
    from configs.utils import CONFIG_CATEGORIES
    
    if category:
        keys = CONFIG_CATEGORIES.get(category, [])
        print(f"\n=== {category} ===")
        for key in keys:
            if hasattr(config, key):
                print(f"  {key}: {getattr(config, key)}")
    else:
        # 打印所有
        for key, value in config_to_dict(config).items():
            print(f"  {key}: {value}")


# 配置类别映射
CONFIG_CATEGORIES = {
    "池与种群": ["MAX_AGENTS", "REPRODUCTION_THRESHOLD", "CHILD_ENERGY_RATIO", 
                "MIN_REPRO_ENERGY", "SPAWN_RADIUS", "MUTATION_RATE"],
    
    "代谢与能量": ["BASE_METABOLISM", "ACTIVATION_COST", "BASAL_COST", "NEURAL_COST",
                 "MOVEMENT_PENALTY", "NONLINEAR_METABOLISM", "METABOLISM_SIGMOID",
                 "AGE_ENABLED", "AGE_ALPHA", "ENERGY_RECIRCULATION_ENABLED"],
    
    "认知与感知": ["COGNITIVE_PREMIUM_MULTIPLIER", "ENABLE_INVISIBLE_SENSING",
                 "ACTIVE_SENSING_ENABLED", "ACTIVE_SENSING_THRESHOLD", 
                 "INVISIBLE_SENSING_BOOST", "VISIBLE_REWARD_MULTIPLIER",
                 "INVISIBLE_REWARD_MULTIPLIER", "VISIBLE_RATIO"],
    
    "演化与变异": ["HEBBIAN_ENABLED", "HEBBIAN_BASE_LR", "SUPERNODE_ENABLED",
                 "SUPERNODE_METABOLIC_BONUS", "CAMBRIAN_INIT", "MUTATION_RATE",
                 "NOISY_IDENTITY_INIT", "MODULATOR_BIAS"],
    
    "生态与调控": ["CROWDING_PENALTY_ENABLED", "CROWDING_RADIUS", "CROWDING_DECAY_EXPONENT",
                 "CROWDING_MIN_FACTOR", "SOFT_CARRYING_CAP", "GLOBAL_ENERGY_BUDGET"],
    
    "环境与任务": ["SEASONS_ENABLED", "SEASON_LENGTH", "T_MAZE_ENABLED",
                 "RESOURCE_CYCLE_ENABLED", "RED_QUEEN_ENABLED"],
    
    "捕食与战斗": ["PREDATION_ENABLED", "PREDATION_RANGE", "PREDATION_RATE",
                 "ATTACK_RADIUS", "STRIKE_COST"],
    
    "建造与物质": ["MATTER_GRID_ENABLED", "CONSTRUCT_ENERGY_COST",
                 "DECONSTRUCT_ENERGY_GAIN"],
    
    "诊断": ["DIAGNOSTICS_ENABLED"],
    
    "特殊机制": ["DIFFERENTIABLE_BRAIN", "PRETRAINED_INIT", "BALDWIN_ASSIMILATION_KAPPA"]
}


__all__ = [
    'validate_config',
    'merge_config', 
    'config_to_dict',
    'print_config',
    'CONFIG_CATEGORIES'
]