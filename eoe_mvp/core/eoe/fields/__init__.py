"""
物理场模块
==========
统一场物理系统的实现

Classes:
    Field: 场抽象基类
    SourceField: 带能量源的场
    DiffusiveField: 带扩散的场
    StaticField: 静态场
    
    EnergyField: EPF 能量场
    ImpedanceField: KIF 阻抗场
    StigmergyField: ISF 压痕场
"""

from .base import Field, SourceField, DiffusiveField, StaticField
from .energy import EnergyField
from .impedance import ImpedanceField
from .stigmergy import StigmergyField

# 默认场配置
DEFAULT_FIELD_CONFIG = {
    'energy': {
        'n_sources': 3,
        'source_strength': 50.0,
        'decay_rate': 0.99
    },
    'impedance': {
        'noise_scale': 1.0,
        'obstacle_density': 0.15
    },
    'stigmergy': {
        'diffusion_rate': 0.1,
        'decay_rate': 0.98
    }
}

__all__ = [
    'Field',
    'SourceField', 
    'DiffusiveField',
    'StaticField',
    'EnergyField',
    'ImpedanceField', 
    'StigmergyField',
    'DEFAULT_FIELD_CONFIG'
]