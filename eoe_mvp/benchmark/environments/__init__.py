"""
难度环境集合
"""

from .l1_simple_foraging import SimpleForagingEnv
from .l2_multi_food import MultiFoodEnv
from .l3_obstacle import ObstacleEnv
from .l4_hoarding import HoardingEnv
from .l5_compete import CompetitionEnv
from .l6_seasonal import SeasonalEnv
from .l7_dynamic import DynamicTargetEnv

__all__ = [
    'SimpleForagingEnv',
    'MultiFoodEnv', 
    'ObstacleEnv',
    'HoardingEnv',
    'CompetitionEnv',
    'SeasonalEnv',
    'DynamicTargetEnv'
]