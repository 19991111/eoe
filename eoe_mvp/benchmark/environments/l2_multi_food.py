"""
L2: 多食物源环境
目标选择能力测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class MultiFoodEnv:
    """多食物源 - 需要选择目标"""
    
    DIFFICULTY = 2
    NAME = "多食物源"
    DESCRIPTION = "测试目标选择能力"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        return Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=80,
            use_champion=kwargs.get('use_champion', True),
            n_food=10,
            food_energy=40,
            n_walls=0,
            day_night_cycle=False
        )
    
    @staticmethod
    def success_criteria(agent) -> bool:
        return agent.food_eaten >= 3