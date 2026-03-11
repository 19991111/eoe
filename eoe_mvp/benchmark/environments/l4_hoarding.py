"""
L4: 归巢贮粮环境
长期记忆能力测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class HoardingEnv:
    """归巢贮粮 - 需要长期记忆"""
    
    DIFFICULTY = 4
    NAME = "归巢贮粮"
    DESCRIPTION = "测试长期记忆能力"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        pop = Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=100,
            use_champion=kwargs.get('use_champion', True),
            n_food=8,
            food_energy=50,
            seasonal_cycle=True,
            season_length=30,
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.5
        )
        # 启用巢穴
        pop.environment.nest_enabled = True
        pop.environment.nest_position = (
            pop.environment.width * 0.15,
            pop.environment.height * 0.15
        )
        return pop
    
    @staticmethod
    def success_criteria(agent) -> bool:
        # 需要贮粮
        stored = agent.food_carried + agent.food_stored
        return stored >= 1 and agent.food_eaten >= 2