"""
L6: 季节循环环境
规划未来能力测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class SeasonalEnv:
    """季节循环 - 需要规划未来"""
    
    DIFFICULTY = 6
    NAME = "季节循环"
    DESCRIPTION = "测试规划未来能力"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        pop = Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=120,
            use_champion=kwargs.get('use_champion', True),
            n_food=8,
            food_energy=50,
            seasonal_cycle=True,
            season_length=35,
            winter_food_multiplier=0.0,
            winter_metabolic_multiplier=1.3,
            red_queen=True,
            n_rivals=4,
            rival_refresh_interval=30
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
        # 冬天前贮粮,冬天存活
        stored = agent.food_carried + agent.food_stored
        return stored >= 2 and agent.food_eaten >= 3 and agent.steps_alive >= 60