"""
L5: 竞争环境
对抗智能测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class CompetitionEnv:
    """竞争环境 - 红皇后假说"""
    
    DIFFICULTY = 5
    NAME = "竞争环境"
    DESCRIPTION = "测试对抗智能"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        return Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=100,
            use_champion=kwargs.get('use_champion', True),
            n_food=6,
            food_energy=50,
            red_queen=True,
            n_rivals=3,
            rival_refresh_interval=25
        )
    
    @staticmethod
    def success_criteria(agent) -> bool:
        # 在有敌对的情况下仍能存活并获取食物
        return agent.food_eaten >= 2 and agent.steps_alive >= 50