"""
L1: 简单觅食环境
基础感知-运动能力测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class SimpleForagingEnv:
    """简单觅食 - 最基础的环境"""
    
    DIFFICULTY = 1
    NAME = "简单觅食"
    DESCRIPTION = "测试基础感知-运动能力"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        """创建种群"""
        return Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=50,
            use_champion=kwargs.get('use_champion', True),
            n_food=8,
            food_energy=50,
            n_walls=0,
            day_night_cycle=False
        )
    
    @staticmethod
    def success_criteria(agent) -> bool:
        """成功标准: 吃到1个食物"""
        return agent.food_eaten >= 1