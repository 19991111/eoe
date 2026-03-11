"""
L7: 动态目标环境
预测能力测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core'))

from core import Population


class DynamicTargetEnv:
    """动态目标 - 需要预测能力"""
    
    DIFFICULTY = 7
    NAME = "动态目标"
    DESCRIPTION = "测试预测能力"
    
    @staticmethod
    def create_population(population_size: int = 20, **kwargs) -> Population:
        pop = Population(
            population_size=population_size,
            elite_ratio=0.20,
            lifespan=100,
            use_champion=kwargs.get('use_champion', True),
            n_food=5,
            food_energy=60,
            # 动态食物需要特殊处理
        )
        # 启用食物逃逸
        pop.environment.food_escape_enabled = True
        pop.environment.food_escape_speed = 1.2
        pop.environment.food_escape_range = 25
        return pop
    
    @staticmethod
    def success_criteria(agent) -> bool:
        # 追逐移动目标
        return agent.food_eaten >= 3