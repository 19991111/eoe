"""
验证系统运行器
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import numpy as np
from typing import List, Optional, Callable
from core import Population
from .evaluator import BenchmarkResult, Difficulty, Evaluator


class Benchmark:
    """验证系统运行器"""
    
    def __init__(
        self,
        population_size: int = 20,
        champion_path: Optional[str] = None,
        seed: int = 42
    ):
        self.population_size = population_size
        self.champion_path = champion_path
        self.seed = seed
        self.results: List[BenchmarkResult] = []
    
    def run_level(
        self,
        difficulty: Difficulty,
        max_generations: int = 200,
        verbose: bool = True
    ) -> BenchmarkResult:
        """运行单个难度等级"""
        if verbose:
            print(f"\n{'='*50}")
            print(f"测试 L{difficulty.value}: {difficulty.name} ({difficulty.description})")
            print(f"{'='*50}")
        
        # 创建环境配置
        pop = self._create_population(difficulty)
        
        best_fitness = 0.0
        best_agent = None
        success = False
        gen_needed = max_generations
        
        for gen in range(max_generations):
            pop.epoch(verbose=False)
            
            # 计算适应度
            for agent in pop.agents:
                self._calc_fitness(agent, difficulty)
            
            # 找最佳
            best = max(pop.agents, key=lambda a: a.fitness)
            if best.fitness > best_fitness:
                best_fitness = best.fitness
                best_agent = best
            
            # 检查是否成功
            if Evaluator.evaluate(BenchmarkResult(
                env_name=difficulty.name,
                difficulty=difficulty.value,
                success=False,
                fitness=best.fitness,
                food_collected=best.food_eaten,
                survival_time=best.steps_alive,
                nodes=len(best.genome.nodes),
                meta_nodes=sum(1 for n in best.genome.nodes.values() if n.node_type.name == 'META_NODE'),
                generations_needed=gen
            ), difficulty):
                success = True
                gen_needed = gen
                if verbose:
                    print(f"  ✓ 成功! Gen {gen}, fitness={int(best.fitness)}")
                break
            
            pop.reproduce(verbose=False)
            
            if verbose and gen % 20 == 0:
                print(f"  Gen {gen}: best={int(best.fitness)}, food={best.food_eaten}")
        
        # 构建结果
        result = BenchmarkResult(
            env_name=difficulty.name,
            difficulty=difficulty.value,
            success=success,
            fitness=best_fitness,
            food_collected=best_agent.food_eaten if best_agent else 0,
            survival_time=best_agent.steps_alive if best_agent else 0,
            nodes=len(best_agent.genome.nodes) if best_agent else 0,
            meta_nodes=sum(1 for n in best_agent.genome.nodes.values() if n.node_type.name == 'META_NODE') if best_agent else 0,
            generations_needed=gen_needed,
            max_fitness=best_fitness
        )
        
        if verbose and not success:
            print(f"  ✗ 失败! 最高适应度={int(best_fitness)}")
        
        return result
    
    def _create_population(self, difficulty: Difficulty) -> Population:
        """根据难度创建Population"""
        
        # 基础配置
        kwargs = {
            'population_size': self.population_size,
            'elite_ratio': 0.20,
            'lifespan': 80,
            'use_champion': True if self.champion_path else False,
            'n_food': 5,
            'food_energy': 40,
        }
        
        # 根据难度调整
        if difficulty == Difficulty.L1_SIMPLE:
            # L1: 简单环境,更多食物
            kwargs.update({
                'lifespan': 50,
                'n_food': 8,
                'food_energy': 50
            })
        
        elif difficulty == Difficulty.L2_MULTI:
            # L2: 多食物源
            kwargs.update({
                'lifespan': 80,
                'n_food': 10,
                'food_energy': 40
            })
        
        elif difficulty == Difficulty.L3_OBSTACLE:
            # L3: 有障碍
            kwargs.update({
                'lifespan': 100,
                'n_food': 6,
                'food_energy': 50,
                'n_walls': 3
            })
        
        elif difficulty == Difficulty.L4_HOARDING:
            # L4: 需要归巢贮粮
            kwargs.update({
                'lifespan': 100,
                'n_food': 8,
                'food_energy': 50,
                'seasonal_cycle': True,
                'season_length': 30,
                'winter_food_multiplier': 0.0,
                'winter_metabolic_multiplier': 1.5
            })
        
        elif difficulty == Difficulty.L5_COMPETE:
            # L5: 竞争环境 (红皇后)
            kwargs.update({
                'lifespan': 100,
                'n_food': 6,
                'food_energy': 50,
                'red_queen': True,
                'n_rivals': 3,
                'rival_refresh_interval': 25
            })
        
        elif difficulty == Difficulty.L6_SEASONAL:
            # L6: 季节循环 + 红皇后 (双重压力)
            kwargs.update({
                'lifespan': 120,
                'n_food': 8,
                'food_energy': 50,
                'seasonal_cycle': True,
                'season_length': 35,
                'winter_food_multiplier': 0.0,
                'winter_metabolic_multiplier': 1.3,
                'red_queen': True,
                'n_rivals': 4,
                'rival_refresh_interval': 30
            })
        
        elif difficulty == Difficulty.L7_DYNAMIC:
            # L7: 动态目标 (需要预测)
            kwargs.update({
                'lifespan': 100,
                'n_food': 5,
                'food_energy': 60,
                # 动态食物需要特殊处理
            })
        
        pop = Population(**kwargs)
        
        # 调整传感器范围
        pop.environment.sensor_range = 50
        
        # L4 和 L6 需要巢穴
        if difficulty in [Difficulty.L4_HOARDING, Difficulty.L6_SEASONAL]:
            pop.environment.nest_enabled = True
            pop.environment.nest_position = (
                pop.environment.width * 0.15,
                pop.environment.height * 0.15
            )
        
        return pop
    
    def _calc_fitness(self, agent, difficulty: Difficulty):
        """计算适应度"""
        food = agent.food_eaten
        
        # L4/L6 需要考虑贮粮
        if difficulty in [Difficulty.L4_HOARDING, Difficulty.L6_SEASONAL]:
            food += agent.food_carried + agent.food_stored
        
        # L5/L6 需要考虑存活时间
        survival_bonus = agent.steps_alive * 0.5
        
        # 代谢惩罚
        n_nodes = len(agent.genome.nodes)
        n_edges = len([e for e in agent.genome.edges if e['enabled']])
        metabolic = n_nodes * 0.05 + n_edges * 0.01
        
        agent.fitness = food * 100 + survival_bonus - metabolic
    
    def run_full_suite(
        self,
        max_generations: int = 200,
        start_level: int = 1,
        end_level: int = 7,
        verbose: bool = True
    ) -> List[BenchmarkResult]:
        """运行完整验证套件"""
        self.results = []
        
        if verbose:
            print("\n" + "="*60)
            print(" EOE 智能体能力验证系统 v1.0")
            print("="*60)
        
        for level in range(start_level, end_level + 1):
            difficulty = Difficulty(level)
            result = self.run_level(difficulty, max_generations, verbose)
            self.results.append(result)
        
        # 打印报告
        Evaluator.print_report(self.results)
        
        return self.results
    
    def save_results(self, path: str = "benchmark/results.json"):
        """保存结果"""
        Evaluator.save_results(self.results, path)