"""
评估器 - 定义难度等级和结果结构
"""

import json
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Optional, List


class Difficulty(IntEnum):
    """难度等级"""
    L1_SIMPLE = 1    # 简单觅食
    L2_MULTI = 2     # 多食物源
    L3_OBSTACLE = 3  # 障碍躲避
    L4_HOARDING = 4  # 归巢贮粮
    L5_COMPETE = 5   # 竞争环境
    L6_SEASONAL = 6  # 季节循环
    L7_DYNAMIC = 7   # 动态目标
    
    @property
    def name(self) -> str:
        names = {
            1: "简单觅食",
            2: "多食物源",
            3: "障碍躲避",
            4: "归巢贮粮",
            5: "竞争环境",
            6: "季节循环",
            7: "动态目标"
        }
        return names.get(self, str(self))
    
    @property
    def description(self) -> str:
        descs = {
            1: "基础感知-运动能力",
            2: "目标选择能力",
            3: "空间推理能力",
            4: "长期记忆能力",
            5: "对抗智能",
            6: "规划未来能力",
            7: "预测能力"
        }
        return descs.get(self, "")


@dataclass
class BenchmarkResult:
    """单次验证结果"""
    env_name: str
    difficulty: int
    success: bool
    fitness: float
    food_collected: int
    survival_time: float
    nodes: int
    meta_nodes: int
    generations_needed: int
    max_fitness: float = 0.0
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def grade(self) -> str:
        if self.success:
            return "✓ PASS"
        return "✗ FAIL"
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"L{self.difficulty} {self.env_name:<12} {status} (Gen {self.generations_needed})"


class Evaluator:
    """评估器 - 判定是否通过"""
    
    # 成功标准配置
    CRITERIA = {
        Difficulty.L1_SIMPLE: {
            'min_food': 1,
            'min_fitness': 100,
            'max_gen': 50
        },
        Difficulty.L2_MULTI: {
            'min_food': 3,
            'min_fitness': 300,
            'max_gen': 80
        },
        Difficulty.L3_OBSTACLE: {
            'min_food': 2,
            'min_fitness': 200,
            'max_gen': 100
        },
        Difficulty.L4_HOARDING: {
            'min_food': 3,
            'min_stored': 1,
            'min_fitness': 500,
            'max_gen': 150
        },
        Difficulty.L5_COMPETE: {
            'min_food': 2,
            'min_survival': 50,
            'min_fitness': 200,
            'max_gen': 150
        },
        Difficulty.L6_SEASONAL: {
            'min_food': 5,
            'min_stored': 2,
            'min_survival_winter': 20,
            'min_fitness': 800,
            'max_gen': 200
        },
        Difficulty.L7_DYNAMIC: {
            'min_food': 3,
            'min_fitness': 400,
            'max_gen': 200
        }
    }
    
    @classmethod
    def evaluate(cls, result: BenchmarkResult, difficulty: Difficulty) -> bool:
        """判定是否通过"""
        criteria = cls.CRITERIA.get(difficulty, {})
        
        # 简单检查
        if result.food_collected >= criteria.get('min_food', 0):
            return True
        if result.fitness >= criteria.get('min_fitness', 0):
            return True
        
        return False
    
    @classmethod
    def get_success_rate(cls, results: List[BenchmarkResult]) -> float:
        """计算成功率"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)
    
    @classmethod
    def print_report(cls, results: List[BenchmarkResult]):
        """打印验证报告"""
        print()
        print("╔" + "═" * 50 + "╗")
        print("║" + " EOE 智能体能力验证报告 ".center(50) + "║")
        print("╠" + "═" * 50 + "╣")
        
        for r in results:
            status = "✓ PASS" if r.success else "✗ FAIL"
            print(f"║ L{r.difficulty} {r.env_name:<14} {status:<15} (Gen {r.generations_needed:3d}) ║")
        
        print("╠" + "═" * 50 + "╣")
        
        # 统计
        passed = sum(1 for r in results if r.success)
        total = len(results)
        rate = passed / total * 100 if total > 0 else 0
        
        # 涌现统计
        total_meta = sum(r.meta_nodes for r in results)
        total_nodes = sum(r.nodes for r in results)
        
        print(f"║ 总体: {passed}/{total} 通过 ({rate:.0f}%)" + " " * 25 + "║")
        if total_meta > 0:
            print(f"║ 涌现: META节点 {total_meta}个, 平均节点 {total_nodes//total}" + " " * 10 + "║")
        
        # 评级
        rating = cls._get_rating(passed)
        print("╠" + "═" * 50 + "╣")
        print(f"║ 评级: {rating}" + " " * 40 + "║")
        print("╚" + "═" * 50 + "╝")
    
    @classmethod
    def _get_rating(cls, passed: int) -> str:
        """根据通过数量返回评级"""
        ratings = {
            0: "基础智能 (0/7)",
            1: "初级智能 (1/7)",
            2: "初级智能 (2/7)",
            3: "中级智能 (3/7)",
            4: "中级智能 (4/7)",
            5: "高级智能 (5/7)",
            6: "超高级智能 (6/7)",
            7: "AGI 候选 (7/7)"
        }
        return ratings.get(passed, f"{passed}/7")
    
    @classmethod
    def save_results(cls, results: List[BenchmarkResult], path: str = "benchmark/results.json"):
        """保存结果到文件"""
        data = [r.to_dict() for r in results]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n结果已保存到: {path}")