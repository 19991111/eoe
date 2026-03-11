"""
EOE 智能体能力验证系统
======================

分层难度环境,验证智能体在不同任务上的表现。

等级:
- L1: 简单觅食 (基础感知-运动)
- L2: 多食物源 (目标选择)
- L3: 障碍躲避 (空间推理)
- L4: 归巢贮粮 (长期记忆)
- L5: 竞争环境 (对抗智能)
- L6: 季节循环 (规划未来)
- L7: 动态目标 (预测能力)

作者: 104助手
日期: 2026-03-10
"""

from .runner import Benchmark
from .evaluator import BenchmarkResult, Difficulty

__all__ = ['Benchmark', 'BenchmarkResult', 'Difficulty']