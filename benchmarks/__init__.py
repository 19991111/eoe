"""
EOE 智能体认知能力基准测试套件

Levels:
- Level 1: 基础运动 (T-Maze直线)
- Level 2: 短期记忆 (T-Maze延迟)
- Level 3: 外部存储 (多回合Stigmergy)
- Level 4: 元认知 (性能自我评估)
- Level 5: 组合推理 (多任务迁移)
"""

from .benchmark_runner import (
    BenchmarkRunner,
    BenchmarkTask,
    BenchmarkResult,
    FrozenAgent,
    TaskFactory,
    MetricsCalculator,
    run_benchmark
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkTask", 
    "BenchmarkResult",
    "FrozenAgent",
    "TaskFactory",
    "MetricsCalculator",
    "run_benchmark"
]