"""
BenchmarkRunner 模块测试
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import numpy as np
from benchmarks.benchmark_runner import (
    BenchmarkRunner, BenchmarkTask, TaskFactory, 
    MetricsCalculator
)


def test_metrics():
    """测试轨迹熵计算"""
    print("\n=== Test: Metrics Calculator ===")
    
    # 直线轨迹 (低熵)
    straight = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    entropy_straight = MetricsCalculator.trajectory_entropy(straight)
    print(f"Straight trajectory entropy: {entropy_straight:.3f}")
    
    # 随机轨迹 (高熵)
    np.random.seed(42)
    random_trajectory = [(np.random.rand() * 100, np.random.rand() * 100) for _ in range(50)]
    entropy_random = MetricsCalculator.trajectory_entropy(random_trajectory)
    print(f"Random trajectory entropy: {entropy_random:.3f}")
    
    # 路径效率
    efficiency = MetricsCalculator.path_efficiency(
        [(0, 0), (3, 4), (5, 5)],  # 绕路
        (0, 0), (5, 5)
    )
    print(f"Path efficiency (detour): {efficiency:.3f}")
    
    # 直线效率 = 1.0
    efficiency_direct = MetricsCalculator.path_efficiency(
        [(0, 0), (5, 5)],
        (0, 0), (5, 5)
    )
    print(f"Path efficiency (direct): {efficiency_direct:.3f}")
    
    assert entropy_straight < entropy_random, "Straight should have lower entropy"
    assert 0 <= efficiency <= 1, "Efficiency should be in [0, 1]"
    print("✓ Metrics tests passed")


def test_task_factory():
    """测试任务工厂配置合并"""
    print("\n=== Test: Task Factory ===")
    
    # 测试配置合并
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 100}, "e": 4}
    
    result = TaskFactory._merge_config(base, override)
    print(f"Merged config: {result}")
    
    assert result["a"] == 1
    assert result["b"]["c"] == 100  # 被覆盖
    assert result["b"]["d"] == 3    # 保留
    assert result["e"] == 4         # 新增
    
    print("✓ Config merge test passed")


def test_benchmark_tasks():
    """测试Benchmark任务定义"""
    print("\n=== Test: Benchmark Tasks ===")
    
    runner = BenchmarkRunner(verbose=False)
    
    # 打印可用任务
    print("Available tasks:")
    for name, task in runner.TASK_TEMPLATES.items():
        episodic_tag = " [EPISODIC]" if task.episodic else ""
        print(f"  - {name}: Level {task.level}, {task.max_steps} steps{episodic_tag}")
    
    # 验证Level分布
    levels = set(task.level for task in runner.TASK_TEMPLATES.values())
    print(f"\nCovered levels: {sorted(levels)}")
    
    # 验证Level 3是episodic
    l3_task = runner.TASK_TEMPLATES["t_maze_stigmergy"]
    assert l3_task.level == 3
    assert l3_task.episodic == True
    assert l3_task.num_episodes == 2
    
    print("✓ Benchmark tasks test passed")


def test_benchmark_result():
    """测试结果计算"""
    print("\n=== Test: Benchmark Result ===")
    
    from benchmarks.benchmark_runner import BenchmarkResult
    
    result = BenchmarkResult(
        task_name="test",
        level=1,
        success=True,
        steps_taken=50,
        final_position=(90, 50),
        trajectory=[(10, 50), (50, 50), (90, 50)],
        trajectory_entropy=2.5,
        path_efficiency=0.95,
        success_reward=100.0,
        step_penalty=-0.1
    )
    
    print(f"Fitness: {result.fitness:.2f}")
    assert result.fitness > 0, "Success should have positive fitness"
    
    # 失败结果
    fail_result = BenchmarkResult(
        task_name="test",
        level=1,
        success=False,
        steps_taken=200,
        final_position=(50, 50),
        trajectory=[(10, 50), (20, 50)],
        trajectory_entropy=1.0,
        path_efficiency=0.5,
        success_reward=100.0,
        step_penalty=-0.1
    )
    print(f"Failed fitness: {fail_result.fitness:.2f}")
    assert fail_result.fitness < 0, "Failure should have negative fitness"
    
    print("✓ Benchmark result test passed")


if __name__ == "__main__":
    test_metrics()
    test_task_factory()
    test_benchmark_tasks()
    test_benchmark_result()
    print("\n=== All tests completed ===")
    print("\n📊 BenchmarkRunner Phase 1 核心模块创建完成!")
    print("   - benchmarks/")
    print("     - __init__.py")
    print("     - benchmark_runner.py")
    print("     - test_benchmark_runner.py")
    print("\n下一步: 与真实Agent集成测试")