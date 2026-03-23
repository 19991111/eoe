"""
完整基准测试 - 5个Level全部测试
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

from core.eoe.agent import Agent
from benchmarks.benchmark_runner import BenchmarkRunner, MetricsCalculator


def test_all_levels():
    """测试所有5个Level"""
    print("="*60)
    print("完整基准测试 - Level 1-5")
    print("="*60)
    
    # 创建Agent (使用真实genome)
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    brain = agent.genome
    
    runner = BenchmarkRunner(verbose=True)
    
    results = {}
    
    for level in range(1, 6):
        print(f"\n{'='*60}")
        print(f"Testing Level {level}")
        print(f"{'='*60}")
        
        try:
            level_results = runner.run_level(level, brain, start_pos=(10.0, 50.0))
            results[level] = level_results
            
            for r in level_results:
                status = "✓" if r.success else "✗"
                print(f"  {status} {r.task_name}: {r.steps_taken} steps, fitness={r.fitness:.2f}")
                
        except Exception as e:
            print(f"  Error: {e}")
            results[level] = []
    
    # 汇总报告
    print(f"\n{'='*60}")
    print("测试汇总")
    print(f"{'='*60}")
    
    report = runner.generate_report()
    print(report)
    
    # 计算统计数据
    total_tasks = sum(len(v) for v in results.values())
    total_success = sum(sum(1 for r in v if r.success) for v in results.values())
    
    print(f"\n总任务数: {total_tasks}")
    print(f"成功: {total_success}")
    print(f"成功率: {total_success/total_tasks*100:.1f}%")
    
    return results


def test_specific_level(level: int):
    """测试指定Level"""
    print(f"\n{'='*60}")
    print(f"Testing Level {level}")
    print(f"{'='*60}")
    
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    brain = agent.genome
    
    runner = BenchmarkRunner(verbose=True)
    results = runner.run_level(level, brain, start_pos=(10.0, 50.0))
    
    print(f"\nLevel {level} 结果:")
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.task_name}")
        print(f"      Steps: {r.steps_taken}, Fitness: {r.fitness:.2f}")
        print(f"      Trajectory: {len(r.trajectory)} points")
        print(f"      Entropy: {r.trajectory_entropy:.2f}, Efficiency: {r.path_efficiency:.2f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        level = int(sys.argv[1])
        test_specific_level(level)
    else:
        test_all_levels()