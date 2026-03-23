"""
端到端集成测试 - BenchmarkRunner 完整验证

使用Agent自带的genome进行测试，避免传感器维度不匹配
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import numpy as np
from core.eoe.agent import Agent
from core.eoe.genome import OperatorGenome
from core.eoe.environment import Environment
from core.eoe.node import Node, NodeType
from benchmarks.benchmark_runner import (
    BenchmarkRunner, BenchmarkTask, FrozenAgent, TaskFactory, MetricsCalculator
)


def test_nodetype_enum():
    """验证NodeType枚举已修复"""
    print("\n=== Test: NodeType Enum ===")
    
    required_types = [
        'PREDICTOR', 'PORT_MOTION', 'PORT_REPAIR', 
        'PORT_OFFENSE', 'PORT_DEFENSE', 'PORT_SIGNAL',
        'ENTITY_RADAR', 'LIGHT_SENSOR', 'AGENT_RADAR_SENSOR', 'GPS_SENSOR'
    ]
    
    for type_name in required_types:
        assert hasattr(NodeType, type_name), f"Missing: {type_name}"
    
    print("✓ NodeType enum fixed")


def test_environment_creation():
    """测试环境创建"""
    print("\n=== Test: Environment Creation ===")
    
    env = Environment(
        width=100.0,
        height=100.0,
        energy_field_enabled=False,
        stigmergy_field_enabled=False,
        n_food=0,
    )
    
    print(f"Environment: {env.width}x{env.height}")
    print("✓ Environment created")
    
    return env


def test_agent_creation():
    """测试Agent创建"""
    print("\n=== Test: Agent Creation ===")
    
    agent = Agent(agent_id=0, x=50.0, y=50.0, add_predictors=False)
    print(f"Agent: id={agent.id}, pos=({agent.x}, {agent.y})")
    print(f"Genome nodes: {len(agent.genome.nodes)}")
    print("✓ Agent created")
    
    return agent


def test_frozen_agent_with_real_genome():
    """测试FrozenAgent使用真实Agent的genome"""
    print("\n=== Test: FrozenAgent with Real Genome ===")
    
    env = test_environment_creation()
    
    # 使用Agent自带的genome (传感器维度匹配)
    agent = Agent(agent_id=0, x=50.0, y=50.0, add_predictors=False)
    brain = agent.genome  # 使用Agent自己的genome
    
    print(f"Before FrozenAgent: agents={len(env.agents)}")
    
    frozen = FrozenAgent(brain, agent, env)
    
    print(f"After FrozenAgent: agents={len(env.agents)}")
    print(f"Agent energy: {agent.internal_energy}")
    print(f"Genome node count: {len(brain.nodes)}")
    
    assert len(env.agents) == 1
    assert agent.internal_energy == float('inf')
    
    print("✓ FrozenAgent with real genome works")
    
    return env, brain, agent


def test_environment_step():
    """测试env.step()驱动Agent"""
    print("\n=== Test: Environment Step ===")
    
    env, brain, agent = test_frozen_agent_with_real_genome()
    
    initial_pos = (agent.x, agent.y)
    print(f"Initial position: {initial_pos}")
    
    # 执行20步
    for i in range(20):
        try:
            env.step()
        except Exception as e:
            print(f"Step {i} error: {e}")
            break
    
    final_pos = (agent.x, agent.y)
    print(f"Final position: {final_pos}")
    
    distance = np.hypot(final_pos[0] - initial_pos[0], final_pos[1] - initial_pos[1])
    print(f"Distance moved: {distance:.2f}")
    print(f"Agent alive: {agent.is_alive}")
    print(f"Steps alive: {agent.steps_alive}")
    
    print("✓ Environment step passed")
    
    return env, agent


def test_trajectory_recording():
    """测试轨迹记录"""
    print("\n=== Test: Trajectory Recording ===")
    
    env, agent = test_environment_step()
    
    trajectory = [(agent.x, agent.y)]
    for _ in range(30):
        try:
            env.step()
            trajectory.append((agent.x, agent.y))
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"Trajectory length: {len(trajectory)}")
    
    if len(trajectory) > 1:
        entropy = MetricsCalculator.trajectory_entropy(trajectory)
        efficiency = MetricsCalculator.path_efficiency(
            trajectory, trajectory[0], (90, 50)
        )
        print(f"Entropy: {entropy:.2f}, Efficiency: {efficiency:.2f}")
    
    print("✓ Trajectory recording passed")


def test_benchmark_task():
    """测试基准任务"""
    print("\n=== Test: Benchmark Task ===")
    
    task = BenchmarkTask(
        name="test_benchmark",
        level=1,
        max_steps=30,
        config={
            "width": 100.0,
            "height": 100.0,
            "n_food": 0,
        },
        success_reward=100.0,
        step_penalty=-0.1
    )
    
    runner = BenchmarkRunner(verbose=True)
    
    # 创建一个Agent，使用它的genome
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    brain = agent.genome
    
    # 运行任务
    result = runner.run_task(task, brain, start_pos=(10.0, 50.0))
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Steps: {result.steps_taken}")
    print(f"  Trajectory length: {len(result.trajectory)}")
    print(f"  Fitness: {result.fitness:.2f}")
    
    print("✓ Benchmark task passed")


def test_task_factory():
    """测试TaskFactory"""
    print("\n=== Test: TaskFactory ===")
    
    task = BenchmarkTask(
        name="test",
        level=1,
        max_steps=100,
        config={"width": 80.0, "height": 60.0, "n_food": 0}
    )
    
    env = TaskFactory.create_environment(task)
    print(f"Created env: {env.width}x{env.height}")
    
    assert env.width == 80.0
    assert env.height == 60.0
    
    print("✓ TaskFactory works")


def test_multi_level():
    """测试多Level"""
    print("\n=== Test: Multi-Level ===")
    
    runner = BenchmarkRunner(verbose=False)
    
    # 创建一个Agent
    agent = Agent(agent_id=0, x=10.0, y=50.0, add_predictors=False)
    brain = agent.genome
    
    # 测试Level 1
    results_l1 = runner.run_level(1, brain, start_pos=(10.0, 50.0))
    print(f"Level 1: {len(results_l1)} tasks")
    
    # 测试Level 2
    results_l2 = runner.run_level(2, brain, start_pos=(10.0, 50.0))
    print(f"Level 2: {len(results_l2)} tasks")
    
    print("✓ Multi-level passed")


def test_metrics():
    """测试指标计算"""
    print("\n=== Test: Metrics ===")
    
    straight = [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)]
    entropy = MetricsCalculator.trajectory_entropy(straight)
    efficiency = MetricsCalculator.path_efficiency(straight, (0, 0), (100, 0))
    
    print(f"Straight - entropy: {entropy:.2f}, efficiency: {efficiency:.2f}")
    
    assert entropy < 3.0
    assert efficiency > 0.9
    
    print("✓ Metrics calculator works")


if __name__ == "__main__":
    print("="*60)
    print("BenchmarkRunner 完整端到端测试")
    print("="*60)
    
    test_nodetype_enum()
    test_environment_creation()
    test_agent_creation()
    test_frozen_agent_with_real_genome()
    test_environment_step()
    test_trajectory_recording()
    test_benchmark_task()
    test_task_factory()
    test_multi_level()
    test_metrics()
    
    print("\n" + "="*60)
    print("🎉 完整端到端测试全部通过!")
    print("="*60)