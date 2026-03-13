#!/usr/bin/env python3
"""
v13.0 性能探针脚本 (简化版)
===============
直接测试核心模块性能，避免完整仿真的维度问题
"""

import sys
sys.path.insert(0, '.')

import time
import numpy as np
from collections import defaultdict


# ============================================================================
# 性能计时器
# ============================================================================

class Timer:
    def __init__(self):
        self.times = defaultdict(list)
    
    def record(self, name, duration):
        self.times[name].append(duration)
    
    def report(self):
        print("\n" + "="*70)
        print("📊 v13.0 性能探针报告")
        print("="*70)
        
        total = sum(sum(v) for v in self.times.values())
        print(f"\n总耗时: {total:.4f}s\n")
        
        print(f"{'模块':<45} {'调用次数':>8} {'总耗时':>10} {'占比':>8}")
        print("-"*70)
        
        for name, vals in sorted(self.times.items(), key=lambda x: sum(x[1]), reverse=True):
            t = sum(vals)
            cnt = len(vals)
            pct = t/total*100 if total > 0 else 0
            print(f"{name:<45} {cnt:>8} {t:>9.4f}s {pct:>7.1f}%")


timer = Timer()


# ============================================================================
# 测试 A: 场物理更新
# ============================================================================

def test_field_updates():
    """测试场物理更新性能"""
    from core.eoe.environment import Environment
    
    print("\n" + "="*70)
    print("🧪 测试 A: 场物理更新")
    print("="*70)
    
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True,
        stress_field_enabled=True,
        n_food=0
    )
    
    # 预热
    for _ in range(5):
        env.step()
    
    # 测试
    n_iterations = 100
    
    print(f"\n运行 {n_iterations} 步场更新...")
    start = time.perf_counter()
    
    for i in range(n_iterations):
        step_start = time.perf_counter()
        env.step()
        timer.record("Environment.step()", time.perf_counter() - step_start)
    
    elapsed = time.perf_counter() - start
    
    print(f"总耗时: {elapsed:.4f}s")
    print(f"平均每步: {elapsed/n_iterations*1000:.2f}ms")
    
    # 验证场状态
    print(f"\n场状态:")
    print(f"  EPF: {env.energy_field.field.shape}")
    print(f"  KIF: {env.impedance_field.field.shape}")
    print(f"  ISF: {env.stigmergy_field.field.shape}")
    print(f"  ESF: {env.stress_field.temperature_field.shape if hasattr(env.stress_field, 'temperature_field') else 'N/A'}")


def test_gradient_performance():
    """测试梯度计算性能"""
    print("\n" + "="*70)
    print("🧪 测试 A2: np.gradient 性能")
    print("="*70)
    
    # 不同大小的场
    sizes = [50, 100, 200, 500]
    
    print(f"\n{'网格大小':>12} {'单次耗时':>12} {'100次总耗时':>14}")
    print("-"*40)
    
    for size in sizes:
        field = np.random.rand(size, size).astype(np.float64)
        
        # 单次
        start = time.perf_counter()
        gx, gy = np.gradient(field)
        single = time.perf_counter() - start
        
        # 100次
        start = time.perf_counter()
        for _ in range(100):
            gx, gy = np.gradient(field)
        total = time.perf_counter() - start
        
        print(f"{size:>10}x{size:<1} {single*1000:>10.2f}ms {total*1000:>12.2f}ms")
        
        timer.record(f"np.gradient({size}x{size})", total)


def test_isf_diffusion():
    """测试ISF扩散性能"""
    print("\n" + "="*70)
    print("🧪 测试 A3: ISF扩散性能")
    print("="*70)
    
    from core.eoe.stigmergy_field import StigmergyField
    
    sizes = [50, 100, 200]
    
    print(f"\n{'网格大小':>12} {'扩散耗时':>12}")
    print("-"*30)
    
    for size in sizes:
        isf = StigmergyField(
            width=size,
            height=size,
            resolution=1.0,
            diffusion_rate=0.1,
            decay_rate=0.98
        )
        
        # 初始化一些值
        isf.field[size//2, size//2] = 100.0
        
        # 测试单次扩散
        start = time.perf_counter()
        for _ in range(50):
            isf.step()
        elapsed = time.perf_counter() - start
        
        print(f"{size:>10}x{size:<1} {elapsed*1000/50:>10.2f}ms")
        
        timer.record(f"ISF.diffusion({size}x{size})", elapsed/50)


# ============================================================================
# 测试 B: 传感器采样
# ============================================================================

def test_sensor_sampling():
    """测试传感器采样性能"""
    print("\n" + "="*70)
    print("🧪 测试 B: 传感器采样")
    print("="*70)
    
    from core.eoe.environment import Environment
    from core.eoe.agent import Agent
    
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        impedance_field_enabled=False,  # 简化避免边界问题
        stigmergy_field_enabled=True,
        stress_field_enabled=False,
        n_food=0
    )
    
    # 热启动
    for _ in range(5):
        env.step()
    
    # 创建多个Agent (在安全范围内)
    n_agents = 100
    
    print(f"\n测试 {n_agents} 个Agent的传感器采样...")
    
    # 使用安全的坐标范围
    np.random.seed(42)
    agents = [Agent(agent_id=i, x=10+np.random.uniform(0, 80), y=10+np.random.uniform(0, 80)) for i in range(n_agents)]
    
    start = time.perf_counter()
    sensor_results = []
    for agent in agents:
        sensor_values = env._compute_sensor_values(agent)
        sensor_results.append(sensor_values)
    elapsed = time.perf_counter() - start
    
    print(f"总耗时: {elapsed*1000:.2f}ms")
    print(f"平均每Agent: {elapsed/n_agents*1000:.3f}ms")
    print(f"吞吐量: {n_agents/elapsed:.0f} agents/sec")
    
    timer.record("传感器采样(100 agents)", elapsed)
    
    # 验证输出维度
    sensor_values = sensor_results[0]
    print(f"\n传感器输出维度: {len(sensor_values)}")


def test_gradient_lookup():
    """测试梯度查表性能"""
    print("\n" + "="*70)
    print("🧪 测试 B2: 梯度查表性能")
    print("="*70)
    
    from core.eoe.environment import Environment
    
    env = Environment(
        width=100, height=100,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True,
        n_food=0
    )
    
    # 热启动
    for _ in range(5):
        env.step()
    
    # 预计算的梯度矩阵
    print(f"\n梯度矩阵形状: {env.epf_grad_x.shape}")
    print(f"梯度矩阵大小: {env.epf_grad_x.nbytes / 1024:.1f} KB")
    
    # 测试查表性能
    n_lookups = 10000
    
    print(f"\n执行 {n_lookups} 次梯度查表...")
    
    # 随机坐标
    coords = np.random.randint(0, 100, (n_lookups, 2))
    
    start = time.perf_counter()
    for x, y in coords:
        _ = env.epf_grad_x[x, y]
        _ = env.epf_grad_y[x, y]
    elapsed = time.perf_counter() - start
    
    print(f"总耗时: {elapsed*1000:.2f}ms")
    print(f"平均每次: {elapsed/n_lookups*1e6:.2f}μs")
    print(f"吞吐量: {n_lookups/elapsed:.0f} lookups/sec")
    
    timer.record("梯度查表(10000次)", elapsed)


# ============================================================================
# 测试 C: 神经网络前向传播
# ============================================================================

def test_neural_network():
    """测试神经网络前向传播性能"""
    print("\n" + "="*70)
    print("🧪 测试 C: 神经网络前向传播")
    print("="*70)
    
    from core.eoe.agent import Agent
    
    n_agents = 100
    
    print(f"\n创建 {n_agents} 个Agent...")
    agents = [Agent(agent_id=i, x=50, y=50) for i in range(n_agents)]
    
    # 检查大脑结构
    sample_genome = agents[0].genome
    print(f"\n大脑结构:")
    print(f"  节点数: {len(sample_genome.nodes)}")
    print(f"  边数: {len(sample_genome.edges)}")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    
    # 创建输入
    input_dim = len(sample_genome.nodes)
    inputs = np.random.randn(n_agents, input_dim).astype(np.float64)
    
    # 单次前向传播 (串行)
    start = time.perf_counter()
    outputs = []
    for i, agent in enumerate(agents):
        out = agent.genome.forward(inputs[i])
        outputs.append(out)
    serial_time = time.perf_counter() - start
    
    print(f"串行处理 {n_agents} 个Agent:")
    print(f"  总耗时: {serial_time*1000:.2f}ms")
    print(f"  平均每个: {serial_time/n_agents*1000:.3f}ms")
    print(f"  吞吐量: {n_agents/serial_time:.0f} agents/sec")
    
    timer.record("神经网络(串行, 100 agents)", serial_time)


def test_genome_forward_variants():
    """测试不同大脑结构的前向传播"""
    print("\n" + "="*70)
    print("🧪 测试 C2: 不同大脑结构性能")
    print("="*70)
    
    from core.eoe.agent import Agent
    from core.eoe.node import NodeType
    
    # 简单测试现有大脑
    print(f"\n测试不同数量Agent的大脑前向传播...")
    
    for n_agents in [10, 50, 100, 200]:
        agents = [Agent(agent_id=i, x=50, y=50) for i in range(n_agents)]
        genome = agents[0].genome
        input_dim = len(genome.nodes)
        
        inputs = np.random.randn(n_agents, input_dim).astype(np.float64)
        
        start = time.perf_counter()
        for inp in inputs:
            _ = genome.forward(inp)
        elapsed = time.perf_counter() - start
        
        print(f"  {n_agents} agents, {input_dim} 节点: {elapsed*1000:.2f}ms ({n_agents/elapsed:.0f}/s)")
        
        timer.record(f"神经网络({n_agents} agents)", elapsed)


# ============================================================================
# GPU 可行性检查
# ============================================================================

def check_gpu():
    """检查GPU"""
    print("\n" + "="*70)
    print("🔍 GPU 可行性检查")
    print("="*70)
    
    # PyTorch
    try:
        import torch
        print(f"\n✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   设备数: {torch.cuda.device_count()}")
            
            # 测试GPU计算
            print(f"\n测试GPU计算...")
            
            # 矩阵乘法
            n = 1000
            a = torch.randn(n, n, device='cuda')
            b = torch.randn(n, n, device='cuda')
            
            start = time.perf_counter()
            for _ in range(100):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - start
            
            print(f"   1000x1000 矩阵乘法 x100: {gpu_time*1000:.2f}ms")
            print(f"   单次: {gpu_time/100*1000:.3f}ms")
            
            # 对比CPU
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            
            start = time.perf_counter()
            for _ in range(10):
                c = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.perf_counter() - start
            
            print(f"\n   CPU 10次: {cpu_time*1000:.2f}ms")
            print(f"   GPU加速比: {cpu_time/10 * 100 / gpu_time:.1f}x")
            
        else:
            print(f"⚠️ CUDA: 不可用")
    except ImportError:
        print(f"❌ PyTorch: 未安装")
    
    # CuPy
    try:
        import cupy as cp
        print(f"\n✅ CuPy: {cp.__version__}")
    except ImportError:
        print(f"⚠️ CuPy: 未安装")


# ============================================================================
# 主入口
# ============================================================================

def main():
    print("="*70)
    print("🚀 v13.0 性能探针测试")
    print("="*70)
    
    # GPU检查
    check_gpu()
    
    # A. 场物理更新
    test_field_updates()
    test_gradient_performance()
    test_isf_diffusion()
    
    # B. 传感器采样
    test_sensor_sampling()
    test_gradient_lookup()
    
    # C. 神经网络
    test_neural_network()
    test_genome_forward_variants()
    
    # 汇总报告
    timer.report()
    
    print("\n" + "="*70)
    print("💡 结论与建议")
    print("="*70)
    print("""
根据性能探针结果:

A. 场物理更新 (ISF扩散 + 梯度计算)
   - 瓶颈: np.gradient 在大网格上耗时
   - 优化: 使用PyTorch GPU加速

B. 传感器采样
   - 当前: Python循环 + 数组索引
   - 优化: torch.grid_sample 批量采样

C. 神经网络前向
   - 当前: for循环串行处理
   - 优化: 批量矩阵乘法 (Batched GEMM)
""")


if __name__ == "__main__":
    main()