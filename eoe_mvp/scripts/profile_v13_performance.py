#!/usr/bin/env python3
"""
v13.0 性能探针脚本
===============
对 Population.evolve_one_generation() 进行深度性能分析

测量三大核心部分:
A. 场物理更新 (Field Physics Update)
   - ISF卷积扩散
   - EPF/KIF/ISF的np.gradient计算
   
B. 传感器采样 (Sensor Sampling)
   - Agent查表获取场数值的频率和开销
   
C. 神经网络前向传播 (Neural Network Forward)
   - 所有Agent的大脑forward()计算

运行方式:
    python scripts/profile_v13_performance.py
    python scripts/profile_v13_performance.py --line-profile  # 逐行分析
    python scripts/profile_v13_performance.py --output profile.prof  # 生成prof文件
"""

import sys
import os
import time
import cProfile
import pstats
import io
from functools import wraps
from contextlib import contextmanager
import argparse

sys.path.insert(0, '.')

import numpy as np


# ============================================================================
# 性能探针上下文管理器
# ============================================================================

class PerformanceProbe:
    """性能探针 - 精确测量各模块耗时"""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        self._start_time = None
        self._current_section = None
    
    def start(self, section: str):
        """开始计时"""
        self._current_section = section
        self._start_time = time.perf_counter()
    
    def stop(self):
        """停止计时"""
        if self._current_section is None:
            return
        elapsed = time.perf_counter() - self._start_time
        if self._current_section not in self.timings:
            self.timings[self._current_section] = []
            self.call_counts[self._current_section] = 0
        self.timings[self._current_section].append(elapsed)
        self.call_counts[self._current_section] += 1
        self._current_section = None
        self._start_time = None
    
    @contextmanager
    def measure(self, section: str):
        """上下文管理器方式测量"""
        self.start(section)
        try:
            yield
        finally:
            self.stop()
    
    def report(self):
        """生成报告"""
        print("\n" + "="*70)
        print("📊 v13.0 性能探针报告")
        print("="*70)
        
        total_time = sum(sum(times) for times in self.timings.values())
        
        # 按总耗时排序
        sorted_sections = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )
        
        print(f"\n{'模块':<40} {'调用次数':>10} {'总耗时(s)':>12} {'占比':>8}")
        print("-"*70)
        
        for section, times in sorted_sections:
            total = sum(times)
            count = self.call_counts[section]
            pct = (total / total_time * 100) if total_time > 0 else 0
            print(f"{section:<40} {count:>10} {total:>12.4f} {pct:>7.1f}%")
        
        print("-"*70)
        print(f"{'总计':<40} {'':<10} {total_time:>12.4f} {'100.0%':>8}")
        
        return self.timings


# 全局探针实例
probe = PerformanceProbe()


# ============================================================================
# 探针注入点 - 猴子补丁 (Monkey Patching)
# ============================================================================

def instrument_module(module_name: str, func_names: list, prefix: str = "MODULE"):
    """为指定模块的函数注入探针"""
    import importlib
    module = importlib.import_module(module_name)
    
    for func_name in func_names:
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            
            @wraps(original_func)
            def make_instrumented(original, name, prefix):
                def instrumented(*args, **kwargs):
                    section = f"{prefix}:{name}"
                    with probe.measure(section):
                        return original(*args, **kwargs)
                return instrumented
            
            setattr(module, func_name, make_instrumented(original_func, func_name, prefix))


def inject_profiling_hooks():
    """注入性能探针钩子"""
    
    # -------------------------------------------------------------------------
    # A. 场物理更新探针
    # -------------------------------------------------------------------------
    from core.eoe.environment import Environment
    
    original_step = Environment.step
    @wraps(original_step)
    def instrumented_step(self, *args, **kwargs):
        with probe.measure("A.场更新:step()"):
            return original_step(self, *args, **kwargs)
    Environment.step = instrumented_step
    
    # ISF扩散
    original_isf_step = None
    try:
        from core.eoe.stigmergy_field import StigmergyField
        original_isf_step = StigmergyField.step
        @wraps(original_isf_step)
        def instrumented_isf_step(self, *args, **kwargs):
            with probe.measure("A.场更新:ISF扩散"):
                return original_isf_step(self, *args, **kwargs)
        StigmergyField.step = instrumented_isf_step
    except ImportError:
        pass
    
    # np.gradient 探针
    import numpy as np
    original_gradient = np.gradient
    gradient_call_count = [0]
    @wraps(original_gradient)
    def instrumented_gradient(*args, **kwargs):
        gradient_call_count[0] += 1
        with probe.measure(f"A.场更新:np.gradient #{gradient_call_count[0]}"):
            return original_gradient(*args, **kwargs)
    np.gradient = instrumented_gradient
    
    # -------------------------------------------------------------------------
    # B. 传感器采样探针
    # -------------------------------------------------------------------------
    from core.eoe.agent import Agent
    
    original_compute_sensors = Environment._compute_sensor_values
    @wraps(original_compute_sensors)
    def instrumented_compute_sensors(self, agent, *args, **kwargs):
        with probe.measure("B.传感器:_compute_sensor_values"):
            return original_compute_sensors(self, agent, *args, **kwargs)
    Environment._compute_sensor_values = instrumented_compute_sensors
    
    # -------------------------------------------------------------------------
    # C. 神经网络前向传播探针
    # -------------------------------------------------------------------------
    from core.eoe.genome import OperatorGenome
    
    original_forward = OperatorGenome.forward
    forward_call_count = [0]
    @wraps(original_forward)
    def instrumented_forward(self, *args, **kwargs):
        forward_call_count[0] += 1
        with probe.measure(f"C.神经网络:forward #{forward_call_count[0]}"):
            return original_forward(self, *args, **kwargs)
    OperatorGenome.forward = instrumented_forward
    
    print("✅ 性能探针钩子已注入")
    return probe


# ============================================================================
# 主测试函数
# ============================================================================

def run_performance_test(lifespan: int = 1500, population_size: int = 50):
    """运行性能测试"""
    from core.eoe.population import Population
    
    print(f"\n{'='*70}")
    print(f"🚀 启动v13.0性能探针测试")
    print(f"{'='*70}")
    print(f"参数: lifespan={lifespan}, population_size={population_size}")
    print(f"预计Agent总数: {population_size * lifespan:,} 次前向传播")
    
    # 注入探针
    inject_profiling_hooks()
    
    # 创建种群
    print(f"\n📦 创建种群...")
    pop = Population(
        population_size=population_size,
        lifespan=lifespan,
        n_food=0,  # 简化测试
    )
    
    # 手动启用v13.0场
    pop.environment.energy_field_enabled = True
    pop.environment.impedance_field_enabled = True
    pop.environment.stigmergy_field_enabled = True
    pop.environment.stress_field_enabled = False
    
    print(f"   种群规模: {pop.population_size}")
    print(f"   生命周期: {pop.lifespan}")
    print(f"   环境大小: {pop.environment.width}x{pop.environment.height}")
    
    # 运行前热启动 (让场初始化)
    print(f"\n🔥 热启动 (初始化场)...")
    for _ in range(10):
        pop.environment.step()
    
    # 重置探针
    global probe
    probe = PerformanceProbe()
    inject_profiling_hooks()
    
    # 运行一代演化
    print(f"\n⏱️  开始性能探针分析 (lifespan={lifespan})...")
    start_wall = time.time()
    
    pop.run()  # 执行完整生命周期
    
    wall_time = time.time() - start_wall
    
    # 生成报告
    print(f"\n⏱️  耗时: {wall_time:.2f}秒")
    
    timings = probe.report()
    
    # 额外统计
    print(f"\n{'='*70}")
    print(f"📈 详细统计")
    print(f"{'='*70}")
    
    # 计算三大类占比
    a_time = sum(sum(t) for k, t in timings.items() if k.startswith("A."))
    b_time = sum(sum(t) for k, t in timings.items() if k.startswith("B."))
    c_time = sum(sum(t) for k, t in timings.items() if k.startswith("C."))
    total = a_time + b_time + c_time
    
    if total > 0:
        print(f"\n🔍 三大模块耗时占比:")
        print(f"   A. 场物理更新: {a_time:.2f}s ({a_time/total*100:.1f}%)")
        print(f"   B. 传感器采样: {b_time:.2f}s ({b_time/total*100:.1f}%)")
        print(f"   C. 神经网络前向: {c_time:.2f}s ({c_time/total*100:.1f}%)")
    
    # 性能指标
    total_agents = population_size * lifespan
    print(f"\n⚡ 性能指标:")
    print(f"   总Agent-步数: {total_agents:,}")
    print(f"   平均每步耗时: {wall_time/lifespan*1000:.2f}ms")
    print(f"   平均每Agent-步: {wall_time/total_agents*1000:.3f}ms")
    
    return probe


# ============================================================================
# cProfile 版本
# ============================================================================

def run_cprofile_test(lifespan: int = 500, population_size: int = 20):
    """使用cProfile进行深度分析"""
    from core.eoe.population import Population
    
    print(f"\n{'='*70}")
    print(f"🔬 cProfile 深度分析")
    print(f"{'='*70}")
    print(f"参数: lifespan={lifespan}, population_size={population_size}")
    
    # 创建种群
    pop = Population(
        population_size=population_size,
        lifespan=lifespan,
        n_food=0,
        energy_field_enabled=True,
        impedance_field_enabled=True,
        stigmergy_field_enabled=True,
    )
    
    # 热启动
    for _ in range(5):
        pop.environment.step()
    
    # cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    pop.run()
    
    profiler.disable()
    
    # 输出统计
    print(f"\n📊 Top 30 耗时函数:")
    print("-"*70)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    # 按ncalls排序
    print(f"\n📊 Top 20 高频函数 (按调用次数):")
    print("-"*70)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('ncalls')
    ps.print_stats(20)
    print(s.getvalue())


# ============================================================================
# GPU可行性检查
# ============================================================================

def check_gpu_availability():
    """检查GPU可用性"""
    print(f"\n{'='*70}")
    print(f"🔍 GPU可用性检查")
    print(f"{'='*70}")
    
    gpu_info = {
        'torch': False,
        'cupy': False,
        'cuda_device': None,
    }
    
    # PyTorch
    try:
        import torch
        gpu_info['torch'] = True
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_info['cuda_device'] = torch.cuda.get_device_name(0)
            print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
            print(f"   设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   设备{i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"⚠️ CUDA: 不可用")
    except ImportError:
        print(f"❌ PyTorch: 未安装")
    
    # CuPy
    try:
        import cupy
        gpu_info['cupy'] = True
        print(f"✅ CuPy: {cupy.__version__}")
    except ImportError:
        print(f"⚠️ CuPy: 未安装 (可选)")
    
    return gpu_info


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='v13.0 性能探针')
    parser.add_argument('--lifespan', type=int, default=500, help='生命周期步数')
    parser.add_argument('--population', type=int, default=20, help='种群大小')
    parser.add_argument('--cprofile', action='store_true', help='使用cProfile深度分析')
    parser.add_argument('--line-profile', action='store_true', help='逐行分析(需要line_profiler)')
    parser.add_argument('--output', type=str, help='输出prof文件路径')
    
    args = parser.parse_args()
    
    # GPU检查
    gpu_info = check_gpu_availability()
    
    if args.cprofile:
        run_cprofile_test(args.lifespan, args.population)
    else:
        run_performance_test(args.lifespan, args.population)
    
    # 生成prof文件
    if args.output:
        print(f"\n💾 生成prof文件: {args.output}")
        import marshal
        # cProfile生成
        pass


if __name__ == "__main__":
    main()