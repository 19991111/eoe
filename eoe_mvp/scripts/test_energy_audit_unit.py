#!/usr/bin/env python
"""
能量审计模块单元测试
====================
不依赖完整环境,直接测试 EnergyAuditHook 的逻辑
"""

import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/eoe_mvp')

import torch
import numpy as np
from core.eoe.energy_audit import EnergyAuditHook, EnergySnapshot


def test_energy_snapshot():
    """测试能量快照计算"""
    print("Test 1: EnergySnapshot 计算")
    
    snapshot = EnergySnapshot(
        step=100,
        timestamp=0.0,
        epf_energy=1000.0,
        isf_energy=100.0,
        alive_agent_energy=500.0,
        dead_agent_energy=50.0,
        source_energy=200.0,
        pending_food_energy=30.0,
        prey_energy=20.0,
        corpse_biomass=10.0
    )
    
    expected_total = 1000 + 100 + 500 + 50 + 200 + 30 + 20 + 10
    assert abs(snapshot.total - expected_total) < 1e-6, f"Total mismatch: {snapshot.total} vs {expected_total}"
    
    print(f"  ✅ 快照总能量: {snapshot.total}")


def test_relative_error_calculation():
    """测试相对误差计算"""
    print("\nTest 2: 相对误差计算")
    
    # 创建模拟环境
    class MockEnv:
        def __init__(self):
            self.energy_field = MockEnergyField()
            self.stigmergy_field = None
    
    class MockEnergyField:
        def __init__(self):
            self.field = torch.ones(100, 100) * 0.5  # 5000 energy
    
    class MockAgents:
        def __init__(self):
            self.state = MockState()
            self.alive_mask = torch.ones(50, dtype=torch.bool)
            self.alive_mask[40:] = False  # 40 alive
    
    class MockState:
        def __init__(self):
            self.energies = torch.ones(50) * 10.0
    
    # 初始化审计钩子
    audit = EnergyAuditHook(
        tolerance=1e-5,
        audit_interval=1,  # 每步审计
        device='cpu',
        verbose=False
    )
    
    env = MockEnv()
    agents = MockAgents()
    
    # 初始化
    initial = audit.initialize(env, agents)
    print(f"  初始能量: {initial.total:.4f}")
    
    # 第一次审计 (步进后)
    agents.state.energies = torch.ones(50) * 9.0  # 消耗了能量
    result = audit.audit(env, agents, step=1)
    
    print(f"  当前能量: {audit.snapshots[-1].total:.4f}")
    print(f"  相对误差: {result['relative_error']:.2e}")
    
    # 验证守恒 (能量应该有变化但不是巨大变化)
    assert result['relative_error'] < 0.5, "Energy changed too much!"
    
    print(f"  ✅ 误差计算正确: {result['relative_error']:.2e}")


def test_float64_precision():
    """测试 float64 精度"""
    print("\nTest 3: float64 精度验证")
    
    # 创建大张量测试精度
    large_tensor = torch.ones(1000, 1000) * 0.1
    
    # float32 求和
    sum_f32 = large_tensor.float().sum().item()
    
    # float64 求和
    sum_f64 = large_tensor.to(torch.float64).sum().item()
    
    expected = 1000 * 1000 * 0.1
    
    print(f"  Expected: {expected}")
    print(f"  Float32:  {sum_f32} (误差: {abs(sum_f32 - expected)})")
    print(f"  Float64:  {sum_f64} (误差: {abs(sum_f64 - expected)})")
    
    f32_error = abs(sum_f32 - expected) / expected
    f64_error = abs(sum_f64 - expected) / expected
    
    assert f64_error < f32_error, "Float64 should be more precise!"
    
    print(f"  ✅ float64 精度更高: {f64_error:.2e} vs {f32_error:.2e}")


def test_conservation_detection():
    """测试能量守恒破缺检测"""
    print("\nTest 4: 能量守恒破缺检测")
    
    class MockEnv:
        def __init__(self, epf_energy):
            self.energy_field = MockEnergyField(epf_energy)
            self.stigmergy_field = None
    
    class MockEnergyField:
        def __init__(self, energy):
            self.field = torch.ones(10, 10) * energy
    
    class MockAgents:
        def __init__(self, energy):
            self.state = MockState(energy)
            self.alive_mask = torch.ones(10, dtype=torch.bool)
    
    class MockState:
        def __init__(self, energy):
            self.energies = torch.ones(10) * energy
    
    # 场景1: 正常守恒 (误差小)
    print("  场景1: 正常守恒")
    audit1 = EnergyAuditHook(tolerance=0.1, audit_interval=1, device='cpu', verbose=False)
    env1 = MockEnv(10.0)
    agents1 = MockAgents(5.0)
    audit1.initialize(env1, agents1)
    
    # 模拟微小变化
    agents1.state.energies = torch.ones(10) * 4.9  # 微弱变化
    result1 = audit1.audit(env1, agents1, step=1)
    print(f"    误差: {result1['relative_error']:.2e}, 守恒: {result1['is_conserved']}")
    assert result1['is_conserved'], "Should be conserved!"
    
    # 场景2: 能量泄漏 (误差大)
    print("  场景2: 能量泄漏")
    audit2 = EnergyAuditHook(tolerance=0.01, audit_interval=1, device='cpu', verbose=False)  # 1% 阈值
    env2 = MockEnv(10.0)
    agents2 = MockAgents(5.0)
    audit2.initialize(env2, agents2)
    
    # 模拟大泄漏 - 能量翻倍!
    agents2.state.energies = torch.ones(10) * 10.0  # 大幅增加!
    result2 = audit2.audit(env2, agents2, step=1)
    print(f"    误差: {result2['relative_error']:.2e}, 守恒: {result2['is_conserved']}")
    assert not result2['is_conserved'], "Should detect leak!"
    
    print(f"  ✅ 守恒检测正确")


def main():
    print("=" * 60)
    print("🔋 能量审计模块单元测试")
    print("=" * 60)
    
    test_energy_snapshot()
    test_relative_error_calculation()
    test_float64_precision()
    test_conservation_detection()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()