#!/usr/bin/env python3
"""
v14.1 演化机制测试脚本
测试5个未注册机制是否能正确加载和触发
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# 尝试导入manifest并查看注册的机制
print("=" * 60)
print("测试1: 检查机制注册")
print("=" * 60)

try:
    from core.eoe.manifest import PhysicsManifest
    
    # 创建manifest
    manifest = PhysicsManifest.from_yaml("full")
    
    print(f"\n已注册的物理法则:")
    for law_info in manifest.registry.list_laws():
        status = "✓" if law_info['enabled'] else "✗"
        law_class = law_info['class']
        name = law_class if isinstance(law_class, str) else law_class.__name__
        print(f"  {status} {law_info['name']}: {name}")
    
    # 获取演化机制
    evo_mechanisms = manifest.registry.get_evo_mechanisms()
    event_mechanisms = manifest.registry.get_event_mechanisms()
    
    print(f"\n每Step调用机制: {[m.name for m in evo_mechanisms]}")
    print(f"事件触发机制: {[m.name for m in event_mechanisms]}")
    
except Exception as e:
    print(f"❌ manifest加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试2: 检查BatchedAgents集成")
print("=" * 60)

try:
    from core.eoe.batched_agents import BatchedAgents, PoolConfig
    
    config = PoolConfig()
    config.HEBBIAN_ENABLED = True
    config.SUPERNODE_ENABLED = False  # 简化测试
    
    agents = BatchedAgents(
        initial_population=50,
        max_agents=500,
        device='cpu',  # 使用CPU避免CUDA依赖
        config=config
    )
    
    print(f"✅ BatchedAgents初始化完成")
    print(f"  演化机制: {[m.name for m in agents.evo_mechanisms]}")
    print(f"  事件机制: {[m.name for m in agents.event_mechanisms]}")
    
except Exception as e:
    print(f"❌ BatchedAgents初始化失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试3: 运行单步测试")
print("=" * 60)

try:
    # 运行几步测试
    for step in range(3):
        result = agents.step(dt=0.1)
        print(f"Step {step+1}: alive={result['n_alive']}, births={result['births']}, deaths={result['deaths']}")
    
    print(f"✅ 步进测试完成")
    
except Exception as e:
    print(f"❌ 步进测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)